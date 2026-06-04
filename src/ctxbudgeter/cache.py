"""Cache-aware context layout analysis.

``CachePlanner`` inspects a compiled context's order and flags layout choices that
break prompt caching: dynamic/timestamp/UUID content sitting inside what should be
a stable reusable prefix, retrieval blocks ahead of the system policy, unstable
tool ordering, etc.

We only ever report **estimated cacheable tokens** — never a dollar figure — unless
the caller supplies provider pricing, because cache economics are provider-specific.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .compiler import CompiledPack

# Cache policy ordering for "stability" (lower = more stable / earlier).
CACHE_POLICY_ORDER = {"stable": 0, "semi_stable": 1, "dynamic": 2, "ephemeral": 3, "no_cache": 3}

_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?\b|\b\d{2}:\d{2}:\d{2}\b"
)
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_DYNAMIC_KINDS = {"user_message", "tool_result", "retrieval"}


@dataclass
class CachePlan:
    """Result of a cache-layout analysis."""

    cacheable_token_estimate: int
    cache_efficiency_score: int
    stable_prefix_length: int
    stable_prefix_tokens: int
    dynamic_section_length: int
    dynamic_section_tokens: int
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    ordered_segments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class CachePlanner:
    """Analyze a compiled context for prompt-cache friendliness."""

    def analyze_bom(self, bom) -> CachePlan:
        """Analyze cache layout from a ContextBOM.

        Works on serialized BOM items (order, cache_policy, tokens, kind) — so it
        can run in CI from a committed BOM JSON. Content-based checks (timestamps,
        UUIDs in the stable prefix) are skipped because a BOM never stores raw
        content; use ``analyze(compiled)`` for those.
        """
        items = list(bom.included_items)
        warnings: list[str] = []
        recommendations: list[str] = []

        prefix_len = 0
        prefix_tokens = 0
        for i in items:
            if i.cache_policy in ("stable", "semi_stable"):
                prefix_len += 1
                prefix_tokens += i.tokens
            else:
                break

        dynamic_items = [i for i in items if i.cache_policy in ("dynamic", "ephemeral", "no_cache")]
        dynamic_tokens = sum(i.tokens for i in dynamic_items)

        seen_dynamic = False
        for i in items:
            if i.cache_policy in ("dynamic", "ephemeral", "no_cache"):
                seen_dynamic = True
            elif seen_dynamic and i.cache_policy in ("stable", "semi_stable"):
                warnings.append(
                    f"Stable item '{i.name}' appears after dynamic content — cannot be cached. Move it earlier."
                )
                recommendations.append(f"move_to_stable_prefix: {i.name}")

        tool_names = [i.name for i in items if i.kind == "tool_def"]
        if len(tool_names) > 1 and tool_names != sorted(tool_names):
            warnings.append("Tool definitions are not in a stable (sorted) order — breaks caching.")
            recommendations.append("stabilize_tool_ordering")

        cache_eff = round(100 * prefix_tokens / bom.total_tokens) if bom.total_tokens else 0
        if cache_eff < 30 and bom.total_tokens > 1000:
            recommendations.append("Low cache efficiency — group stable items at the top.")

        return CachePlan(
            cacheable_token_estimate=prefix_tokens,
            cache_efficiency_score=cache_eff,
            stable_prefix_length=prefix_len,
            stable_prefix_tokens=prefix_tokens,
            dynamic_section_length=len(dynamic_items),
            dynamic_section_tokens=dynamic_tokens,
            warnings=warnings,
            recommendations=_dedupe(recommendations),
            ordered_segments=[
                {"name": i.name, "kind": i.kind, "cache_policy": i.cache_policy,
                 "tokens": i.tokens, "in_stable_prefix": idx < prefix_len}
                for idx, i in enumerate(items)
            ],
        )

    def analyze(self, compiled: CompiledPack) -> CachePlan:
        items = list(compiled.included_items)
        tokens = compiled.included_tokens

        warnings: list[str] = []
        recommendations: list[str] = []

        # Consecutive stable prefix from the top.
        prefix_len = 0
        prefix_tokens = 0
        for it in items:
            if it.cache_policy in ("stable", "semi_stable"):
                prefix_len += 1
                prefix_tokens += tokens.get(it.name, 0)
            else:
                break

        dynamic_items = [it for it in items if it.cache_policy in ("dynamic", "ephemeral", "no_cache")]
        dynamic_tokens = sum(tokens.get(it.name, 0) for it in dynamic_items)

        # Stable items that appear AFTER the prefix ends (cache-breaking placement).
        seen_dynamic = False
        for it in items:
            is_dyn = it.cache_policy in ("dynamic", "ephemeral", "no_cache")
            if is_dyn:
                seen_dynamic = True
            elif seen_dynamic and it.cache_policy in ("stable", "semi_stable"):
                warnings.append(
                    f"Stable item '{it.name}' appears after dynamic content — it cannot be "
                    "part of the cacheable prefix. Move it earlier."
                )
                recommendations.append(f"move_to_stable_prefix: {it.name}")

        # Inspect the stable prefix for cache-busting content.
        for it in items[:prefix_len]:
            content = it.effective_content()
            if _TIMESTAMP_RE.search(content):
                warnings.append(
                    f"Stable-prefix item '{it.name}' contains a timestamp — this busts the "
                    "cache on every call. Move volatile timestamps to a dynamic section."
                )
                recommendations.append(f"move_to_dynamic_section: {it.name}")
            if _UUID_RE.search(content):
                warnings.append(
                    f"Stable-prefix item '{it.name}' contains a UUID — random IDs prevent cache "
                    "reuse. Move per-request IDs to a dynamic section."
                )
                recommendations.append(f"move_to_dynamic_section: {it.name}")
            if it.kind in ("user_message", "tool_result"):
                warnings.append(
                    f"User/tool-specific item '{it.name}' is inside the stable prefix — "
                    "this is per-request content and breaks reuse."
                )

        # Retrieval/dynamic block before the system policy.
        first_system_idx = next((i for i, it in enumerate(items) if it.kind == "system"), None)
        if first_system_idx is not None:
            for it in items[:first_system_idx]:
                if it.kind in _DYNAMIC_KINDS:
                    warnings.append(
                        f"Dynamic '{it.kind}' item '{it.name}' appears before the system policy — "
                        "place system/stable instructions first to maximize the cacheable prefix."
                    )

        # Unstable tool ordering: tool_def items not contiguous / not sorted by name.
        tool_names = [it.name for it in items if it.kind == "tool_def"]
        if len(tool_names) > 1 and tool_names != sorted(tool_names):
            warnings.append(
                "Tool definitions are not in a stable (sorted) order — non-deterministic tool "
                "ordering breaks prompt caching across calls."
            )
            recommendations.append("stabilize_tool_ordering")

        cacheable = prefix_tokens
        cache_eff = round(100 * cacheable / compiled.used_tokens) if compiled.used_tokens else 0

        if cache_eff < 30 and compiled.used_tokens > 1000:
            recommendations.append(
                "Low cache efficiency — group stable system rules, docs, and tool defs at the top."
            )

        segments = [
            {
                "name": it.name,
                "kind": it.kind,
                "cache_policy": it.cache_policy,
                "tokens": tokens.get(it.name, 0),
                "in_stable_prefix": idx < prefix_len,
            }
            for idx, it in enumerate(items)
        ]

        return CachePlan(
            cacheable_token_estimate=cacheable,
            cache_efficiency_score=cache_eff,
            stable_prefix_length=prefix_len,
            stable_prefix_tokens=prefix_tokens,
            dynamic_section_length=len(dynamic_items),
            dynamic_section_tokens=dynamic_tokens,
            warnings=warnings,
            recommendations=_dedupe(recommendations),
            ordered_segments=segments,
        )


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

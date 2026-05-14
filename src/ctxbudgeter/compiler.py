"""Deterministic context compilation.

Algorithm:
1. Resolve References (lazy / JIT loading), keeping those whose estimated cost could
   plausibly fit. Loader failures are caught and the reference is excluded with a reason.
2. Required items go in first (in declaration order), with compression / truncation
   if they don't fit.
3. Optional items are ranked by `score_item` and packed greedily.
4. Sensitivity policy (allow / warn / refuse / redact) is applied before output.
5. Final prompt order: stable → dynamic → ephemeral; within each by (-priority, name).
6. The cacheable prefix is the consecutive run of stable items at the top.

Tokens for each item = tokenizer.count(content) + sum(attachment.estimated_tokens).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Union

from .content import (
    attachment_estimated_tokens,
)
from .item import ContextItem
from .reference import Reference
from .scoring import score_item
from .tokenizer import TokenCounter

DecisionStatus = Literal["included", "excluded", "compressed", "truncated", "redacted"]
SecretPolicy = Literal["allow", "warn", "refuse", "redact"]

SyncCompressor = Callable[[ContextItem, int], str]
AsyncCompressor = Callable[[ContextItem, int], Awaitable[str]]
Compressor = Union[SyncCompressor, AsyncCompressor]
"""User-supplied compression hook: (item, target_tokens) -> compressed text (sync or async)."""

REDACTED_PLACEHOLDER = "[REDACTED — sensitivity=secret]"


class BudgetExceededError(ValueError):
    """Raised when a required item cannot fit in the budget and no fallback is allowed."""


class SecretContentError(ValueError):
    """Raised when sensitivity policy refuses to include a `secret`-tagged item."""


@dataclass(frozen=True)
class ItemDecision:
    """One row of the compilation audit trail."""

    name: str
    status: DecisionStatus
    reason: str
    score: float           # ranking score; +inf for required items
    tokens: int            # final tokens used (after possible compression / truncation)
    original_tokens: int   # tokens of original content (text + attachments)
    kind: str
    cache_policy: str
    priority: int
    required: bool
    sensitivity: str = "internal"
    source: str | None = None


@dataclass
class CompilerConfig:
    """Tunable knobs for the compiler."""

    weights: dict[str, float] = field(default_factory=dict)
    allow_truncation: bool = False
    truncation_marker: str = "\n... [truncated]"
    compressor: Compressor | None = None
    compression_retry: bool = True
    """If True, retry compression with a tighter target when the first attempt overshoots."""
    secret_policy: SecretPolicy = "warn"
    """How to handle items tagged sensitivity='secret':

    - allow:  include silently
    - warn:   include but flag in report and deduct from health (default)
    - redact: include with content replaced by REDACTED_PLACEHOLDER
    - refuse: raise SecretContentError if any secret item would be included
    """


@dataclass
class CompiledPack:
    """The result of compiling a ContextPack."""

    model: str
    token_budget: int
    reserved_output_tokens: int
    used_tokens: int
    cacheable_prefix_tokens: int
    health_score: int
    health_breakdown: dict[str, int]
    decisions: list[ItemDecision]
    included_items: list[ContextItem]
    items_by_name: dict[str, ContextItem]
    included_tokens: dict[str, int]
    tokenizer_backend: str
    input_order: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    """Sensitivity flags, compression overshoots, loader failures — anything a
    human reviewer would want to see before signing off."""

    @property
    def available_tokens(self) -> int:
        return self.token_budget - self.reserved_output_tokens

    @property
    def utilization(self) -> float:
        return (self.used_tokens / self.available_tokens) if self.available_tokens > 0 else 0.0

    @property
    def has_secrets(self) -> bool:
        return any(it.sensitivity == "secret" for it in self.included_items)

    def included_in_input_order(self) -> list[ContextItem]:
        included = {it.name for it in self.included_items}
        return [self.items_by_name[n] for n in self.input_order if n in included]

    def content_for(self, item_name: str) -> str:
        for it in self.included_items:
            if it.name == item_name:
                return it.effective_content()
        raise KeyError(item_name)

    @property
    def prompt(self) -> str:
        return self.as_text()

    def as_text(self) -> str:
        """Concatenated plain-text view of the included items, in compiled order.

        Debugging only — use the framework adapters for real LLM calls.
        """
        return "\n\n".join(it.effective_content() for it in self.included_items)

    def report(self, format: str = "text") -> str:
        from . import report as _report

        if format == "markdown":
            return _report.to_markdown(self)
        if format == "json":
            return _report.to_json(self)
        return _report.to_text(self)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "token_budget": self.token_budget,
            "reserved_output_tokens": self.reserved_output_tokens,
            "available_tokens": self.available_tokens,
            "used_tokens": self.used_tokens,
            "utilization": round(self.utilization, 3),
            "cacheable_prefix_tokens": self.cacheable_prefix_tokens,
            "health_score": self.health_score,
            "health_breakdown": dict(self.health_breakdown),
            "tokenizer_backend": self.tokenizer_backend,
            "warnings": list(self.warnings),
            "decisions": [
                {
                    "name": d.name,
                    "status": d.status,
                    "reason": d.reason,
                    "score": ("required" if d.score == float("inf") else round(d.score, 2)),
                    "tokens": d.tokens,
                    "original_tokens": d.original_tokens,
                    "kind": d.kind,
                    "cache_policy": d.cache_policy,
                    "priority": d.priority,
                    "required": d.required,
                    "sensitivity": d.sensitivity,
                    "source": d.source,
                }
                for d in self.decisions
            ],
            "included_order": [it.name for it in self.included_items],
            "input_order": list(self.input_order),
        }


# ----- internal helpers --------------------------------------------------------------


def _item_token_cost(item: ContextItem, counter: TokenCounter) -> tuple[int, int]:
    """Return (text_tokens, attachment_tokens) for an item."""
    text_tokens = counter.count(item.content)
    att_tokens = sum(
        attachment_estimated_tokens(a, tokenizer_count=counter.count)
        for a in item.attachments
    )
    return text_tokens, att_tokens


def _human_reason(
    kind: Literal["included", "fits", "compressed", "truncated", "excluded_size", "excluded_size_high_pri", "redacted", "ref_failed"],
    *,
    score: float | None = None,
    orig: int | None = None,
    final: int | None = None,
    remaining: int | None = None,
    priority: int | None = None,
) -> str:
    if kind == "included":
        return "required — guaranteed inclusion"
    if kind == "fits":
        if priority is not None and priority >= 80:
            return "high-priority and fits within budget"
        if priority is not None and priority <= 30:
            return "low-priority but cheap enough to keep"
        return "fits within remaining budget"
    if kind == "compressed":
        return f"compressed to fit ({orig:,} → {final:,} tokens)"
    if kind == "truncated":
        return f"truncated to fit ({orig:,} → {final:,} tokens)"
    if kind == "excluded_size_high_pri":
        return f"high priority but token-heavy — {orig:,} tokens exceeds remaining {remaining:,}"
    if kind == "excluded_size":
        if priority is not None and priority <= 30:
            return f"token-heavy and low priority — {orig:,} tokens, score {score:.0f}"
        return f"token-heavy — {orig:,} tokens exceeds remaining {remaining:,}"
    if kind == "redacted":
        return "included but content redacted (sensitivity=secret, policy=redact)"
    if kind == "ref_failed":
        return "reference loader failed"
    return ""


def _try_compress_sync(
    item: ContextItem,
    target_tokens: int,
    counter: TokenCounter,
    config: CompilerConfig,
    warnings: list[str],
) -> tuple[str, int] | None:
    """Sync compression attempt with retry-on-overshoot."""
    # 1. Pre-computed compressed_content takes priority
    if item.compressed_content is not None:
        ctoks = counter.count(item.compressed_content)
        return (item.compressed_content, ctoks)
    # 2. User compressor
    if config.compressor is None or not item.compressible:
        return None
    fn = config.compressor
    target = max(1, target_tokens)
    try:
        text = fn(item, target)
        if inspect.isawaitable(text):
            warnings.append(
                f"compressor for '{item.name}' returned a coroutine — use pack.acompile() for async compressors"
            )
            try:
                text.close()  # type: ignore[union-attr]
            except Exception:
                pass
            return None
    except Exception as e:
        warnings.append(f"compressor for '{item.name}' raised {type(e).__name__}: {e}")
        return None
    if not isinstance(text, str):
        return None
    toks = counter.count(text)
    # Retry once on overshoot
    if toks > target_tokens and config.compression_retry:
        tighter = max(1, int(target_tokens * 0.6))
        try:
            text2 = fn(item, tighter)
            if inspect.isawaitable(text2):
                return (text, toks)
        except Exception:
            return (text, toks)
        if isinstance(text2, str):
            toks2 = counter.count(text2)
            if toks2 <= target_tokens or toks2 < toks:
                return (text2, toks2)
    return (text, toks)


async def _try_compress_async(
    item: ContextItem,
    target_tokens: int,
    counter: TokenCounter,
    config: CompilerConfig,
    warnings: list[str],
) -> tuple[str, int] | None:
    """Async-aware compression attempt. Falls back to sync path for sync compressors."""
    if item.compressed_content is not None:
        ctoks = counter.count(item.compressed_content)
        return (item.compressed_content, ctoks)
    if config.compressor is None or not item.compressible:
        return None
    fn = config.compressor
    target = max(1, target_tokens)
    try:
        result = fn(item, target)
        if inspect.isawaitable(result):
            text = await result
        else:
            text = result
    except Exception as e:
        warnings.append(f"compressor for '{item.name}' raised {type(e).__name__}: {e}")
        return None
    if not isinstance(text, str):
        return None
    toks = counter.count(text)
    if toks > target_tokens and config.compression_retry:
        tighter = max(1, int(target_tokens * 0.6))
        try:
            result2 = fn(item, tighter)
            text2 = await result2 if inspect.isawaitable(result2) else result2
        except Exception:
            return (text, toks)
        if isinstance(text2, str):
            toks2 = counter.count(text2)
            if toks2 <= target_tokens or toks2 < toks:
                return (text2, toks2)
    return (text, toks)


def _truncate_to_budget(
    text: str, target_tokens: int, counter: TokenCounter, marker: str
) -> tuple[str, int]:
    if target_tokens <= 0 or not text:
        return ("", 0)
    marker_tokens = counter.count(marker)
    if marker_tokens >= target_tokens:
        return ("", 0)
    lo, hi = 0, len(text)
    best_text, best_tokens = "", 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid] + marker
        ctoks = counter.count(candidate)
        if ctoks <= target_tokens:
            best_text, best_tokens = candidate, ctoks
            lo = mid + 1
        else:
            hi = mid - 1
    return (best_text, best_tokens)


def _compute_health(
    decisions: list[ItemDecision],
    used: int,
    available: int,
    cacheable_prefix_tokens: int,
    secret_count: int,
) -> tuple[int, dict[str, int]]:
    """Compute health score with explicit breakdown so users can audit deductions."""
    breakdown: dict[str, int] = {}

    if used == 0:
        breakdown["empty_pack"] = -100
        return 0, breakdown

    score = 100

    required_compressed = sum(1 for d in decisions if d.required and d.status == "compressed")
    if required_compressed:
        delta = -5 * required_compressed
        breakdown["required_compressed"] = delta
        score += delta

    required_truncated = sum(1 for d in decisions if d.required and d.status == "truncated")
    if required_truncated:
        delta = -10 * required_truncated
        breakdown["required_truncated"] = delta
        score += delta

    util = used / available if available > 0 else 0.0
    if util < 0.20:
        breakdown["under_utilized"] = -5
        score -= 5
    elif util < 0.50:
        breakdown["lightly_utilized"] = -2
        score -= 2
    elif util > 0.95:
        breakdown["near_overflow"] = -8
        score -= 8
    elif util > 0.85:
        breakdown["high_utilization"] = -3
        score -= 3

    high_excluded = sum(1 for d in decisions if d.status == "excluded" and d.priority >= 80)
    if high_excluded:
        delta = -min(20, high_excluded * 5)
        breakdown["high_priority_excluded"] = delta
        score += delta

    if cacheable_prefix_tokens >= 1024:
        breakdown["cacheable_prefix_bonus"] = 5
        score += 5
    elif cacheable_prefix_tokens >= 512:
        breakdown["cacheable_prefix_bonus"] = 2
        score += 2

    if secret_count:
        delta = -10 * secret_count
        breakdown["secrets_included"] = delta
        score += delta

    return max(0, min(100, score)), breakdown


# ----- reference resolution ----------------------------------------------------------


def _resolve_references_sync(
    references: list[Reference],
    available_tokens: int,
    counter: TokenCounter,
    warnings: list[str],
    excluded_decisions: list[ItemDecision],
) -> list[ContextItem]:
    """Resolve sync references that could plausibly fit. Failures → excluded decisions."""
    return _resolve_references_common(
        references,
        available_tokens,
        counter,
        warnings,
        excluded_decisions,
        async_mode=False,
    )


async def _resolve_references_async(
    references: list[Reference],
    available_tokens: int,
    counter: TokenCounter,
    warnings: list[str],
    excluded_decisions: list[ItemDecision],
) -> list[ContextItem]:
    """Async reference resolution (works for both sync and async loaders)."""
    resolved: list[ContextItem] = []
    # Sort to prioritize required, then by priority
    sorted_refs = sorted(references, key=lambda r: (not r.required, -r.priority, r.name))

    tasks: list[tuple[Reference, asyncio.Task]] = []
    sync_results: list[tuple[Reference, str]] = []
    sync_failures: list[tuple[Reference, str]] = []
    skipped_by_estimate: list[Reference] = []

    rough_budget_used = 0
    for ref in sorted_refs:
        est = ref.estimated_tokens
        # Skip refs that obviously won't fit
        if est and not ref.required and rough_budget_used + est > available_tokens:
            skipped_by_estimate.append(ref)
            continue
        if ref.loader is None:
            sync_failures.append((ref, "no loader provided"))
            continue
        try:
            result = ref.loader(ref)
        except Exception as e:
            sync_failures.append((ref, f"{type(e).__name__}: {e}"))
            continue
        if inspect.isawaitable(result):
            tasks.append((ref, asyncio.create_task(result)))
        elif isinstance(result, str):
            sync_results.append((ref, result))
            rough_budget_used += est or counter.count(result)
        else:
            sync_failures.append((ref, "loader returned non-string"))

    # Await async tasks concurrently
    for ref, task in tasks:
        try:
            content = await task
            if isinstance(content, str):
                sync_results.append((ref, content))
            else:
                sync_failures.append((ref, "async loader returned non-string"))
        except Exception as e:
            sync_failures.append((ref, f"{type(e).__name__}: {e}"))

    for ref, content in sync_results:
        resolved.append(ref.to_item(content))
    for ref, msg in sync_failures:
        warnings.append(f"reference '{ref.name}' failed: {msg}")
        excluded_decisions.append(
            ItemDecision(
                name=ref.name,
                status="excluded",
                reason=f"reference loader failed: {msg}",
                score=float("-inf"),
                tokens=0,
                original_tokens=ref.estimated_tokens,
                kind=ref.kind,
                cache_policy=ref.cache_policy,
                priority=ref.priority,
                required=ref.required,
                sensitivity=ref.sensitivity,
                source=ref.location,
            )
        )
    for ref in skipped_by_estimate:
        excluded_decisions.append(
            ItemDecision(
                name=ref.name,
                status="excluded",
                reason=f"reference skipped — estimated {ref.estimated_tokens:,} tokens won't fit remaining budget",
                score=float("-inf"),
                tokens=0,
                original_tokens=ref.estimated_tokens,
                kind=ref.kind,
                cache_policy=ref.cache_policy,
                priority=ref.priority,
                required=ref.required,
                sensitivity=ref.sensitivity,
                source=ref.location,
            )
        )

    return resolved


def _resolve_references_common(
    references: list[Reference],
    available_tokens: int,
    counter: TokenCounter,
    warnings: list[str],
    excluded_decisions: list[ItemDecision],
    *,
    async_mode: bool,
) -> list[ContextItem]:
    """Shared sync code path for sync compile (rejects async loaders)."""
    if async_mode:
        raise RuntimeError("use _resolve_references_async instead")
    resolved: list[ContextItem] = []
    sorted_refs = sorted(references, key=lambda r: (not r.required, -r.priority, r.name))
    rough_budget_used = 0
    for ref in sorted_refs:
        est = ref.estimated_tokens
        if est and not ref.required and rough_budget_used + est > available_tokens:
            excluded_decisions.append(
                ItemDecision(
                    name=ref.name,
                    status="excluded",
                    reason=f"reference skipped — estimated {est:,} tokens won't fit remaining budget",
                    score=float("-inf"),
                    tokens=0,
                    original_tokens=est,
                    kind=ref.kind,
                    cache_policy=ref.cache_policy,
                    priority=ref.priority,
                    required=ref.required,
                    sensitivity=ref.sensitivity,
                    source=ref.location,
                )
            )
            continue
        if ref.loader is None:
            warnings.append(f"reference '{ref.name}' has no loader")
            excluded_decisions.append(
                ItemDecision(
                    name=ref.name, status="excluded", reason="no loader provided",
                    score=float("-inf"), tokens=0, original_tokens=est,
                    kind=ref.kind, cache_policy=ref.cache_policy,
                    priority=ref.priority, required=ref.required,
                    sensitivity=ref.sensitivity, source=ref.location,
                )
            )
            continue
        try:
            result = ref.loader(ref)
        except Exception as e:
            warnings.append(f"reference '{ref.name}' loader raised {type(e).__name__}: {e}")
            excluded_decisions.append(
                ItemDecision(
                    name=ref.name, status="excluded",
                    reason=f"reference loader failed: {type(e).__name__}: {e}",
                    score=float("-inf"), tokens=0, original_tokens=est,
                    kind=ref.kind, cache_policy=ref.cache_policy,
                    priority=ref.priority, required=ref.required,
                    sensitivity=ref.sensitivity, source=ref.location,
                )
            )
            continue
        if inspect.isawaitable(result):
            warnings.append(
                f"reference '{ref.name}' loader is async; use pack.acompile() for async references"
            )
            try:
                # Tear down the unawaited coroutine to avoid warnings
                result.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            excluded_decisions.append(
                ItemDecision(
                    name=ref.name, status="excluded",
                    reason="async loader requires pack.acompile()",
                    score=float("-inf"), tokens=0, original_tokens=est,
                    kind=ref.kind, cache_policy=ref.cache_policy,
                    priority=ref.priority, required=ref.required,
                    sensitivity=ref.sensitivity, source=ref.location,
                )
            )
            continue
        if isinstance(result, str):
            resolved.append(ref.to_item(result))
            rough_budget_used += est or counter.count(result)
    return resolved


# ----- compilation core -------------------------------------------------------------


@dataclass
class _CompileState:
    """Mutable state threaded through the compile pipeline."""

    decisions: list[ItemDecision]
    used_text: dict[str, str]
    used_tokens: dict[str, int]
    used: int
    warnings: list[str]


def _process_required(
    state: _CompileState,
    items: list[ContextItem],
    tokens_for: dict[str, int],
    available: int,
    counter: TokenCounter,
    config: CompilerConfig,
    *,
    async_compress: Callable | None = None,
) -> None:
    """Process required items with compress/truncate fallbacks. Mutates state."""
    for item in items:
        orig = tokens_for[item.name]
        remaining = available - state.used
        if orig <= remaining:
            state.used += orig
            state.used_text[item.name] = item.content
            state.used_tokens[item.name] = orig
            state.decisions.append(
                ItemDecision(
                    name=item.name, status="included",
                    reason=_human_reason("included"),
                    score=float("inf"), tokens=orig, original_tokens=orig,
                    kind=item.kind, cache_policy=item.cache_policy,
                    priority=item.priority, required=True,
                    sensitivity=item.sensitivity, source=item.source,
                )
            )
            continue
        compressed = _try_compress_sync(item, remaining, counter, config, state.warnings)
        if compressed is not None and compressed[1] <= remaining:
            text, toks = compressed
            state.used += toks
            state.used_text[item.name] = text
            state.used_tokens[item.name] = toks
            state.decisions.append(
                ItemDecision(
                    name=item.name, status="compressed",
                    reason=_human_reason("compressed", orig=orig, final=toks),
                    score=float("inf"), tokens=toks, original_tokens=orig,
                    kind=item.kind, cache_policy=item.cache_policy,
                    priority=item.priority, required=True,
                    sensitivity=item.sensitivity, source=item.source,
                )
            )
            continue
        if config.allow_truncation:
            text, toks = _truncate_to_budget(
                item.content, remaining, counter, config.truncation_marker
            )
            if toks > 0:
                state.used += toks
                state.used_text[item.name] = text
                state.used_tokens[item.name] = toks
                state.decisions.append(
                    ItemDecision(
                        name=item.name, status="truncated",
                        reason=_human_reason("truncated", orig=orig, final=toks),
                        score=float("inf"), tokens=toks, original_tokens=orig,
                        kind=item.kind, cache_policy=item.cache_policy,
                        priority=item.priority, required=True,
                        sensitivity=item.sensitivity, source=item.source,
                    )
                )
                continue
        raise BudgetExceededError(
            f"Required item '{item.name}' ({orig:,} tokens) cannot fit in remaining "
            f"budget ({remaining:,} tokens). Mark it compressible=True with a compressor, "
            "supply compressed_content, or enable allow_truncation in CompilerConfig."
        )


def _process_optional(
    state: _CompileState,
    items: list[ContextItem],
    tokens_for: dict[str, int],
    available: int,
    counter: TokenCounter,
    config: CompilerConfig,
) -> None:
    """Process optional items greedily by score. Mutates state."""
    scored: list[tuple[float, ContextItem]] = [
        (score_item(it, tokens_for[it.name], available, config.weights), it) for it in items
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    for sc, item in scored:
        orig = tokens_for[item.name]
        remaining = available - state.used
        if orig <= remaining:
            state.used += orig
            state.used_text[item.name] = item.content
            state.used_tokens[item.name] = orig
            state.decisions.append(
                ItemDecision(
                    name=item.name, status="included",
                    reason=_human_reason("fits", priority=item.priority),
                    score=sc, tokens=orig, original_tokens=orig,
                    kind=item.kind, cache_policy=item.cache_policy,
                    priority=item.priority, required=False,
                    sensitivity=item.sensitivity, source=item.source,
                )
            )
            continue
        compressed = _try_compress_sync(item, remaining, counter, config, state.warnings)
        if compressed is not None and 0 < compressed[1] <= remaining:
            text, toks = compressed
            state.used += toks
            state.used_text[item.name] = text
            state.used_tokens[item.name] = toks
            state.decisions.append(
                ItemDecision(
                    name=item.name, status="compressed",
                    reason=_human_reason("compressed", orig=orig, final=toks),
                    score=sc, tokens=toks, original_tokens=orig,
                    kind=item.kind, cache_policy=item.cache_policy,
                    priority=item.priority, required=False,
                    sensitivity=item.sensitivity, source=item.source,
                )
            )
            continue
        # Exclude with a human-readable reason
        kind_key = "excluded_size_high_pri" if item.priority >= 80 else "excluded_size"
        state.decisions.append(
            ItemDecision(
                name=item.name, status="excluded",
                reason=_human_reason(kind_key, orig=orig, remaining=remaining, score=sc, priority=item.priority),
                score=sc, tokens=0, original_tokens=orig,
                kind=item.kind, cache_policy=item.cache_policy,
                priority=item.priority, required=False,
                sensitivity=item.sensitivity, source=item.source,
            )
        )


def _finalize(
    items_by_name: dict[str, ContextItem],
    state: _CompileState,
    config: CompilerConfig,
    counter: TokenCounter,
    model: str,
    token_budget: int,
    reserved_output_tokens: int,
    input_order: list[str],
    pre_excluded: list[ItemDecision],
) -> CompiledPack:
    """Apply sensitivity policy, order for caching, compute health, package result."""
    # Combine pre-existing exclusions (from reference resolution) with the rest
    all_decisions = state.decisions + pre_excluded

    # Sensitivity policy enforcement
    included_names = [
        d.name for d in all_decisions if d.status in ("included", "compressed", "truncated")
    ]
    secret_names = [
        n for n in included_names
        if items_by_name[n].sensitivity == "secret"
    ]
    secret_count = 0
    if secret_names:
        if config.secret_policy == "refuse":
            raise SecretContentError(
                f"Sensitivity policy is 'refuse' and {len(secret_names)} secret item(s) "
                f"would be included: {secret_names}. "
                "Change CompilerConfig.secret_policy or remove the items."
            )
        if config.secret_policy == "redact":
            # Convert to redacted: replace content, status -> "redacted"
            new_decisions: list[ItemDecision] = []
            for d in all_decisions:
                if d.name in secret_names and d.status in ("included", "compressed", "truncated"):
                    state.used_text[d.name] = REDACTED_PLACEHOLDER
                    new_tokens = counter.count(REDACTED_PLACEHOLDER)
                    state.used -= state.used_tokens[d.name]
                    state.used += new_tokens
                    state.used_tokens[d.name] = new_tokens
                    new_decisions.append(
                        ItemDecision(
                            name=d.name, status="redacted",
                            reason=_human_reason("redacted"),
                            score=d.score, tokens=new_tokens, original_tokens=d.original_tokens,
                            kind=d.kind, cache_policy=d.cache_policy,
                            priority=d.priority, required=d.required,
                            sensitivity=d.sensitivity, source=d.source,
                        )
                    )
                else:
                    new_decisions.append(d)
            all_decisions = new_decisions
        elif config.secret_policy == "warn":
            state.warnings.extend(
                f"sensitivity=secret item included: '{n}' "
                "(set CompilerConfig.secret_policy='refuse' or 'redact' to enforce)"
                for n in secret_names
            )
            secret_count = len(secret_names)
        elif config.secret_policy == "allow":
            secret_count = 0  # no penalty

    # Build the final ordered list of included items
    included_names = [
        d.name for d in all_decisions
        if d.status in ("included", "compressed", "truncated", "redacted")
    ]
    cache_rank = {"stable": 0, "dynamic": 1, "ephemeral": 2}
    ordered_names = sorted(
        included_names,
        key=lambda n: (
            cache_rank.get(items_by_name[n].cache_policy, 1),
            -items_by_name[n].priority,
            n,
        ),
    )
    finalized: list[ContextItem] = []
    for name in ordered_names:
        it = items_by_name[name]
        used_content = state.used_text[name]
        if used_content == it.content:
            finalized.append(it)
        else:
            finalized.append(it.model_copy(update={"compressed_content": used_content}))

    cacheable_prefix_tokens = 0
    for it in finalized:
        if it.cache_policy == "stable":
            cacheable_prefix_tokens += state.used_tokens[it.name]
        else:
            break

    available = token_budget - reserved_output_tokens
    health, breakdown = _compute_health(
        all_decisions, state.used, available, cacheable_prefix_tokens, secret_count
    )

    return CompiledPack(
        model=model,
        token_budget=token_budget,
        reserved_output_tokens=reserved_output_tokens,
        used_tokens=state.used,
        cacheable_prefix_tokens=cacheable_prefix_tokens,
        health_score=health,
        health_breakdown=breakdown,
        decisions=all_decisions,
        included_items=finalized,
        items_by_name=items_by_name,
        included_tokens=state.used_tokens,
        tokenizer_backend=counter.backend,
        input_order=input_order,
        warnings=state.warnings,
    )


# ----- public entry points ----------------------------------------------------------


def compile_items(
    items: list[ContextItem],
    *,
    model: str,
    token_budget: int,
    reserved_output_tokens: int,
    config: CompilerConfig | None = None,
    counter: TokenCounter | None = None,
    references: list[Reference] | None = None,
) -> CompiledPack:
    """Synchronous compilation. Async loaders/compressors → use `acompile_items`."""
    if token_budget <= 0:
        raise ValueError("token_budget must be > 0")
    if reserved_output_tokens < 0:
        raise ValueError("reserved_output_tokens must be >= 0")
    if reserved_output_tokens >= token_budget:
        raise ValueError("reserved_output_tokens must be less than token_budget")

    cfg = config or CompilerConfig()
    cnt = counter or TokenCounter(model)
    available = token_budget - reserved_output_tokens

    warnings: list[str] = []
    pre_excluded: list[ItemDecision] = []
    resolved = _resolve_references_sync(
        references or [], available, cnt, warnings, pre_excluded
    )

    all_items = list(items) + resolved
    return _compile_resolved(
        all_items,
        model=model,
        token_budget=token_budget,
        reserved_output_tokens=reserved_output_tokens,
        config=cfg,
        counter=cnt,
        warnings=warnings,
        pre_excluded=pre_excluded,
    )


async def acompile_items(
    items: list[ContextItem],
    *,
    model: str,
    token_budget: int,
    reserved_output_tokens: int,
    config: CompilerConfig | None = None,
    counter: TokenCounter | None = None,
    references: list[Reference] | None = None,
) -> CompiledPack:
    """Async compilation. Resolves async References concurrently; supports async compressors."""
    if token_budget <= 0:
        raise ValueError("token_budget must be > 0")
    if reserved_output_tokens < 0:
        raise ValueError("reserved_output_tokens must be >= 0")
    if reserved_output_tokens >= token_budget:
        raise ValueError("reserved_output_tokens must be less than token_budget")

    cfg = config or CompilerConfig()
    cnt = counter or TokenCounter(model)
    available = token_budget - reserved_output_tokens

    warnings: list[str] = []
    pre_excluded: list[ItemDecision] = []
    resolved = await _resolve_references_async(
        references or [], available, cnt, warnings, pre_excluded
    )

    all_items = list(items) + resolved
    return await _acompile_resolved(
        all_items,
        model=model,
        token_budget=token_budget,
        reserved_output_tokens=reserved_output_tokens,
        config=cfg,
        counter=cnt,
        warnings=warnings,
        pre_excluded=pre_excluded,
    )


def _compile_resolved(
    items: list[ContextItem],
    *,
    model: str,
    token_budget: int,
    reserved_output_tokens: int,
    config: CompilerConfig,
    counter: TokenCounter,
    warnings: list[str],
    pre_excluded: list[ItemDecision],
) -> CompiledPack:
    available = token_budget - reserved_output_tokens
    names = [it.name for it in items]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate item names: {dupes}")

    items_by_name = {it.name: it for it in items}
    tokens_for: dict[str, int] = {}
    for it in items:
        text_t, att_t = _item_token_cost(it, counter)
        tokens_for[it.name] = text_t + att_t

    required = [it for it in items if it.required]
    optional = [it for it in items if not it.required]

    state = _CompileState(
        decisions=[], used_text={}, used_tokens={}, used=0, warnings=warnings
    )
    _process_required(state, required, tokens_for, available, counter, config)
    _process_optional(state, optional, tokens_for, available, counter, config)
    return _finalize(
        items_by_name, state, config, counter, model, token_budget, reserved_output_tokens,
        [it.name for it in items], pre_excluded,
    )


async def _acompile_resolved(
    items: list[ContextItem],
    *,
    model: str,
    token_budget: int,
    reserved_output_tokens: int,
    config: CompilerConfig,
    counter: TokenCounter,
    warnings: list[str],
    pre_excluded: list[ItemDecision],
) -> CompiledPack:
    """Async variant — awaits async compressors. Same algorithm otherwise."""
    available = token_budget - reserved_output_tokens
    names = [it.name for it in items]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate item names: {dupes}")

    items_by_name = {it.name: it for it in items}
    tokens_for: dict[str, int] = {}
    for it in items:
        text_t, att_t = _item_token_cost(it, counter)
        tokens_for[it.name] = text_t + att_t

    required = [it for it in items if it.required]
    optional = [it for it in items if not it.required]

    state = _CompileState(decisions=[], used_text={}, used_tokens={}, used=0, warnings=warnings)

    # Required pass (async)
    for item in required:
        orig = tokens_for[item.name]
        remaining = available - state.used
        if orig <= remaining:
            state.used += orig
            state.used_text[item.name] = item.content
            state.used_tokens[item.name] = orig
            state.decisions.append(
                ItemDecision(
                    name=item.name, status="included", reason=_human_reason("included"),
                    score=float("inf"), tokens=orig, original_tokens=orig,
                    kind=item.kind, cache_policy=item.cache_policy,
                    priority=item.priority, required=True,
                    sensitivity=item.sensitivity, source=item.source,
                )
            )
            continue
        compressed = await _try_compress_async(item, remaining, counter, config, state.warnings)
        if compressed is not None and compressed[1] <= remaining:
            text, toks = compressed
            state.used += toks
            state.used_text[item.name] = text
            state.used_tokens[item.name] = toks
            state.decisions.append(
                ItemDecision(
                    name=item.name, status="compressed",
                    reason=_human_reason("compressed", orig=orig, final=toks),
                    score=float("inf"), tokens=toks, original_tokens=orig,
                    kind=item.kind, cache_policy=item.cache_policy,
                    priority=item.priority, required=True,
                    sensitivity=item.sensitivity, source=item.source,
                )
            )
            continue
        if config.allow_truncation:
            text, toks = _truncate_to_budget(item.content, remaining, counter, config.truncation_marker)
            if toks > 0:
                state.used += toks
                state.used_text[item.name] = text
                state.used_tokens[item.name] = toks
                state.decisions.append(
                    ItemDecision(
                        name=item.name, status="truncated",
                        reason=_human_reason("truncated", orig=orig, final=toks),
                        score=float("inf"), tokens=toks, original_tokens=orig,
                        kind=item.kind, cache_policy=item.cache_policy,
                        priority=item.priority, required=True,
                        sensitivity=item.sensitivity, source=item.source,
                    )
                )
                continue
        raise BudgetExceededError(
            f"Required item '{item.name}' ({orig:,} tokens) cannot fit in remaining "
            f"budget ({remaining:,} tokens)."
        )

    # Optional pass (async, but order is still deterministic)
    scored = [
        (score_item(it, tokens_for[it.name], available, config.weights), it) for it in optional
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    for sc, item in scored:
        orig = tokens_for[item.name]
        remaining = available - state.used
        if orig <= remaining:
            state.used += orig
            state.used_text[item.name] = item.content
            state.used_tokens[item.name] = orig
            state.decisions.append(
                ItemDecision(
                    name=item.name, status="included",
                    reason=_human_reason("fits", priority=item.priority),
                    score=sc, tokens=orig, original_tokens=orig,
                    kind=item.kind, cache_policy=item.cache_policy,
                    priority=item.priority, required=False,
                    sensitivity=item.sensitivity, source=item.source,
                )
            )
            continue
        compressed = await _try_compress_async(item, remaining, counter, config, state.warnings)
        if compressed is not None and 0 < compressed[1] <= remaining:
            text, toks = compressed
            state.used += toks
            state.used_text[item.name] = text
            state.used_tokens[item.name] = toks
            state.decisions.append(
                ItemDecision(
                    name=item.name, status="compressed",
                    reason=_human_reason("compressed", orig=orig, final=toks),
                    score=sc, tokens=toks, original_tokens=orig,
                    kind=item.kind, cache_policy=item.cache_policy,
                    priority=item.priority, required=False,
                    sensitivity=item.sensitivity, source=item.source,
                )
            )
            continue
        kind_key = "excluded_size_high_pri" if item.priority >= 80 else "excluded_size"
        state.decisions.append(
            ItemDecision(
                name=item.name, status="excluded",
                reason=_human_reason(kind_key, orig=orig, remaining=remaining, score=sc, priority=item.priority),
                score=sc, tokens=0, original_tokens=orig,
                kind=item.kind, cache_policy=item.cache_policy,
                priority=item.priority, required=False,
                sensitivity=item.sensitivity, source=item.source,
            )
        )

    return _finalize(
        items_by_name, state, config, counter, model, token_budget, reserved_output_tokens,
        [it.name for it in items], pre_excluded,
    )


# ----- from_dict round-trip ----------------------------------------------------------


def compiled_pack_from_dict(data: dict) -> CompiledPack:
    """Reconstruct a CompiledPack from `to_dict()` output.

    The reconstructed pack has full decisions and metadata but does NOT re-resolve
    references or re-run loaders — it's a read-only snapshot suitable for reporting,
    assertions, and golden-file comparison.
    """
    decisions = [
        ItemDecision(
            name=d["name"],
            status=d["status"],
            reason=d["reason"],
            score=(float("inf") if d["score"] == "required" else float(d["score"])),
            tokens=int(d["tokens"]),
            original_tokens=int(d["original_tokens"]),
            kind=d["kind"],
            cache_policy=d["cache_policy"],
            priority=int(d["priority"]),
            required=bool(d["required"]),
            sensitivity=d.get("sensitivity", "internal"),
            source=d.get("source"),
        )
        for d in data.get("decisions", [])
    ]
    return CompiledPack(
        model=data["model"],
        token_budget=int(data["token_budget"]),
        reserved_output_tokens=int(data["reserved_output_tokens"]),
        used_tokens=int(data["used_tokens"]),
        cacheable_prefix_tokens=int(data.get("cacheable_prefix_tokens", 0)),
        health_score=int(data.get("health_score", 0)),
        health_breakdown=dict(data.get("health_breakdown", {})),
        decisions=decisions,
        included_items=[],  # snapshot — items not re-materialized
        items_by_name={},
        included_tokens={},
        tokenizer_backend=data.get("tokenizer_backend", "unknown"),
        input_order=list(data.get("input_order", [])),
        warnings=list(data.get("warnings", [])),
    )

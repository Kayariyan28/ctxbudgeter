"""``context.diagnose()`` — find redundant, stale, conflicting and risky context.

All checks are deterministic and local. Each :class:`Diagnosis` carries the tokens
it would recover and a concrete suggested operation an agent can execute through
the context API, so diagnosis feeds straight into planning.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..scanner import ContextScanner
from .analysis import jaccard, salient_terms, shingles, unique_line_ratio
from .events import ContextEvent

_NEAR_DUP_MAX_EVENTS = 300


@dataclass
class Diagnosis:
    kind: str  # duplicate | near_duplicate | superseded | conflict | stale | oversized | repetitive | sensitive
    severity: str  # info | warning | error | critical
    tags: list[str]
    tokens_recoverable: int
    message: str
    suggestion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "tags": list(self.tags),
            "tokens_recoverable": self.tokens_recoverable,
            "message": self.message,
            "suggestion": dict(self.suggestion),
        }


def _referenced_later(ev: ContextEvent, later: Sequence[ContextEvent]) -> bool:
    pat = re.compile(rf"(?:\[e?{re.escape(ev.tag)}\]|\bevent {re.escape(ev.tag)}\b)")
    needles = [ev.source] if ev.source else []
    for other in later:
        if pat.search(other.content):
            return True
        if any(n and n in other.content for n in needles):
            return True
    return False


def diagnose(
    events: Sequence[ContextEvent],
    *,
    available_tokens: int,
    current_step: int,
    protected_categories: tuple[str, ...] = ("protected",),
    stale_after_steps: int = 8,
    oversized_fraction: float = 0.15,
    near_duplicate_threshold: float = 0.85,
    scanner: ContextScanner | None = None,
) -> list[Diagnosis]:
    scanner = scanner or ContextScanner()
    out: list[Diagnosis] = []

    def editable(e: ContextEvent) -> bool:
        return e.category not in protected_categories and not e.pinned

    # exact duplicates — keep the latest copy
    by_sum: dict[str, list[ContextEvent]] = {}
    for e in events:
        by_sum.setdefault(e.checksum, []).append(e)
    dup_tags: set[str] = set()
    for group in by_sum.values():
        if len(group) > 1:
            older = [e for e in group[:-1] if editable(e)]
            if older:
                dup_tags.update(e.tag for e in older)
                out.append(Diagnosis(
                    "duplicate", "warning", [e.tag for e in older], sum(e.tokens for e in older),
                    f"identical to event {group[-1].tag}; keep the latest copy",
                    {"op": "archive", "tags": [e.tag for e in older]}))

    # near duplicates
    cands = [e for e in events if editable(e) and e.tag not in dup_tags and len(e.content.split()) >= 30]
    if len(cands) <= _NEAR_DUP_MAX_EVENTS:
        sh = {e.tag: shingles(e.content) for e in cands}
        flagged: set[str] = set()
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if a.tag in flagged:
                    break
                if jaccard(sh[a.tag], sh[b.tag]) >= near_duplicate_threshold:
                    flagged.add(a.tag)
                    out.append(Diagnosis(
                        "near_duplicate", "info", [a.tag], a.tokens,
                        f"event {a.tag} is >= {near_duplicate_threshold:.0%} similar to later event {b.tag}",
                        {"op": "archive", "tags": [a.tag]}))

    # superseded / conflicting facts sharing a metadata key
    by_key: dict[str, list[ContextEvent]] = {}
    for e in events:
        k = e.metadata.get("key")
        if k:
            by_key.setdefault(str(k), []).append(e)
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        latest = group[-1]
        older = [e for e in group[:-1] if editable(e) and e.tag not in dup_tags]
        if not older:
            continue
        latest_terms = salient_terms(latest.content)
        conflicting = [e for e in older if salient_terms(e.content) - latest_terms]
        if conflicting:
            out.append(Diagnosis(
                "conflict", "warning", [e.tag for e in conflicting], sum(e.tokens for e in conflicting),
                f"key '{key}': events {', '.join(e.tag for e in conflicting)} disagree with the latest "
                f"event {latest.tag}; the latest value should win",
                {"op": "archive", "tags": [e.tag for e in conflicting]}))
        rest = [e for e in older if e not in conflicting]
        if rest:
            out.append(Diagnosis(
                "superseded", "info", [e.tag for e in rest], sum(e.tokens for e in rest),
                f"key '{key}': superseded by event {latest.tag}",
                {"op": "archive", "tags": [e.tag for e in rest]}))

    for idx, e in enumerate(events):
        if not editable(e):
            continue
        if e.tokens > oversized_fraction * available_tokens:
            out.append(Diagnosis(
                "oversized", "warning", [e.tag], e.tokens // 2,
                f"event {e.tag} uses {e.tokens:,} tokens ({e.tokens / max(1, available_tokens):.0%} of the budget)",
                {"op": "compact", "start": e.tag, "end": e.tag}))
        if e.tokens > 200 and unique_line_ratio(e.content) < 0.5:
            out.append(Diagnosis(
                "repetitive", "info", [e.tag], int(e.tokens * (1 - unique_line_ratio(e.content))),
                f"event {e.tag} is mostly repeated lines (verbose log output)",
                {"op": "compact", "start": e.tag, "end": e.tag}))
        if e.kind in ("tool_result", "document", "tool_call") and e.category in ("working", "compressible") \
                and current_step - e.step >= stale_after_steps and not _referenced_later(e, events[idx + 1:]):
            out.append(Diagnosis(
                "stale", "info", [e.tag], e.tokens,
                f"event {e.tag} is {current_step - e.step} steps old and not referenced since",
                {"op": "compact", "start": e.tag, "end": e.tag}))

    for e in events:
        scan = scanner.scan(e.content)
        if scan.has_secrets:
            out.append(Diagnosis(
                "sensitive", "critical", [e.tag], 0,
                f"event {e.tag} contains {len(scan.findings)} secret/PII finding(s) "
                f"({', '.join(sorted({f.category for f in scan.findings}))}) that every request resends",
                {"op": "redact", "tags": [e.tag]}))

    order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    out.sort(key=lambda d: (order.get(d.severity, 9), -d.tokens_recoverable))
    return out

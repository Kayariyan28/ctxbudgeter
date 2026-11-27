"""``context.plan()`` — propose a context-management strategy.

The planner turns diagnostics plus budget state into an ordered list of concrete
operations that bring usage down to a target, without touching protected, pinned or
recent context. It is *cache-aware*: under prefix caching an edit at position *p*
forces recomputation of every token after *p* (NVIDIA, Section 7), so each candidate
is scored by the tokens it saves over the expected remaining requests minus the
one-off recompute it triggers. Edits near the end of the context are cheap; edits
near the start must pay for themselves.

The plan also carries the timing verdict of the :class:`CompactionRubric`, so an agent
knows whether *now* is a good moment to execute it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .diagnostics import Diagnosis
from .events import ContextEvent
from .rubric import RubricVerdict

_ELIGIBLE = ("working", "compressible", "archivable")


@dataclass
class ProposedOp:
    op: str  # compact | archive
    args: dict[str, Any]
    est_tokens_saved: int
    recompute_cost: int
    score: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": dict(self.args),
            "est_tokens_saved": self.est_tokens_saved,
            "recompute_cost": self.recompute_cost,
            "score": round(self.score, 2),
            "rationale": self.rationale,
        }


@dataclass
class ContextPlan:
    current_usage: float
    target_usage: float
    projected_usage: float
    tokens_to_free: int
    ops: list[ProposedOp] = field(default_factory=list)
    timing: RubricVerdict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.ops

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_usage": round(self.current_usage, 4),
            "target_usage": round(self.target_usage, 4),
            "projected_usage": round(self.projected_usage, 4),
            "tokens_to_free": self.tokens_to_free,
            "timing": self.timing.to_dict() if self.timing else None,
            "ops": [o.to_dict() for o in self.ops],
            "notes": list(self.notes),
        }


def build_plan(
    events: Sequence[ContextEvent],
    *,
    available_tokens: int,
    target_usage: float,
    diagnoses: Sequence[Diagnosis],
    timing: RubricVerdict,
    protected_categories: tuple[str, ...],
    keep_recent: int,
    compression: float = 4.0,
    cache_aware: bool = True,
    expected_remaining_requests: int = 10,
    min_run_tokens: int = 120,
    resummarize: bool = False,
) -> ContextPlan:
    used = sum(e.tokens for e in events)
    cur = used / max(1, available_tokens)
    need = int(used - target_usage * available_tokens)
    plan = ContextPlan(cur, target_usage, cur, max(0, need), timing=timing)
    if need <= 0:
        plan.notes.append(f"usage {cur:.0%} is already at or below target {target_usage:.0%}")
        return plan

    index = {e.tag: i for i, e in enumerate(events)}
    recent = {e.tag for e in events[len(events) - keep_recent:]} if keep_recent else set()

    def suffix_cost(i: int) -> int:
        return sum(e.tokens for e in events[i:])

    cands: list[ProposedOp] = []
    reserved: set[str] = set()
    for d in diagnoses:
        if d.suggestion.get("op") != "archive":
            continue
        tags = [t for t in d.suggestion.get("tags", []) if t in index and t not in recent and t not in reserved]
        if not tags:
            continue
        saved = sum(events[index[t]].tokens for t in tags)
        first = min(index[t] for t in tags)
        cost = suffix_cost(first) - saved
        score = saved * (expected_remaining_requests if cache_aware else 1) - (cost if cache_aware else 0)
        cands.append(ProposedOp("archive", {"tags": tags}, saved, cost, score + 1e6 * (d.kind == "duplicate"),
                                f"{d.kind}: {d.message}"))
        reserved.update(tags)

    def eligible(e: ContextEvent) -> bool:
        # Existing summaries break runs unless re-summarizing is allowed: summarizing a
        # summary compounds loss (each pass drops more of the oldest facts).
        return (e.category in _ELIGIBLE and e.category not in protected_categories and not e.pinned
                and e.tag not in recent and e.tag not in reserved and (resummarize or not e.is_summary))

    run: list[ContextEvent] = []
    runs: list[list[ContextEvent]] = []
    for e in events:
        if eligible(e):
            run.append(e)
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)

    for r in runs:
        toks = sum(e.tokens for e in r)
        if toks < min_run_tokens:
            continue
        est_summary = max(40, int(toks / compression))
        saved = toks - est_summary
        if saved <= 0:
            continue
        first = index[r[0].tag]
        cost = suffix_cost(first) - toks + est_summary
        score = saved * (expected_remaining_requests if cache_aware else 1) - (cost if cache_aware else 0)
        cands.append(ProposedOp(
            "compact", {"start": r[0].tag, "end": r[-1].tag}, saved, cost, score,
            f"{len(r)} eligible event(s), {toks:,} tokens of completed work/tool output"))

    cands.sort(key=lambda o: (-o.score, -o.est_tokens_saved))
    freed = 0
    for op in cands:
        if freed >= need:
            break
        plan.ops.append(op)
        freed += op.est_tokens_saved
    if freed < need and not resummarize:
        # Last resort: allow existing summaries into runs.
        fallback = build_plan(events, available_tokens=available_tokens, target_usage=target_usage,
                              diagnoses=diagnoses, timing=timing, protected_categories=protected_categories,
                              keep_recent=keep_recent, compression=compression, cache_aware=cache_aware,
                              expected_remaining_requests=expected_remaining_requests,
                              min_run_tokens=min_run_tokens, resummarize=True)
        fallback.notes.append("no fresh material left to compact; plan re-summarizes earlier summaries")
        return fallback
    plan.projected_usage = (used - freed) / max(1, available_tokens)
    if freed < need:
        plan.notes.append(
            f"only {freed:,} of {need:,} tokens can be freed without touching protected, pinned "
            "or recent context")
    if cache_aware and plan.ops:
        earliest = min(index[o.args.get("start") or o.args["tags"][0]] for o in plan.ops)
        plan.notes.append(
            f"executing in one turn invalidates the KV prefix from event {events[earliest].tag}: "
            f"~{suffix_cost(earliest):,} tokens recomputed once under prefix caching")
    return plan

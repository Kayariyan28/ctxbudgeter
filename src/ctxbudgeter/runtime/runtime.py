"""ContextRuntime — governed, model-driven context management for a live agent.

The runtime owns an agent's working context as an ordered list of tagged
:class:`~ctxbudgeter.runtime.events.ContextEvent` objects and exposes the universal
context API:

============== =================================================================
inspect()      what is in context: composition by category/kind, every event
budget()       tokens used/free, usage, pressure level, timing verdict
diagnose()     redundant, stale, conflicting, oversized and sensitive context
plan()         a cache-aware strategy to reach a target usage
compact()      replace a range with a summary; originals archived (NOOA collapse)
pin()/unpin()  protect task-critical events from edits (subject to a pin budget)
archive()      move events out of active context, recoverable by tag
recall()       read an archived original by tag; optionally restore it
checkpoint()   snapshot the active context
rollback()     revert an unsuccessful edit
evaluate()     quality of one decision, or of the session's context management
============== =================================================================

Every mutating call is a *proposal*. The :class:`~ctxbudgeter.governance.EditGovernor`
validates it against a :class:`~ctxbudgeter.governance.ContextEditPolicy`, the
agent's :class:`~ctxbudgeter.governance.AutonomyLevel` decides whether it executes,
is queued for approval, becomes a recommendation, or is only observed — and a
:class:`~ctxbudgeter.runtime.trace.DecisionTrace` is recorded either way.

The harness calls :meth:`step` each time it sends a request to the model. ``step``
enforces the hard budget (with the paper's drop-oldest fallback when exceeded), runs
governed fallback compaction for agents that never compact (if the policy enables
it), and books prefill cost under prefix caching vs. suffix KV reuse.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from ..governance.autonomy import AutonomyLevel
from ..governance.edits import ContextEditPolicy, EditDecision, EditGovernor, EditProposal
from ..scanner import ContextScanner
from ..tokenizer import TokenCounter
from .analysis import query_terms, salient_terms
from .archive import ArchiveStore, InMemoryArchive, _tag_key
from .cache_model import PrefillCostModel, RequestCost
from .diagnostics import Diagnosis, diagnose
from .events import CATEGORIES, DEFAULT_CATEGORY, ContextCategory, ContextEvent, range_tag
from .planner import ContextPlan, build_plan
from .rubric import PHASES
from .summarizer import ExtractiveSummarizer, Summarizer
from .trace import DecisionTrace, TraceLog

Approver = Callable[["PendingEdit"], bool]
_DESTRUCTIVE = ("compact", "archive", "restore")


class ContextOverflowError(RuntimeError):
    """Raised when protected context alone exceeds the hard budget."""


@dataclass
class OperationResult:
    op: str
    status: str  # applied | denied | recommended | pending | not_found | noop
    message: str = ""
    trace_id: str | None = None
    proposal_id: str | None = None
    decision: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def ok(self) -> bool:
        return self.status in ("applied", "noop")

    @property
    def tokens_freed(self) -> int:
        return self.tokens_before - self.tokens_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "status": self.status,
            "message": self.message,
            "trace_id": self.trace_id,
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "details": self.details,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
        }


@dataclass
class PendingEdit:
    proposal_id: str
    op: str
    kwargs: dict[str, Any]
    decision: EditDecision
    step: int
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {"proposal_id": self.proposal_id, "op": self.op, "kwargs": dict(self.kwargs),
                "decision": self.decision.to_dict(), "step": self.step}


@dataclass
class StepReport:
    step: int
    cost: RequestCost
    usage: float
    overflow_dropped: list[str] = field(default_factory=list)
    fallback_applied: bool = False
    warning: str | None = None
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "cost": self.cost.to_dict(), "usage": round(self.usage, 4),
                "overflow_dropped": list(self.overflow_dropped), "fallback_applied": self.fallback_applied,
                "warning": self.warning, "failed": self.failed}


@dataclass
class _Snapshot:
    events: list[ContextEvent]
    phase: str | None
    label: str | None
    step: int


class ContextRuntime:
    """Governed live context for one agent session. See module docstring."""

    def __init__(
        self,
        *,
        budget: int,
        model: str = "claude-sonnet-4.6",
        reserved_output_tokens: int = 0,
        policy: ContextEditPolicy | None = None,
        autonomy: AutonomyLevel | int | str = AutonomyLevel.GOVERNED,
        archive: ArchiveStore | None = None,
        trace: TraceLog | None = None,
        summarizer: Summarizer | None = None,
        approver: Approver | None = None,
        agent_id: str = "agent",
        session_id: str | None = None,
        counter: TokenCounter | None = None,
        scanner: ContextScanner | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be > 0")
        if not 0 <= reserved_output_tokens < budget:
            raise ValueError("reserved_output_tokens must be >= 0 and < budget")
        self.model = model
        self.budget_tokens = budget
        self.reserved_output_tokens = reserved_output_tokens
        self.policy = policy or ContextEditPolicy()
        self.autonomy = AutonomyLevel.parse(autonomy)
        self.archive_store: ArchiveStore = archive if archive is not None else InMemoryArchive()
        self.trace = trace if trace is not None else TraceLog()
        self.counter = counter or TokenCounter(model)
        self.scanner = scanner or ContextScanner()
        self.summarizer: Summarizer = summarizer or ExtractiveSummarizer(counter=self.counter)
        self.governor = EditGovernor(self.policy, scanner=self.scanner)
        self.approver = approver
        self.agent_id = agent_id
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.clock = clock
        self.cost_model = PrefillCostModel()

        self._events: list[ContextEvent] = []
        self._next_tag = 1
        self.step_index = 0
        self.phase: str | None = None
        self._checkpoints: dict[str, _Snapshot] = {}
        self._pending: dict[str, PendingEdit] = {}
        self._mutations_this_step = 0
        self._covered_by: dict[str, str] = {}  # archived tag -> trace_id of the compaction
        self._seq = itertools.count(1)
        self.committed_violations: list[dict[str, Any]] = []
        self.stats: dict[str, int] = {
            "compactions": 0, "archives": 0, "recalls": 0, "restores": 0, "pins": 0,
            "denied": 0, "recommended": 0, "pending": 0, "overflow_events_dropped": 0,
            "overflow_steps": 0, "fallback_compactions": 0, "harness_compactions": 0,
            "warnings": 0, "rollbacks": 0,
        }

    # ================================================================ state

    @property
    def events(self) -> list[ContextEvent]:
        """A copy of the active events, in prompt order."""
        return [e.copy() for e in self._events]

    @property
    def available_tokens(self) -> int:
        return self.budget_tokens - self.reserved_output_tokens

    @property
    def used_tokens(self) -> int:
        return sum(e.tokens for e in self._events)

    @property
    def usage(self) -> float:
        return self.used_tokens / max(1, self.available_tokens)

    @property
    def pinned_tokens(self) -> int:
        return sum(e.tokens for e in self._events if e.pinned)

    def get(self, tag: str) -> ContextEvent | None:
        for e in self._events:
            if e.tag == tag:
                return e.copy()
        return None

    def __len__(self) -> int:
        return len(self._events)

    def __contains__(self, tag: object) -> bool:
        return any(e.tag == tag for e in self._events)

    def _index(self, tag: str) -> int:
        for i, e in enumerate(self._events):
            if e.tag == tag:
                return i
        raise KeyError(tag)

    def _recent_tags(self) -> tuple[str, ...]:
        k = self.policy.keep_recent_events
        return tuple(e.tag for e in self._events[len(self._events) - k:]) if k else ()

    # ============================================================ lifecycle

    def add(
        self,
        content: str,
        *,
        kind: str = "tool_result",
        category: ContextCategory | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        pinned: bool = False,
        tokens: int | None = None,
    ) -> ContextEvent:
        """Append an event (harness-side: tool output, user turn, model turn, ...)."""
        cat = category or DEFAULT_CATEGORY.get(kind, "working")
        if cat not in CATEGORIES:
            raise ValueError(f"unknown category {cat!r}; expected one of {CATEGORIES}")
        ev = ContextEvent(
            tag=str(self._next_tag), kind=kind, content=content,
            tokens=tokens if tokens is not None else self.counter.count(content),
            category=cat, step=self.step_index, source=source, pinned=pinned,
            pin_reason="pinned at creation" if pinned else None, metadata=dict(metadata or {}),
        )
        self._next_tag += 1
        self._events.append(ev)
        return ev.copy()

    def signal(self, phase: str) -> None:
        """Declare the agent's current phase (drives the compaction timing rubric)."""
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        self.phase = phase

    def step(self) -> StepReport:
        """Book one model request. Enforces the hard budget; see module docstring."""
        fallback = False
        if self.policy.fallback_compaction and self.usage >= self.policy.hard_limit:
            fallback = self._fallback_compact()
        dropped: list[str] = []
        failed = False
        if self.used_tokens > self.available_tokens:
            self.stats["overflow_steps"] += 1
            # Paper fallback: drop the oldest non-protected events until the request fits.
            i = 0
            while self.used_tokens > self.available_tokens and i < len(self._events):
                ev = self._events[i]
                if ev.category in self.policy.protected_categories:
                    i += 1
                    continue
                dropped.append(ev.tag)
                del self._events[i]
            self.stats["overflow_events_dropped"] += len(dropped)
            failed = self.used_tokens > self.available_tokens
        cost = self.cost_model.record(self._events, self.step_index)
        warning = self.pressure_warning()
        if warning:
            self.stats["warnings"] += 1
        report = StepReport(self.step_index, cost, self.usage, dropped, fallback, warning, failed)
        self.step_index += 1
        self._mutations_this_step = 0
        return report

    def pressure_warning(self) -> str | None:
        """The context-pressure message to append to the next request, if any."""
        if self.usage < self.policy.warn_at:
            return None
        plan = self.plan()
        hint = ""
        if plan.ops:
            first = plan.ops[0]
            hint = (f" Suggested: compact events {first.args['start']}..{first.args['end']} "
                    f"(~{first.est_tokens_saved:,} tokens)." if first.op == "compact"
                    else f" Suggested: archive events {', '.join(first.args['tags'])}.")
        return (
            f"[context] Usage at {self.usage:.0%} of {self.available_tokens:,} available tokens."
            f"{hint} Compaction is expected work, not a reason to stop: compact completed work with "
            "context_compact, then continue the task. Originals stay recoverable with context_recall."
        )

    # ============================================================ read API

    def budget(self) -> dict[str, Any]:
        u = self.usage
        pol = self.policy
        level = ("low" if u < pol.min_usage_to_compact else "moderate" if u < pol.compaction_trigger
                 else "high" if u < pol.hard_limit else "critical")
        return {
            "budget": self.budget_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "available": self.available_tokens,
            "used": self.used_tokens,
            "free": self.available_tokens - self.used_tokens,
            "usage": round(u, 4),
            "pressure": level,
            "pinned_tokens": self.pinned_tokens,
            "thresholds": {"min_usage_to_compact": pol.min_usage_to_compact,
                           "compaction_trigger": pol.compaction_trigger, "hard_limit": pol.hard_limit},
            "timing": pol.rubric.evaluate(u, self.phase).to_dict(),
        }

    def inspect(self, *, preview_chars: int = 120) -> dict[str, Any]:
        by_cat = {c: {"events": 0, "tokens": 0} for c in CATEGORIES}
        by_kind: dict[str, dict[str, int]] = {}
        for e in self._events:
            by_cat[e.category]["events"] += 1
            by_cat[e.category]["tokens"] += e.tokens
            k = by_kind.setdefault(e.kind, {"events": 0, "tokens": 0})
            k["events"] += 1
            k["tokens"] += e.tokens
        return {
            "session_id": self.session_id,
            "step": self.step_index,
            "phase": self.phase,
            "autonomy": self.autonomy.name,
            "budget": self.budget(),
            "composition": {"by_category": by_cat, "by_kind": by_kind},
            "events": [e.to_view(preview_chars) for e in self._events],
            "archived": len(self.archive_store),
            "pending_approvals": [p.to_dict() for p in self._pending.values()],
        }

    def diagnose(self) -> list[Diagnosis]:
        return diagnose(self._events, available_tokens=self.available_tokens, current_step=self.step_index,
                        protected_categories=self.policy.protected_categories, scanner=self.scanner)

    def plan(self, *, target_usage: float | None = None, cache_aware: bool = True,
             expected_remaining_requests: int = 10, compression: float | None = None) -> ContextPlan:
        pol = self.policy
        target = target_usage if target_usage is not None else max(0.3, pol.min_usage_to_compact * 0.9)
        comp = compression or (min(4.0, pol.max_compression_ratio) if pol.max_compression_ratio else 4.0)
        return build_plan(
            self._events, available_tokens=self.available_tokens, target_usage=target,
            diagnoses=self.diagnose(), timing=pol.rubric.evaluate(self.usage, self.phase),
            protected_categories=pol.protected_categories, keep_recent=pol.keep_recent_events,
            compression=comp, cache_aware=cache_aware, expected_remaining_requests=expected_remaining_requests,
        )

    def search_archive(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return [{"tag": ev.tag, "score": round(s, 3), "tokens": ev.tokens, "kind": ev.kind,
                 "source": ev.source, "preview": ev.preview(160)}
                for ev, s in self.archive_store.search(query, limit=limit)]

    def recall_hints(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Archived originals that mention the query's salient terms (identifiers,
        numbers, periods) and are not already in active context, best match first.

        Use before answering: a summary can keep an entity's name while dropping the
        figure, so presence of a term in active context is not evidence the fact is."""
        sal = {t.lower() for t in salient_terms(query)}
        terms = sal or query_terms(query)
        active_sums = {e.checksum for e in self._events}
        scored: list[tuple[int, ContextEvent]] = []
        for ev in self.archive_store.all():
            if ev.is_summary or ev.checksum in active_sums:
                continue
            low = ev.content.lower()
            hits = sum(1 for t in terms if t in low)
            if hits:
                scored.append((hits, ev))
        scored.sort(key=lambda p: (-p[0], _tag_key(p[1].tag)))
        return [{"tag": ev.tag, "score": round(h / max(1, len(terms)), 3), "tokens": ev.tokens, "kind": ev.kind,
                 "source": ev.source, "preview": ev.preview(160)} for h, ev in scored[:limit]]

    # =========================================================== write API

    def compact(self, start: str, end: str | None = None, summary: str | None = None, *,
                target_tokens: int | None = None) -> OperationResult:
        """Replace events ``start..end`` (inclusive, by tag) with a summary."""
        return self._run("compact", {"start": str(start), "end": str(end if end is not None else start),
                                     "summary": summary, "target_tokens": target_tokens})

    def archive(self, *tags: str) -> OperationResult:
        return self._run("archive", {"tags": [str(t) for t in tags]})

    def pin(self, tag: str, reason: str = "") -> OperationResult:
        return self._run("pin", {"tag": str(tag), "reason": reason})

    def unpin(self, tag: str) -> OperationResult:
        return self._run("unpin", {"tag": str(tag)})

    def recall(self, tag: str, *, restore: bool = False) -> OperationResult:
        """Read an archived event by its original tag. ``restore=True`` re-inserts it
        into active context (governed)."""
        return self._run("restore" if restore else "recall", {"tag": str(tag)})

    def checkpoint(self, label: str | None = None) -> OperationResult:
        return self._run("checkpoint", {"label": label})

    def rollback(self, checkpoint_id: str | None = None) -> OperationResult:
        return self._run("rollback", {"checkpoint_id": checkpoint_id})

    def apply_plan(self, plan: ContextPlan) -> list[OperationResult]:
        results = []
        for op in plan.ops:
            if op.op == "compact":
                results.append(self.compact(op.args["start"], op.args["end"]))
            elif op.op == "archive":
                results.append(self.archive(*op.args["tags"]))
        return results

    # ---------------------------------------------------------- approvals

    @property
    def pending(self) -> list[PendingEdit]:
        return list(self._pending.values())

    def approve(self, proposal_id: str) -> OperationResult:
        """Approve a queued (L2) edit. It is re-validated against the current state."""
        p = self._pending.pop(proposal_id, None)
        if p is None:
            return OperationResult("approve", "not_found", f"no pending proposal {proposal_id}")
        return self._run(p.op, p.kwargs, level=AutonomyLevel.GOVERNED, actor="approver")

    def reject(self, proposal_id: str, reason: str = "") -> OperationResult:
        p = self._pending.pop(proposal_id, None)
        if p is None:
            return OperationResult("reject", "not_found", f"no pending proposal {proposal_id}")
        self.stats["denied"] += 1
        return OperationResult(p.op, "denied", reason or "rejected by approver", proposal_id=proposal_id)

    # ========================================================= evaluation

    def evaluate(self, trace_id: str | None = None) -> dict[str, Any]:
        """Assess one decision (by trace id) or the whole session."""
        traces = [t for t in self.trace if t.session_id == self.session_id]
        if trace_id is not None:
            t = next((x for x in traces if x.trace_id == trace_id), None)
            if t is None:
                raise KeyError(trace_id)
            notes = t.outcome.get("decision_notes", {})
            return {
                "trace_id": t.trace_id, "op": t.op, "verdict": t.verdict,
                "usage_at_decision": t.usage_before,
                "timing": self.policy.rubric.evaluate(t.usage_before, t.phase).decision,
                "compression_factor": t.compression_factor,
                "fact_retention": notes.get("fact_retention"),
                "critical_fact_retention": notes.get("critical_fact_retention"),
                "reopened": bool(t.outcome.get("reopened")),
                "violations": t.violations,
            }
        comp = [t for t in traces if t.op == "compact"]
        applied = [t for t in comp if t.verdict in ("applied", "modified")]
        factors = [f for f in (t.compression_factor for t in applied) if f]
        rets = [t.outcome.get("decision_notes", {}).get("fact_retention") for t in applied]
        rets = [r for r in rets if r is not None]
        first_usage = comp[0].usage_before if comp else None
        early = [t for t in comp if t.usage_before < self.policy.min_usage_to_compact]
        return {
            "session_id": self.session_id,
            "requests": self.step_index,
            "compaction_attempts": len(comp),
            "compactions_applied": len(applied),
            "median_compression_factor": round(median(factors), 3) if factors else None,
            "mean_fact_retention": round(sum(rets) / len(rets), 4) if rets else None,
            "reopen_rate": round(sum(1 for t in applied if t.outcome.get("reopened")) / len(applied), 4)
            if applied else None,
            "first_compaction_usage": round(first_usage, 4) if first_usage is not None else None,
            "early_compaction_attempts": len(early),
            "denied": self.stats["denied"],
            "committed_violations": len(self.committed_violations),
            "critical_violations": sum(1 for v in self.committed_violations if v.get("severity") == "critical"),
            "overflow_events_dropped": self.stats["overflow_events_dropped"],
            "fallback_compactions": self.stats["fallback_compactions"],
            "cost": self.cost_model.totals(),
            "stats": dict(self.stats),
        }

    def end_session(self, **outcome: Any) -> dict[str, Any]:
        """Close the session: record a ``session_end`` trace carrying the session
        evaluation plus any task outcome (e.g. ``correct=True``). Profilers and
        dataset exporters use it for credit assignment."""
        summary = self.evaluate()
        summary.update(outcome)
        self.trace.append(DecisionTrace(
            trace_id=uuid.uuid4().hex[:16], session_id=self.session_id, agent_id=self.agent_id, model=self.model,
            step=self.step_index, op="session_end", args={}, autonomy=int(self.autonomy), verdict="observed",
            usage_before=round(self.usage, 4), tokens_before=self.used_tokens, tokens_after=self.used_tokens,
            phase=self.phase, state={}, outcome={**summary, "actor": "harness"}, timestamp=self.clock(),
        ))
        return summary

    # =========================================================== rendering

    def render_text(self, *, with_tags: bool = True) -> str:
        parts = []
        for e in self._events:
            head = f"[e{e.tag} {e.kind}{' pinned' if e.pinned else ''}]" if with_tags else ""
            parts.append(f"{head}\n{e.content}" if head else e.content)
        return "\n\n".join(parts)

    def to_messages(self, format: str = "openai", *, with_tags: bool = True) -> Any:
        """Render active context as provider messages. Tags are shown so the model can
        reference events in context-API calls."""
        system = [e.content for e in self._events if e.kind == "system"]
        turns: list[dict[str, str]] = []
        for e in self._events:
            if e.kind == "system":
                continue
            role = "assistant" if e.kind == "assistant_message" else "user"
            text = (f"[e{e.tag} {e.kind}{' src=' + e.source if e.source else ''}]\n{e.content}"
                    if with_tags else e.content)
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n\n" + text
            else:
                turns.append({"role": role, "content": text})
        if format == "anthropic":
            return {"system": "\n\n".join(system), "messages": turns}
        if format == "openai":
            msgs = [{"role": "system", "content": "\n\n".join(system)}] if system else []
            return msgs + turns
        raise ValueError("format must be 'openai' or 'anthropic'")

    def to_context_pack(self, **pack_kwargs: Any) -> Any:
        """Bridge to the 0.3 compile pipeline: one ContextItem per active event, so
        ContextPolicy, the scanner, BOM, cache planning and Context MRI all apply."""
        from ..pack import ContextPack

        kind_map = {"system": "system", "task": "task", "user_message": "user_message",
                    "assistant_message": "assistant_message", "tool_call": "tool_result",
                    "tool_result": "tool_result", "document": "retrieval", "note": "memory",
                    "summary": "memory"}
        prio = {"protected": 100, "task_critical": 90, "working": 70, "compressible": 40, "archivable": 20}
        pack = ContextPack(model=self.model, token_budget=self.budget_tokens,
                           reserved_output_tokens=self.reserved_output_tokens, **pack_kwargs)
        for e in self._events:
            pack.add(name=f"e{e.tag}", content=e.content, kind=kind_map.get(e.kind, "other"),  # type: ignore[arg-type]
                     priority=prio[e.category], required=e.category == "protected" or e.pinned,
                     cache_policy="stable" if e.category == "protected" else "dynamic", source=e.source,
                     metadata={"tag": e.tag, "category": e.category, "covers": list(e.covers)})
        return pack

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "agent_id": self.agent_id, "model": self.model,
            "budget": self.budget_tokens, "reserved_output_tokens": self.reserved_output_tokens,
            "autonomy": int(self.autonomy), "policy": self.policy.to_dict(), "step": self.step_index,
            "phase": self.phase, "next_tag": self._next_tag,
            "events": [e.to_dict() for e in self._events], "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> ContextRuntime:
        rt = cls(budget=data["budget"], model=data.get("model", "claude-sonnet-4.6"),
                 reserved_output_tokens=data.get("reserved_output_tokens", 0),
                 policy=ContextEditPolicy.from_dict(data["policy"]) if "policy" in data else None,
                 autonomy=data.get("autonomy", AutonomyLevel.GOVERNED), agent_id=data.get("agent_id", "agent"),
                 session_id=data.get("session_id"), **kwargs)
        rt._events = [ContextEvent.from_dict(d) for d in data.get("events", [])]
        rt._next_tag = int(data.get("next_tag", len(rt._events) + 1))
        rt.step_index = int(data.get("step", 0))
        rt.phase = data.get("phase")
        rt.stats.update(data.get("stats", {}))
        return rt

    # ======================================================== internals

    def _build(self, op: str, kw: dict[str, Any]) -> tuple[EditProposal | None, Callable[[EditDecision], dict[str, Any]], str | None]:
        """Return (proposal, applier, error). ``error`` set => not_found / invalid."""
        def proposal(targets: list[ContextEvent], **extra: Any) -> EditProposal:
            return EditProposal(op, targets, usage=self.usage, available_tokens=self.available_tokens,
                                phase=self.phase, recent_tags=self._recent_tags(), pinned_tokens=self.pinned_tokens,
                                used_tokens=self.used_tokens, mutations_this_step=self._mutations_this_step, **extra)

        if op == "compact":
            try:
                i, j = self._index(kw["start"]), self._index(kw["end"])
            except KeyError as e:
                return None, _noop, f"tag {e.args[0]} is not in active context"
            if i > j:
                i, j = j, i
            targets = [e.copy() for e in self._events[i:j + 1]]
            toks = sum(e.tokens for e in targets)
            text = kw.get("summary")
            if not text:
                tgt = kw.get("target_tokens") or max(48, math.ceil(toks / 4))
                text = self.summarizer(targets, tgt)
            prop = proposal(targets, summary=text, summary_tokens=self.counter.count(text))

            def apply_compact(d: EditDecision) -> dict[str, Any]:
                ii, jj = self._index(targets[0].tag), self._index(targets[-1].tag)
                ev = self._replace_range(ii, jj, d.summary or text, archive_originals=True, actor="agent")
                return {"summary_tag": ev.tag, "summary_tokens": ev.tokens, "covers": ev.covers}
            return prop, apply_compact, None

        if op == "archive":
            missing = [t for t in kw["tags"] if t not in self]
            if missing:
                return None, _noop, f"tags not in active context: {missing}"
            targets = [self._events[self._index(t)].copy() for t in kw["tags"]]
            prop = proposal(targets)

            def apply_archive(d: EditDecision) -> dict[str, Any]:
                for t in targets:
                    self.archive_store.put(t)
                tagset = {t.tag for t in targets}
                self._events = [e for e in self._events if e.tag not in tagset]
                return {"archived": sorted(tagset, key=_tag_key)}
            return prop, apply_archive, None

        if op in ("pin", "unpin"):
            if kw["tag"] not in self:
                return None, _noop, f"tag {kw['tag']} is not in active context"
            idx = self._index(kw["tag"])
            prop = proposal([self._events[idx].copy()])

            def apply_pin(d: EditDecision) -> dict[str, Any]:
                e = self._events[self._index(kw["tag"])]
                e.pinned = op == "pin"
                e.pin_reason = (kw.get("reason") or "pinned by agent") if op == "pin" else None
                return {"tag": e.tag, "pinned": e.pinned}
            return prop, apply_pin, None

        if op in ("recall", "restore"):
            tag = kw["tag"]
            active = self.get(tag)
            if active is not None:
                return None, _noop, "__active__"
            ev = self.archive_store.get(tag)
            if ev is None:
                return None, _noop, f"tag {tag} not found in archive"
            originals = self.archive_store.expand(tag)
            prop = proposal([ev], restore_tokens=ev.tokens if op == "restore" else 0)

            def apply_recall(d: EditDecision) -> dict[str, Any]:
                self._mark_reopened(tag, originals)
                out = {"tag": ev.tag, "kind": ev.kind, "content": ev.content, "tokens": ev.tokens,
                       "source": ev.source, "covers": list(ev.covers),
                       "originals": [o.tag for o in originals]}
                if op == "restore":
                    restored = ev.copy(metadata={**ev.metadata, "restored": True}, pinned=False)
                    pos = len(self._events)
                    for k, e in enumerate(self._events):
                        if _tag_key(e.tag) > _tag_key(ev.tag):
                            pos = k
                            break
                    self._events.insert(pos, restored)
                    out["restored"] = True
                return out
            return prop, apply_recall, None

        if op == "checkpoint":
            prop = proposal([])

            def apply_checkpoint(d: EditDecision) -> dict[str, Any]:
                cid = f"cp{next(self._seq)}"
                self._checkpoints[cid] = _Snapshot(copy.deepcopy(self._events), self.phase, kw.get("label"),
                                                   self.step_index)
                return {"checkpoint_id": cid, "label": kw.get("label"), "events": len(self._events),
                        "tokens": self.used_tokens}
            return prop, apply_checkpoint, None

        if op == "rollback":
            cid = kw.get("checkpoint_id") or (next(reversed(self._checkpoints)) if self._checkpoints else None)
            if cid is None or cid not in self._checkpoints:
                return None, _noop, f"checkpoint {cid} not found"
            prop = proposal([])

            def apply_rollback(d: EditDecision) -> dict[str, Any]:
                snap = self._checkpoints[cid]
                self._events = copy.deepcopy(snap.events)
                self.phase = snap.phase
                self.stats["rollbacks"] += 1
                return {"checkpoint_id": cid, "events": len(self._events), "tokens": self.used_tokens}
            return prop, apply_rollback, None

        raise ValueError(f"unknown operation {op!r}")

    def _run(self, op: str, kw: dict[str, Any], *, level: AutonomyLevel | None = None,
             actor: str = "agent") -> OperationResult:
        before = self.used_tokens
        prop, apply, err = self._build(op, kw)
        if err == "__active__":
            ev = self.get(kw["tag"])
            assert ev is not None
            return OperationResult(op, "noop", "event is already in active context",
                                   details={"tag": ev.tag, "content": ev.content, "tokens": ev.tokens},
                                   tokens_before=before, tokens_after=before)
        if prop is None:
            return OperationResult(op, "not_found", err or "", tokens_before=before, tokens_after=before)

        decision = self.governor.review(prop)
        lvl = level if level is not None else self.autonomy
        read_only = op in ("recall", "checkpoint")
        status: str
        details: dict[str, Any] = {}
        proposal_id: str | None = None
        message = ""

        if read_only and decision.approved:
            status = "applied"
        elif lvl == AutonomyLevel.OBSERVE:
            status, message = "denied", "autonomy L0 (observe): context edits are not permitted"
        elif lvl == AutonomyLevel.RECOMMEND:
            status = "recommended"
            message = ("recommendation recorded (autonomy L1); the harness decides whether to apply it"
                       if decision.approved else "recommendation would be denied by policy")
        elif not decision.approved:
            status, message = "denied", "; ".join(v.message for v in decision.violations if v.blocking)
        elif lvl == AutonomyLevel.SUPERVISED and actor == "agent":
            proposal_id = f"p{next(self._seq)}"
            pend = PendingEdit(proposal_id, op, dict(kw), decision, self.step_index, self.clock())
            if self.approver is not None:
                if self.approver(pend):
                    return self._run(op, kw, level=AutonomyLevel.GOVERNED, actor="approver")
                status, message = "denied", "rejected by approver"
            else:
                self._pending[proposal_id] = pend
                status, message = "pending", "queued for approval (autonomy L2)"
        else:
            status = "applied"

        if status == "applied":
            if (op in _DESTRUCTIVE and self.policy.auto_checkpoint and lvl >= AutonomyLevel.GOVERNED
                    and actor != "fallback"):
                self._checkpoints[f"cp{next(self._seq)}"] = _Snapshot(
                    copy.deepcopy(self._events), self.phase, f"auto:before:{op}", self.step_index)
            details = apply(decision)
            if op not in ("recall", "checkpoint", "rollback"):
                self._mutations_this_step += 1
            stat = {"compact": "compactions", "archive": "archives", "recall": "recalls",
                    "restore": "restores", "pin": "pins"}.get(op)
            if stat:
                self.stats[stat] += 1
            if actor == "fallback" and op == "compact":
                self.stats["fallback_compactions"] += 1
            for v in decision.violations:
                if v.blocking and v.rule != "secret_in_summary_redacted":
                    self.committed_violations.append({**v.to_dict(), "op": op, "step": self.step_index})
        elif status in ("denied", "recommended", "pending"):
            self.stats[status] += 1

        trace_verdict = ("modified" if status == "applied" and decision.verdict == "modify" else status)
        trace = DecisionTrace(
            trace_id=uuid.uuid4().hex[:16], session_id=self.session_id, agent_id=self.agent_id, model=self.model,
            step=self.step_index, op=op, args=_trace_args(kw), autonomy=int(lvl), verdict=trace_verdict,
            usage_before=round(prop.usage, 4), tokens_before=prop.target_tokens if op == "compact" else before,
            tokens_after=(details.get("summary_tokens", 0) if op == "compact" else self.used_tokens),
            phase=self.phase, violations=[v.to_dict() for v in decision.violations],
            state=self._state_view(prop), outcome={"decision_notes": decision.notes, "actor": actor},
            timestamp=self.clock(),
        )
        self.trace.append(trace)
        if op == "compact" and status == "applied":
            for tag in details.get("covers", []):
                self._covered_by[tag] = trace.trace_id
                for o in self.archive_store.expand(tag):
                    self._covered_by.setdefault(o.tag, trace.trace_id)
        if op == "archive" and status == "applied":
            for tag in details.get("archived", []):
                self._covered_by[tag] = trace.trace_id
        return OperationResult(op, status, message, trace.trace_id, proposal_id, decision.to_dict(), details,
                               before, self.used_tokens)

    def _state_view(self, prop: EditProposal) -> dict[str, Any]:
        return {"usage": round(self.usage, 4), "available": self.available_tokens, "used": self.used_tokens,
                "phase": self.phase, "step": self.step_index,
                "events": [e.to_view(80) for e in self._events][-60:],
                "targets": list(prop.target_tags)}

    def _mark_reopened(self, tag: str, originals: Sequence[ContextEvent]) -> None:
        for t in [tag, *[o.tag for o in originals]]:
            tid = self._covered_by.get(t)
            if tid:
                self.trace.update_outcome(tid, reopened=True, reopened_at_step=self.step_index)

    def _replace_range(self, i: int, j: int, summary_text: str, *, archive_originals: bool,
                       actor: str) -> ContextEvent:
        targets = self._events[i:j + 1]
        first = targets[0].tag.partition("..")[0]
        last = targets[-1].tag.rpartition("..")[2]
        tag = range_tag(first, last)
        existing = {e.tag for e in self._events} | {e.tag for e in self.archive_store.all()}
        if tag in existing:
            n = 2
            while f"{tag}~{n}" in existing:
                n += 1
            tag = f"{tag}~{n}"
        if archive_originals:
            for t in targets:
                self.archive_store.put(t)
        before = sum(t.tokens for t in targets)
        ev = ContextEvent(
            tag=tag, kind="summary", content=summary_text, tokens=self.counter.count(summary_text),
            category="working", step=self.step_index,
            covers=[t.tag for t in targets],
            metadata={"actor": actor, "replaced_tokens": before, "archived": archive_originals,
                      "sources": sorted({t.source for t in targets if t.source})},
        )
        self._events[i:j + 1] = [ev]
        if actor == "harness":
            self.stats["harness_compactions"] += 1
        return ev

    def _fallback_compact(self) -> bool:
        """Governed fallback compaction (policy.fallback_compaction), used when an
        agent lets usage reach the hard limit without acting."""
        plan = self.plan(target_usage=max(0.3, self.policy.compaction_trigger - 0.3))
        applied = False
        for op in plan.ops:
            if op.op == "compact":
                r = self._run("compact", {"start": op.args["start"], "end": op.args["end"], "summary": None,
                                          "target_tokens": None}, level=AutonomyLevel.GOVERNED, actor="fallback")
            else:
                r = self._run("archive", {"tags": op.args["tags"]}, level=AutonomyLevel.GOVERNED,
                              actor="fallback")
            applied = applied or r.status == "applied"
        return applied


def _noop(_d: EditDecision) -> dict[str, Any]:  # pragma: no cover - never applied
    return {}


def _trace_args(kw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kw.items():
        if isinstance(v, str) and len(v) > 400:
            out[k] = v[:400] + "…"
        else:
            out[k] = v
    return out

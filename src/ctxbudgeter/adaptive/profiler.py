"""Behavioural profiling: how does a model actually manage its context?

:class:`BehaviorProfiler` aggregates :class:`~ctxbudgeter.runtime.trace.DecisionTrace`
records (from live sessions or :mod:`ctxbudgeter.bench` runs) into a per-model
:class:`ModelProfile` with the metrics NVIDIA's study found to separate models:
how often they compact, how hard, how often they reopen what they archived, and
*when* they start — plus governance-specific signals (denials, critical violations,
fact retention, overflow and recovery).

:func:`classify` maps a profile to named behaviours that drive policy tuning:

* ``avoidant``          rarely compacts; overflows or answers early under pressure
* ``anxious``           compacts while plenty of space remains
* ``over_compressing``  compresses hard and keeps reopening archives
* ``lossy``             task-critical facts are lost (in summaries or by session end)
* ``under_recovering``  needed archived facts are not recalled before answering
* ``stale_keeping``     answers with superseded values
* ``unsafe``            commits critical violations (e.g. touching protected context)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from statistics import mean, median
from typing import Any

from ..runtime.trace import DecisionTrace

BEHAVIORS = ("avoidant", "anxious", "over_compressing", "lossy", "under_recovering", "stale_keeping", "unsafe")


@dataclass
class ModelProfile:
    model: str
    source: str = "observed"  # observed | paper_prior
    sessions: int = 0
    n_traces: int = 0
    compaction_session_rate: float | None = None
    """Share of sessions with at least one compaction attempt."""
    median_first_compaction_usage: float | None = None
    median_compression_factor: float | None = None
    reopen_rate: float | None = None
    mean_fact_retention: float | None = None
    mean_critical_retention: float | None = None
    early_compaction_rate: float | None = None
    """Share of compaction attempts made below the usage floor."""
    denial_rate: float | None = None
    critical_violation_rate: float | None = None
    """Critical violations (blocked or committed) per session."""
    overflow_session_rate: float | None = None
    warning_answer_rate: float | None = None
    """Share of pressure warnings answered by stopping instead of compacting."""
    recovery_rate: float | None = None
    outcome_retention: float | None = None
    """Mean share of critical facts recoverable at session end (from task outcomes)."""
    stale_answer_rate: float | None = None
    accuracy: float | None = None
    behaviors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelProfile:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class ProfileThresholds:
    avoidant_compaction_rate: float = 0.15
    avoidant_pressure_signal: float = 0.3
    anxious_first_usage: float = 0.5
    anxious_early_rate: float = 0.3
    over_compression_factor: float = 3.5
    over_compression_reopen: float = 0.2
    lossy_critical_retention: float = 0.7
    """Salient-term retention of task-critical/pinned events across summaries."""
    lossy_outcome_retention: float = 0.8
    """Share of the task's critical facts still recoverable at session end."""
    under_recovery: float = 0.7
    stale_rate: float = 0.15


def classify(p: ModelProfile, th: ProfileThresholds | None = None) -> list[str]:
    th = th or ProfileThresholds()
    out: list[str] = []
    pressure = max(p.overflow_session_rate or 0.0, p.warning_answer_rate or 0.0)
    if p.compaction_session_rate is not None and p.compaction_session_rate < th.avoidant_compaction_rate \
            and pressure >= th.avoidant_pressure_signal:
        out.append("avoidant")
    if (p.median_first_compaction_usage is not None and p.median_first_compaction_usage < th.anxious_first_usage) \
            or (p.early_compaction_rate is not None and p.early_compaction_rate > th.anxious_early_rate):
        out.append("anxious")
    if p.median_compression_factor is not None and p.reopen_rate is not None \
            and p.median_compression_factor > th.over_compression_factor and p.reopen_rate > th.over_compression_reopen:
        out.append("over_compressing")
    if (p.mean_critical_retention is not None and p.mean_critical_retention < th.lossy_critical_retention) \
            or (p.outcome_retention is not None and p.outcome_retention < th.lossy_outcome_retention):
        out.append("lossy")
    if p.recovery_rate is not None and p.recovery_rate < th.under_recovery:
        out.append("under_recovering")
    if p.stale_answer_rate is not None and p.stale_answer_rate > th.stale_rate:
        out.append("stale_keeping")
    if p.critical_violation_rate:
        out.append("unsafe")
    return out


def _med(xs: list[float]) -> float | None:
    return round(median(xs), 4) if xs else None


def _avg(xs: list[float]) -> float | None:
    return round(mean(xs), 4) if xs else None


class BehaviorProfiler:
    """Build :class:`ModelProfile` objects from decision traces."""

    def __init__(self, *, floor: float = 0.5, thresholds: ProfileThresholds | None = None) -> None:
        self.floor = floor
        self.thresholds = thresholds or ProfileThresholds()

    def profile(self, traces: Iterable[DecisionTrace]) -> dict[str, ModelProfile]:
        by_model: dict[str, list[DecisionTrace]] = defaultdict(list)
        for t in traces:
            by_model[t.model].append(t)
        return {m: self._one(m, ts) for m, ts in sorted(by_model.items())}

    def _one(self, model: str, traces: list[DecisionTrace]) -> ModelProfile:
        sessions: dict[str, list[DecisionTrace]] = defaultdict(list)
        for t in traces:
            sessions[t.session_id].append(t)
        agent_traces = [t for t in traces if t.outcome.get("actor", "agent") == "agent"]
        comp = [t for t in agent_traces if t.op == "compact"]
        applied = [t for t in comp if t.verdict in ("applied", "modified")]
        firsts: list[float] = []
        for ts in sessions.values():
            cs = sorted((t for t in ts if t.op == "compact" and t.outcome.get("actor", "agent") == "agent"),
                        key=lambda t: t.step)
            if cs:
                firsts.append(cs[0].usage_before)
        ends = [t for t in traces if t.op == "session_end"]
        end_out = [t.outcome for t in ends]
        mutating = [t for t in agent_traces if t.op in ("compact", "archive", "pin", "restore")]
        p = ModelProfile(
            model=model,
            sessions=len(sessions),
            n_traces=len(traces),
            compaction_session_rate=round(sum(1 for ts in sessions.values() if any(
                t.op == "compact" and t.outcome.get("actor", "agent") == "agent" for t in ts)) / len(sessions), 4)
            if sessions else None,
            median_first_compaction_usage=_med(firsts),
            median_compression_factor=_med([f for f in (t.compression_factor for t in applied) if f]),
            reopen_rate=round(sum(1 for t in applied if t.outcome.get("reopened")) / len(applied), 4) if applied else None,
            mean_fact_retention=_avg([t.outcome.get("decision_notes", {}).get("fact_retention") for t in applied
                                      if t.outcome.get("decision_notes", {}).get("fact_retention") is not None]),
            mean_critical_retention=_avg([
                t.outcome.get("decision_notes", {}).get("critical_fact_retention") for t in applied
                if t.outcome.get("decision_notes", {}).get("critical_fact_retention") is not None]),
            early_compaction_rate=round(sum(1 for t in comp if t.usage_before < self.floor) / len(comp), 4) if comp else None,
            denial_rate=round(sum(1 for t in mutating if t.verdict == "denied") / len(mutating), 4) if mutating else None,
            critical_violation_rate=round(sum(1 for t in agent_traces if t.critical) / len(sessions), 4) if sessions else None,
            overflow_session_rate=_avg([1.0 if o.get("overflow_events_dropped", 0) > 0 else 0.0 for o in end_out]),
            warning_answer_rate=_avg([float(o["answered_on_warning"]) for o in end_out if "answered_on_warning" in o]),
            recovery_rate=_avg([o["recovery"] for o in end_out if o.get("recovery") is not None]),
            outcome_retention=_avg([o["retention"] for o in end_out if o.get("retention") is not None]),
            stale_answer_rate=_avg([1.0 if o.get("stale_answer") else 0.0 for o in end_out if "stale_answer" in o]),
            accuracy=_avg([1.0 if o.get("correct") else 0.0 for o in end_out if "correct" in o]),
        )
        p.behaviors = classify(p, self.thresholds)
        return p

"""Outcome-based labels and shaped rewards for context decisions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..runtime.trace import DecisionTrace

TRAINABLE_OPS = ("compact", "archive", "pin", "unpin", "recall", "restore", "checkpoint", "rollback")


@dataclass
class Label:
    trace_id: str
    label: str  # good | bad | neutral
    reward: float
    episode_return: float | None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "label": self.label, "reward": round(self.reward, 4),
                "episode_return": self.episode_return, "reasons": list(self.reasons)}


def label_trace(t: DecisionTrace, *, floor: float = 0.5, episode_correct: bool | None = None) -> Label:
    """Shaped reward in [-1, 1] for one decision; see module docstring of :mod:`ctxbudgeter.train`."""
    r = 0.0
    why: list[str] = []
    notes = t.outcome.get("decision_notes", {})
    if t.critical:
        r -= 1.0
        why.append("critical violation (protected context / secret)")
    if t.verdict == "denied":
        r -= 0.5
        why.append("denied by governance: " + ", ".join(v["rule"] for v in t.violations if
                                                          v.get("severity") in ("error", "critical")))
    if t.op == "compact":
        if t.usage_before < floor:
            r -= 0.4
            why.append(f"premature: usage {t.usage_before:.0%} < floor {floor:.0%}")
        if t.verdict in ("applied", "modified"):
            ret = notes.get("fact_retention")
            crit = notes.get("critical_fact_retention")
            cf = t.compression_factor or 1.0
            gain = min(1.0, math.log2(max(cf, 1.0)) / 2.0)  # 4x compression = full credit
            q = (ret if ret is not None else 0.5) * 0.5 + (crit if crit is not None else 1.0) * 0.5
            r += 0.6 * gain * q
            why.append(f"compression {cf:.1f}x, retention {ret}, critical retention {crit}")
            if t.outcome.get("reopened"):
                r -= 0.2
                why.append("archived content was reopened later")
            if crit is not None and crit < 0.7:
                r -= 0.3
                why.append("dropped task-critical facts")
    elif t.op == "archive" and t.verdict == "applied":
        r += 0.2
        why.append("archived redundant/inactive context")
    elif t.op in ("recall", "restore") and t.verdict == "applied":
        r += 0.15
        why.append("recovered archived information")
    elif t.op == "rollback" and t.verdict == "applied":
        r += 0.05
        why.append("reverted an edit")
    if episode_correct is not None:
        r += 0.2 if episode_correct else -0.2
        why.append("episode answered correctly" if episode_correct else "episode answered incorrectly")
    r = max(-1.0, min(1.0, r))
    lab = "good" if r >= 0.2 and not t.critical and t.verdict != "denied" else ("bad" if r < 0 else "neutral")
    ret = (1.0 if episode_correct else 0.0) if episode_correct is not None else None
    return Label(t.trace_id, lab, r, ret, why)


def label_traces(traces: Iterable[DecisionTrace], *, floor: float = 0.5) -> dict[str, Label]:
    traces = list(traces)
    correct: dict[str, bool] = {}
    for t in traces:
        if t.op == "session_end" and "correct" in t.outcome:
            correct[t.session_id] = bool(t.outcome["correct"])
    out: dict[str, Label] = {}
    by_sess: dict[str, list[DecisionTrace]] = defaultdict(list)
    for t in traces:
        by_sess[t.session_id].append(t)
    for sid, ts in by_sess.items():
        for t in ts:
            if t.op not in TRAINABLE_OPS or t.outcome.get("actor", "agent") != "agent":
                continue
            out[t.trace_id] = label_trace(t, floor=floor, episode_correct=correct.get(sid))
    return out

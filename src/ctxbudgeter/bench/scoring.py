"""Context Intelligence Score (CIS).

    CIS = 100 x (0.40·accuracy + 0.25·retention + 0.20·efficiency + 0.15·recovery)

* **accuracy**   — the task was answered correctly (latest values, all required facts)
* **retention**  — share of critical facts still recoverable at the end (active
  context, archive, or recalled) — what a harness that discards originals loses
* **efficiency** — mean headroom per request: ``1 - cumulative input tokens /
  (requests x available budget)``
* **recovery**   — share of needed-but-archived facts recalled before answering
  (not applicable when nothing needed recall; its weight is then redistributed)

A committed **critical governance violation** (e.g. removing protected context, a
secret re-entering context) overrides everything: CIS is 0 and the grade is ``fail``.
Weights are configurable and, as the proposal notes, require empirical validation;
the components are always reported alongside the score.
"""

from __future__ import annotations

DEFAULT_WEIGHTS: dict[str, float] = {"accuracy": 0.40, "retention": 0.25, "efficiency": 0.20, "recovery": 0.15}

GRADES: tuple[tuple[float, str], ...] = ((90.0, "expert"), (75.0, "competent"), (60.0, "developing"), (0.0, "novice"))


def grade_for(score: float, critical_violations: int = 0) -> str:
    if critical_violations:
        return "fail"
    return next(g for floor, g in GRADES if score >= floor)


def context_intelligence_score(
    *,
    accuracy: float,
    retention: float,
    efficiency: float,
    recovery: float | None,
    critical_violations: int = 0,
    weights: dict[str, float] | None = None,
) -> tuple[float, float, str]:
    """Return ``(cis, raw_score_ignoring_override, grade)``."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown CIS weights: {sorted(unknown)}")
        w.update(weights)
    comps = {"accuracy": accuracy, "retention": retention, "efficiency": efficiency, "recovery": recovery}
    for k, v in comps.items():
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"{k} must be within [0, 1], got {v}")
    active = {k: v for k, v in comps.items() if v is not None}
    total_w = sum(w[k] for k in active)
    raw = 100.0 * sum(w[k] * v for k, v in active.items()) / total_w if total_w else 0.0
    raw = round(raw, 2)
    cis = 0.0 if critical_violations else raw
    return cis, raw, grade_for(raw, critical_violations)

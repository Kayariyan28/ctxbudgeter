"""Compaction timing rubric.

Grounded in two findings:

* **SelfCompact** (Apple, arXiv 2606.23525): a lightweight decision rubric — compact
  once a sub-task is resolved or the trajectory is converging; suppress compaction
  mid-derivation or while stuck — is what makes model-triggered compaction beat
  fixed-interval summarization. Replacing the rubric with a yes/no prompt erased the
  gain.
* **NVIDIA, "Letting Models Manage Their Own Context"**: without a timing
  instruction some models compact far too early ("context anxiety": Kimi K3's median
  first compaction fell from ~84% to 29% of the budget), costing ~8 points of
  accuracy at large budgets; others (Claude Opus 5.5) answer early instead of
  compacting when warned.

:class:`CompactionRubric` turns usage plus the agent's declared *phase* into a
deterministic verdict. The governor uses it as an anti-anxiety gate, the planner uses
it for timing, and the context skill teaches agents the same rules in prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Phase = Literal["exploring", "mid_derivation", "stuck", "subtask_complete", "converging", "answering"]
PHASES: tuple[str, ...] = ("exploring", "mid_derivation", "stuck", "subtask_complete", "converging", "answering")

RubricDecision = Literal["compact_now", "allow", "defer", "block"]


@dataclass(frozen=True)
class RubricVerdict:
    decision: RubricDecision
    usage: float
    phase: str | None
    reasons: tuple[str, ...] = ()

    @property
    def permits_compaction(self) -> bool:
        return self.decision in ("compact_now", "allow")

    def to_dict(self) -> dict[str, object]:
        return {"decision": self.decision, "usage": round(self.usage, 4), "phase": self.phase,
                "reasons": list(self.reasons)}


@dataclass
class CompactionRubric:
    """Usage thresholds are fractions of the *available* context budget."""

    min_usage: float = 0.5
    """Below this, compaction is blocked (anti context-anxiety)."""
    trigger: float = 0.85
    """At or above this, compaction is recommended at a natural boundary."""
    hard_limit: float = 0.95
    """At or above this, compaction is required regardless of phase."""
    boundary_phases: tuple[str, ...] = field(default=("subtask_complete", "converging"))
    suppress_phases: tuple[str, ...] = field(default=("mid_derivation", "stuck"))

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_usage <= self.trigger <= self.hard_limit <= 1.0:
            raise ValueError("require 0 <= min_usage <= trigger <= hard_limit <= 1")

    def evaluate(self, usage: float, phase: str | None = None) -> RubricVerdict:
        if usage >= self.hard_limit:
            return RubricVerdict("compact_now", usage, phase,
                                 (f"usage {usage:.0%} >= hard limit {self.hard_limit:.0%}",))
        if usage < self.min_usage:
            return RubricVerdict("block", usage, phase,
                                 (f"usage {usage:.0%} < floor {self.min_usage:.0%}: space remains; "
                                  "compacting now discards context you can still afford",))
        if phase in self.suppress_phases:
            return RubricVerdict("defer", usage, phase,
                                 (f"phase '{phase}': finish the current derivation before compacting",))
        if usage >= self.trigger:
            why: tuple[str, ...] = (f"usage {usage:.0%} >= trigger {self.trigger:.0%}",)
            if phase in self.boundary_phases:
                why += (f"phase '{phase}' is a natural compaction boundary",)
            return RubricVerdict("compact_now", usage, phase, why)
        if phase in self.boundary_phases:
            return RubricVerdict("allow", usage, phase,
                                 (f"phase '{phase}' is a natural boundary; compaction permitted",))
        return RubricVerdict("defer", usage, phase,
                             (f"usage {usage:.0%} below trigger {self.trigger:.0%}; no boundary signalled",))

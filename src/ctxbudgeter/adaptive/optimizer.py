"""Policy optimization and validation — the Level-2 learning loop.

Rule-based tuning proposes; evidence decides. A tuned policy can over-correct (a
compression cap that an agent cannot adapt to just blocks it), so
:class:`PolicyOptimizer` evaluates a small, interpretable candidate set on
*calibration* tasks and keeps the best:

* the fully tuned policy,
* one ablation per detected behaviour (the tuned policy without that rule),
* the base policy with only the governed fallback,
* the base policy.

:func:`validate_policy` then checks a policy against a baseline mode on tasks the
optimizer did not see, and :class:`~ctxbudgeter.adaptive.store.PolicyStore` records
the verdict. Only validated policies become active; transfers to a new model start
unvalidated (see :meth:`AdaptivePolicy.transfer_to`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Any

from ..governance.edits import ContextEditPolicy
from .profiler import ModelProfile
from .tuner import AdaptivePolicy, PolicyTuner

if TYPE_CHECKING:
    from ..bench.agents import ContextAgent
    from ..bench.tasks import BenchTask
    from ..tokenizer import TokenCounter


@dataclass
class OptimizationResult:
    best: AdaptivePolicy
    leaderboard: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"best": self.best.to_dict(), "leaderboard": list(self.leaderboard)}


def _metrics(results: Sequence[Any]) -> dict[str, Any]:
    return {
        "episodes": len(results),
        "cis": round(mean(r.cis for r in results), 2) if results else 0.0,
        "accuracy": round(mean(1.0 if r.correct else 0.0 for r in results), 4) if results else 0.0,
        "retention": round(mean(r.retention for r in results), 4) if results else 0.0,
        "critical_violations": sum(r.critical_violations for r in results),
        "mean_cumulative_input_tokens": round(mean(r.cumulative_input_tokens for r in results)) if results else 0,
    }


def evaluate_policy(
    policy: AdaptivePolicy | None,
    agent_factory: Callable[[], ContextAgent],
    tasks: Sequence[BenchTask],
    *,
    budgets: Sequence[int],
    model: str,
    mode: str = "adaptive",
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    from ..bench.runner import run_episode

    res = [run_episode(t, agent_factory(), mode=mode, budget=b, model=model, adaptive=policy, counter=counter)
           for t in tasks for b in budgets]
    return _metrics(res)


class PolicyOptimizer:
    def __init__(self, agent_factory: Callable[[], ContextAgent], tasks: Sequence[BenchTask], *,
                 budgets: Sequence[int], model: str, counter: TokenCounter | None = None,
                 base: ContextEditPolicy | None = None) -> None:
        self.agent_factory = agent_factory
        self.tasks = list(tasks)
        self.budgets = list(budgets)
        self.model = model
        self.counter = counter
        self.tuner = PolicyTuner(base)

    def candidates(self, profile: ModelProfile) -> list[tuple[str, AdaptivePolicy]]:
        out: list[tuple[str, AdaptivePolicy]] = [("tuned", self.tuner.tune(profile))]
        for b in profile.behaviors:
            p = ModelProfile.from_dict(profile.to_dict())
            p.behaviors = [x for x in profile.behaviors if x != b]
            out.append((f"tuned_without_{b}", self.tuner.tune(p)))
        empty = ModelProfile(model=profile.model, behaviors=[])
        base = self.tuner.tune(empty)
        fb = AdaptivePolicy(model=profile.model, policy=base.policy.with_overrides(fallback_compaction=True),
                            rules_applied=["base policy + governed fallback compaction"], version=1,
                            evidence={"profile": profile.to_dict(), "overrides": {"fallback_compaction": True}})
        out += [("base_with_fallback", fb), ("base", base)]
        seen: set[str] = set()
        uniq: list[tuple[str, AdaptivePolicy]] = []
        for name, pol in out:
            key = json.dumps([pol.policy.to_dict(), pol.guidance], sort_keys=True)
            if key not in seen:
                seen.add(key)
                uniq.append((name, pol))
        return uniq

    def optimize(self, profile: ModelProfile) -> OptimizationResult:
        board: list[dict[str, Any]] = []
        scored: list[tuple[tuple[int, float, int], str, AdaptivePolicy]] = []
        for name, pol in self.candidates(profile):
            m = evaluate_policy(pol, self.agent_factory, self.tasks, budgets=self.budgets, model=self.model,
                                counter=self.counter)
            board.append({"candidate": name, "policy_id": pol.policy_id, "rules": pol.rules_applied, **m})
            scored.append(((-m["critical_violations"], m["cis"], -m["mean_cumulative_input_tokens"]), name, pol))
        scored.sort(key=lambda s: s[0], reverse=True)
        _, best_name, best = scored[0]
        best.evidence["optimization"] = {"selected": best_name, "calibration_tasks": [t.task_id for t in self.tasks],
                                         "budgets": self.budgets}
        board.sort(key=lambda r: (r["critical_violations"], -r["cis"], r["mean_cumulative_input_tokens"]))
        return OptimizationResult(best, board)


def validate_policy(
    policy: AdaptivePolicy,
    agent_factory: Callable[[], ContextAgent],
    tasks: Sequence[BenchTask],
    *,
    budgets: Sequence[int],
    model: str,
    baseline_mode: str = "guided",
    tolerance: float = 1.0,
    counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Pass if the policy has no critical violations and its mean CIS is within
    ``tolerance`` points of (or above) the baseline mode on ``tasks``."""
    cand = evaluate_policy(policy, agent_factory, tasks, budgets=budgets, model=model, counter=counter)
    base = evaluate_policy(None, agent_factory, tasks, budgets=budgets, model=model, mode=baseline_mode,
                           counter=counter)
    delta = round(cand["cis"] - base["cis"], 2)
    passed = cand["critical_violations"] == 0 and delta >= -tolerance
    return {"passed": passed, "baseline_mode": baseline_mode, "candidate": cand, "baseline": base,
            "delta_cis": delta, "tasks": [t.task_id for t in tasks], "budgets": list(budgets)}

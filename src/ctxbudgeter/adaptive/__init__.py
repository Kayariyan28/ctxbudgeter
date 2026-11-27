"""ctxbudgeter.adaptive — behavioural learning (Level 2) without touching model weights.

Observe how a model manages context (decision traces) → profile its behaviour →
tune governance thresholds, fallbacks and guidance → validate on held-out tasks →
version and store the policy → transfer to other models only as unvalidated
candidates.

    from ctxbudgeter.adaptive import BehaviorProfiler, PolicyTuner, resolve_profile

    profile = resolve_profile("kimi-k3", traces)      # observed > published prior
    policy = PolicyTuner().tune(profile)
    rt = ContextRuntime(budget=32_000, policy=policy.policy, autonomy="ADAPTIVE")
    system_prompt += policy.render_guidance(autonomy=4)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..runtime.trace import DecisionTrace
from .priors import PAPER_PRIORS, prior_for
from .profiler import BEHAVIORS, BehaviorProfiler, ModelProfile, ProfileThresholds, classify
from .store import PolicyStore
from .tuner import AdaptivePolicy, PolicyTuner

if TYPE_CHECKING:
    from .optimizer import OptimizationResult, PolicyOptimizer, validate_policy


def resolve_profile(model: str, traces: Iterable[DecisionTrace] = (), *, min_sessions: int = 3) -> ModelProfile:
    """Observed profile if there are at least ``min_sessions`` sessions for ``model``;
    otherwise the published prior; otherwise an empty profile."""
    observed = BehaviorProfiler().profile([t for t in traces if t.model == model]).get(model)
    if observed is not None and observed.sessions >= min_sessions:
        return observed
    prior = prior_for(model)
    if prior is not None:
        return prior
    return observed or ModelProfile(model=model)


def __getattr__(name: str) -> Any:
    if name in ("PolicyOptimizer", "OptimizationResult", "validate_policy"):
        from . import optimizer

        return getattr(optimizer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BEHAVIORS",
    "PAPER_PRIORS",
    "AdaptivePolicy",
    "BehaviorProfiler",
    "ModelProfile",
    "OptimizationResult",
    "PolicyOptimizer",
    "PolicyStore",
    "PolicyTuner",
    "ProfileThresholds",
    "classify",
    "prior_for",
    "resolve_profile",
    "validate_policy",
]

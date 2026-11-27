"""Behavioural priors from published measurements.

Before ctxbudgeter has observed a model, it can start from what the literature
reports. The figures below are taken from NVIDIA's technical report *Letting Models
Manage Their Own Context: Where We Stand* (arXiv 2611.34147, Nov 2026), measured with
NOOA's context API on OfficeQA Pro v2 and MLE-bench Lite:

* share of tasks with at least one compaction (Fig. 5),
* median compression factor per compaction and share of compactions later reopened (Fig. 6),
* behaviour at the first context-pressure warning at a 64K budget (Fig. 7),
* median usage at the first compaction *without* a timing instruction (Fig. 8, 128K;
  DeepSeek at 256K).

They are priors, not measurements of your deployment: profiles built from your own
traces always take precedence (see :func:`ctxbudgeter.adaptive.resolve_profile`).
"""

from __future__ import annotations

from .profiler import ModelProfile, classify

_SRC = "NVIDIA arXiv 2611.34147"

PAPER_PRIORS: dict[str, ModelProfile] = {
    "gpt-6.1-sol": ModelProfile(
        model="gpt-6.1-sol", source="paper_prior", compaction_session_rate=0.42,
        median_compression_factor=3.7, reopen_rate=0.186, warning_answer_rate=0.19,
        notes=[f"{_SRC}: reads API docs before compacting in 69% of warnings; timing least "
               "sensitive to the instruction; ~3x score at 32K on OfficeQA Pro v2"],
    ),
    "kimi-k3": ModelProfile(
        model="kimi-k3", source="paper_prior", compaction_session_rate=0.45,
        median_compression_factor=2.1, reopen_rate=0.069, warning_answer_rate=0.13,
        median_first_compaction_usage=0.29,
        notes=[f"{_SRC}: first compaction falls from ~84% to 29% of budget without the timing "
               "instruction (128K); ~8 points below baseline at >=128K in that setting"],
    ),
    "deepseek-v4-pro": ModelProfile(
        model="deepseek-v4-pro", source="paper_prior", compaction_session_rate=0.53,
        median_compression_factor=1.8, reopen_rate=0.035, warning_answer_rate=0.19,
        median_first_compaction_usage=0.40,
        notes=[f"{_SRC}: first compaction falls from ~83% to 40% without the timing instruction "
               "(256K); inspects history before compacting in 46% of warnings"],
    ),
    "claude-opus-5.5": ModelProfile(
        model="claude-opus-5.5", source="paper_prior", compaction_session_rate=0.04,
        median_compression_factor=1.5, reopen_rate=0.141, warning_answer_rate=0.88,
        median_first_compaction_usage=0.16,
        notes=[f"{_SRC}: answers instead of compacting in 88% of warnings; with the timing "
               "instruction almost never compacts, without it compacts early (16% at 128K)"],
    ),
}
for _p in PAPER_PRIORS.values():
    _p.behaviors = classify(_p)


def _norm(model: str) -> str:
    return model.lower().replace("_", "-").replace(" ", "-").replace("claude-opus-5-5", "claude-opus-5.5") \
        .replace("gpt-6-1", "gpt-6.1")


def prior_for(model: str) -> ModelProfile | None:
    """Best-matching published prior for a model id, or ``None``."""
    m = _norm(model)
    for key, prof in PAPER_PRIORS.items():
        if m == key or m.startswith(key) or key.startswith(m):
            return ModelProfile.from_dict(prof.to_dict())
    return None

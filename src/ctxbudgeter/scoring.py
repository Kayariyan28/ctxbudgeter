"""Deterministic ranking of optional context items.

The compiler ranks optional items by `score_item` and includes them greedily until
the token budget is exhausted. The score formula is intentionally simple and
explainable — every component is exposed so callers can override weights.
"""

from __future__ import annotations

from typing import Optional

from .item import ContextItem

DEFAULT_WEIGHTS: dict[str, float] = {
    "priority": 0.5,   # priority is already 0-100
    "relevance": 0.3,  # multiplied by 100
    "freshness": 0.1,  # multiplied by 100
    "cache": 0.1,      # multiplied by 100
    "cost": 1.0,       # token cost penalty multiplier
}


def _cache_value(item: ContextItem) -> float:
    """0–1 value for the cache component: stable > ephemeral > dynamic."""
    if item.cache_policy == "stable":
        return 1.0
    if item.cache_policy == "ephemeral":
        return 0.4
    return 0.0


def score_item(
    item: ContextItem,
    token_count: int,
    available_tokens: int,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Return a deterministic ranking score for an optional item.

    score = priority*w_p + relevance*100*w_r + freshness*100*w_f + cache*100*w_c - cost_penalty

    At max values (priority=100, relevance=1, freshness=1, cache=1) the raw score is
    100 before the cost penalty. cost_penalty grows up to ~30 when token_count
    approaches the full available budget.
    """
    w = DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)
    token_ratio = token_count / max(1, available_tokens)
    cost_penalty = w["cost"] * (token_ratio * 30.0)
    return (
        item.priority * w["priority"]
        + item.relevance * 100.0 * w["relevance"]
        + item.freshness * 100.0 * w["freshness"]
        + _cache_value(item) * 100.0 * w["cache"]
        - cost_penalty
    )

"""Tests for the scoring function."""

from __future__ import annotations

from ctxbudgeter import ContextItem, score_item


def test_higher_priority_scores_higher() -> None:
    a = ContextItem(name="a", content="x", priority=90)
    b = ContextItem(name="b", content="x", priority=10)
    s_a = score_item(a, 10, 1000)
    s_b = score_item(b, 10, 1000)
    assert s_a > s_b


def test_higher_relevance_scores_higher() -> None:
    a = ContextItem(name="a", content="x", priority=50, relevance=0.9)
    b = ContextItem(name="b", content="x", priority=50, relevance=0.1)
    assert score_item(a, 10, 1000) > score_item(b, 10, 1000)


def test_stable_cache_beats_dynamic() -> None:
    a = ContextItem(name="a", content="x", priority=50, cache_policy="stable")
    b = ContextItem(name="b", content="x", priority=50, cache_policy="dynamic")
    assert score_item(a, 10, 1000) > score_item(b, 10, 1000)


def test_larger_tokens_get_cost_penalty() -> None:
    a = ContextItem(name="a", content="x", priority=50)
    cheap = score_item(a, 10, 1000)
    expensive = score_item(a, 800, 1000)
    assert cheap > expensive


def test_freshness_contributes() -> None:
    fresh = ContextItem(name="a", content="x", priority=50, freshness=1.0)
    stale = ContextItem(name="b", content="x", priority=50, freshness=0.1)
    assert score_item(fresh, 10, 1000) > score_item(stale, 10, 1000)


def test_weights_override_defaults() -> None:
    a = ContextItem(name="a", content="x", priority=10, relevance=0.9)
    # Heavily favor relevance
    high_rel = score_item(a, 10, 1000, weights={"priority": 0.0, "relevance": 1.0})
    low_rel = score_item(a, 10, 1000, weights={"priority": 0.0, "relevance": 0.0})
    assert high_rel > low_rel

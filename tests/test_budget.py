"""Tests for tokenizer & budget arithmetic."""

from __future__ import annotations

import pytest

from ctxbudgeter import ContextPack, TokenCounter


def test_token_counter_empty_is_zero() -> None:
    assert TokenCounter("claude-sonnet-4.6").count("") == 0


def test_token_counter_monotonic() -> None:
    c = TokenCounter("claude-sonnet-4.6")
    short = c.count("hello")
    long_ = c.count("hello " * 100)
    assert long_ > short


def test_heuristic_lower_bound_is_one() -> None:
    c = TokenCounter("nonexistent-model")
    assert c.count("x") >= 1


def test_token_counter_backend_label() -> None:
    c = TokenCounter("claude-sonnet-4.6")
    assert c.backend in ("tiktoken", "heuristic")


def test_pack_rejects_zero_budget() -> None:
    with pytest.raises(ValueError):
        ContextPack(token_budget=0)


def test_pack_rejects_negative_reserved() -> None:
    with pytest.raises(ValueError):
        ContextPack(token_budget=1000, reserved_output_tokens=-1)


def test_pack_rejects_reserved_geq_budget() -> None:
    with pytest.raises(ValueError):
        ContextPack(token_budget=1000, reserved_output_tokens=1000)
    with pytest.raises(ValueError):
        ContextPack(token_budget=1000, reserved_output_tokens=2000)


def test_available_tokens_math() -> None:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_500)
    pack.add(name="x", content="hi", required=True)
    compiled = pack.compile()
    assert compiled.available_tokens == 8_500
    assert compiled.used_tokens >= 1


def test_estimate_tokens_sums_items() -> None:
    pack = ContextPack(token_budget=10_000)
    pack.add(name="a", content="hello world hello world hello world")
    pack.add(name="b", content="x" * 200)
    est = pack.estimate_tokens()
    assert est > 0


def test_utilization_in_range() -> None:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="task", content="do thing", required=True)
    compiled = pack.compile()
    assert 0.0 <= compiled.utilization <= 1.0

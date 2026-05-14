"""Tests for ctxbudgeter.testing (eval/assert layer + GoldenPack)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxbudgeter import ContextPack
from ctxbudgeter.testing import (
    ContextAssertionError,
    GoldenPack,
    assert_cacheable_prefix_at_least,
    assert_excludes,
    assert_health_at_least,
    assert_includes,
    assert_includes_in_order,
    assert_no_compression_of,
    assert_no_loader_failures,
    assert_no_secret_items,
    assert_used_tokens_at_most,
    assert_utilization_between,
    assert_warnings_empty,
)


def _good_pack() -> ContextPack:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="rules" * 100, kind="system", priority=100,
             cache_policy="stable", required=True)
    pack.add(name="task", content="do thing", kind="task", priority=95, required=True)
    pack.add(name="doc", content="docs " * 50, kind="project_doc", priority=70,
             cache_policy="stable")
    return pack


def test_assert_includes_passes() -> None:
    compiled = _good_pack().compile()
    assert_includes(compiled, "sys", "task")


def test_assert_includes_fails_with_clear_message() -> None:
    compiled = _good_pack().compile()
    with pytest.raises(ContextAssertionError, match="missing|excluded"):
        assert_includes(compiled, "nonexistent")


def test_assert_excludes_passes() -> None:
    compiled = _good_pack().compile()
    assert_excludes(compiled, "ghost")  # not in pack → counts as excluded


def test_assert_excludes_fails_if_included() -> None:
    compiled = _good_pack().compile()
    with pytest.raises(ContextAssertionError):
        assert_excludes(compiled, "sys")


def test_assert_health_at_least() -> None:
    compiled = _good_pack().compile()
    assert_health_at_least(compiled, 50)
    with pytest.raises(ContextAssertionError):
        assert_health_at_least(compiled, 200)


def test_assert_cacheable_prefix_at_least() -> None:
    compiled = _good_pack().compile()
    # Should be > 0 because we have stable items
    assert_cacheable_prefix_at_least(compiled, 1)
    with pytest.raises(ContextAssertionError):
        assert_cacheable_prefix_at_least(compiled, 1_000_000)


def test_assert_no_secret_items_passes_for_clean_pack() -> None:
    compiled = _good_pack().compile()
    assert_no_secret_items(compiled)


def test_assert_no_secret_items_fails_when_secret_included() -> None:
    pack = _good_pack()
    pack.add(name="secret_token", content="sk-12345", sensitivity="secret")
    pack.set_secret_policy("warn")  # include but warn
    with pytest.raises(ContextAssertionError, match="secret"):
        assert_no_secret_items(pack.compile())


def test_assert_no_compression_of() -> None:
    compiled = _good_pack().compile()
    assert_no_compression_of(compiled, "sys")


def test_assert_utilization_between() -> None:
    compiled = _good_pack().compile()
    assert_utilization_between(compiled, 0.0, 1.0)
    with pytest.raises(ContextAssertionError):
        assert_utilization_between(compiled, 0.99, 1.0)


def test_assert_used_tokens_at_most() -> None:
    compiled = _good_pack().compile()
    assert_used_tokens_at_most(compiled, 100_000)
    with pytest.raises(ContextAssertionError):
        assert_used_tokens_at_most(compiled, 1)


def test_assert_includes_in_order() -> None:
    compiled = _good_pack().compile()
    # 'sys' should appear before 'task' (stable cache first, then dynamic)
    assert_includes_in_order(compiled, "sys", "task")
    with pytest.raises(ContextAssertionError):
        assert_includes_in_order(compiled, "task", "sys")


def test_assert_no_loader_failures() -> None:
    compiled = _good_pack().compile()
    assert_no_loader_failures(compiled)


def test_assert_warnings_empty() -> None:
    compiled = _good_pack().compile()
    assert_warnings_empty(compiled)


def test_golden_pack_creates_on_first_run(tmp_path: Path) -> None:
    golden = GoldenPack(tmp_path / "g.json")
    compiled = _good_pack().compile()
    golden.check(compiled)
    assert (tmp_path / "g.json").exists()


def test_golden_pack_passes_on_match(tmp_path: Path) -> None:
    p = tmp_path / "g.json"
    GoldenPack(p).check(_good_pack().compile())
    GoldenPack(p).check(_good_pack().compile())  # second run: must pass


def test_golden_pack_fails_on_drift(tmp_path: Path) -> None:
    p = tmp_path / "g.json"
    GoldenPack(p).check(_good_pack().compile())
    # Build a different pack
    drifted = _good_pack()
    drifted.add(name="new_item", content="new content")
    with pytest.raises(ContextAssertionError, match="Golden mismatch"):
        GoldenPack(p).check(drifted.compile())


def test_golden_pack_update_refreshes(tmp_path: Path) -> None:
    p = tmp_path / "g.json"
    GoldenPack(p).check(_good_pack().compile())
    drifted = _good_pack()
    drifted.add(name="new_item", content="new content")
    GoldenPack(p, update=True).check(drifted.compile())  # should not raise
    # Subsequent run with the drifted pack now passes
    GoldenPack(p).check(drifted.compile())

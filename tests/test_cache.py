"""Tests for CachePlanner."""

from __future__ import annotations

from ctxbudgeter import CachePlanner, ContextPack


def test_stable_prefix_detected() -> None:
    pack = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    pack.add(name="sys", content="rules " * 30, kind="system", required=True, cache_policy="stable")
    pack.add(name="doc", content="docs " * 30, kind="project_doc", priority=80, cache_policy="stable")
    pack.add(name="task", content="do", kind="task", required=True, cache_policy="dynamic")
    plan = CachePlanner().analyze(pack.compile())
    assert plan.stable_prefix_length >= 2
    assert plan.cacheable_token_estimate > 0


def test_timestamp_in_stable_prefix_warns() -> None:
    pack = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    pack.add(name="sys", content="System rules. Generated 2026-06-01T12:30:00 today.",
             kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="do", kind="task", required=True)
    plan = CachePlanner().analyze(pack.compile())
    assert any("timestamp" in w.lower() for w in plan.warnings)


def test_uuid_in_stable_prefix_warns() -> None:
    pack = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    pack.add(name="sys", content="session 12345678-1234-1234-1234-123456789abc rules",
             kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="do", kind="task", required=True)
    plan = CachePlanner().analyze(pack.compile())
    assert any("uuid" in w.lower() for w in plan.warnings)


def test_stable_after_dynamic_warns() -> None:
    # Force a stable item to land after a dynamic one by ordering via priority.
    pack = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    pack.add(name="dyn", content="user said hi " * 10, kind="user_message", priority=99, cache_policy="dynamic")
    pack.add(name="late_stable", content="stable doc " * 10, kind="project_doc", priority=10, cache_policy="stable")
    pack.add(name="task", content="do", kind="task", required=True)
    compiled = pack.compile()
    plan = CachePlanner().analyze(compiled)
    # If the compiled order places stable after dynamic, we should warn.
    names = [it.name for it in compiled.included_items]
    if names.index("dyn") < names.index("late_stable"):
        assert any("after dynamic" in w.lower() for w in plan.warnings)


def test_analyze_bom() -> None:
    pack = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    pack.add(name="sys", content="rules " * 30, kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="do", kind="task", required=True)
    bom = pack.compile().bom
    plan = CachePlanner().analyze_bom(bom)
    assert plan.stable_prefix_length >= 1
    assert isinstance(plan.cache_efficiency_score, int)

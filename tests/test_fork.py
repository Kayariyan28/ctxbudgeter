"""Tests for ContextPack.fork (Isolate strategy)."""

from __future__ import annotations

from ctxbudgeter import ContextPack


def _parent() -> ContextPack:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="task body", kind="task", required=True)
    pack.add(name="code_file", content="def foo(): pass", kind="code", priority=50,
             metadata={"namespace": "backend"})
    pack.add(name="ui_file", content="button styles", kind="code", priority=50,
             metadata={"namespace": "frontend"})
    return pack


def test_fork_no_filter_clones_all() -> None:
    parent = _parent()
    child = parent.fork()
    assert len(child.items) == len(parent.items)


def test_fork_with_filter_keeps_subset() -> None:
    parent = _parent()
    child = parent.fork(filter=lambda it: it.kind == "code")
    names = {it.name for it in child.items}
    assert names == {"code_file", "ui_file"}


def test_fork_different_budget() -> None:
    parent = _parent()
    child = parent.fork(token_budget=2_000, reserved_output_tokens=500)
    assert child.token_budget == 2_000
    assert child.reserved_output_tokens == 500


def test_subset_by_kind() -> None:
    parent = _parent()
    child = parent.subset_by_kind("system", "task")
    names = {it.name for it in child.items}
    assert names == {"sys", "task"}


def test_subset_by_namespace() -> None:
    parent = _parent()
    backend = parent.subset_by_namespace("backend")
    frontend = parent.subset_by_namespace("frontend")
    assert {it.name for it in backend.items} == {"code_file"}
    assert {it.name for it in frontend.items} == {"ui_file"}


def test_fork_is_independent_mutation() -> None:
    parent = _parent()
    child = parent.fork()
    child.add(name="child_only", content="x")
    assert "child_only" in child
    assert "child_only" not in parent

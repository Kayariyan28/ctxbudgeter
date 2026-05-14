"""Tests for lazy References (just-in-time context loading)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxbudgeter import ContextPack, Reference
from ctxbudgeter.loaders import (
    file_loader,
    get_loader,
    inline_loader,
    list_loaders,
    register_loader,
)


def test_reference_resolves_inline() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add_reference(
        name="hello",
        location="Hello world from inline loader.",
        loader=inline_loader,
        priority=70,
    )
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["hello"] == "included"
    assert "hello" in {it.name for it in compiled.included_items}


def test_reference_file_loader(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Docs\nUseful info.")
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add_reference(
        name="doc", location=str(f), loader=file_loader, priority=70, cache_policy="stable",
    )
    compiled = pack.compile()
    assert compiled.content_for("doc") == "# Docs\nUseful info."


def test_reference_skipped_when_estimate_overflows() -> None:
    pack = ContextPack(token_budget=2_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    calls = []

    def expensive_loader(ref):
        calls.append(ref.name)
        return "x" * 10_000

    pack.add_reference(
        name="too_big",
        location="-",
        loader=expensive_loader,
        estimated_tokens=10_000,  # way over remaining budget
        priority=30,
    )
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["too_big"] == "excluded"
    assert calls == [], "loader should not run when estimated tokens won't fit"
    assert "won't fit" in next(d.reason for d in compiled.decisions if d.name == "too_big")


def test_reference_loader_failure_excluded_with_reason() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)

    def bad_loader(ref):
        raise RuntimeError("network down")

    pack.add_reference(name="api", location="-", loader=bad_loader, priority=70)
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["api"] == "excluded"
    assert "loader failed" in next(d.reason for d in compiled.decisions if d.name == "api")
    assert any("api" in w for w in compiled.warnings)


def test_reference_async_loader_rejected_by_sync_compile() -> None:
    async def async_loader(ref):
        return "async content"

    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add_reference(name="async_ref", location="-", loader=async_loader, priority=70)
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["async_ref"] == "excluded"
    assert "acompile" in next(d.reason for d in compiled.decisions if d.name == "async_ref")


async def test_reference_async_loader_works_with_acompile() -> None:
    async def async_loader(ref):
        return "async content body"

    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add_reference(name="async_ref", location="-", loader=async_loader, priority=70)
    compiled = await pack.acompile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["async_ref"] == "included"
    assert compiled.content_for("async_ref") == "async content body"


def test_reference_no_loader_excluded() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add_reference(name="nope", location="-", loader=None, priority=70)
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["nope"] == "excluded"


def test_reference_duplicate_name_rejected() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="x", content="a")
    with pytest.raises(ValueError):
        pack.add_reference(name="x", location="-", loader=inline_loader)


def test_register_loader_decorator() -> None:
    @register_loader("custom_test_loader")
    def my_loader(ref):
        return f"loaded {ref.name}"

    assert "custom_test_loader" in list_loaders()
    assert get_loader("custom_test_loader") is my_loader


def test_get_loader_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_loader("definitely-not-registered")


def test_reference_construction_validates() -> None:
    with pytest.raises(ValueError):
        Reference(name="  ", location="-", loader=inline_loader)
    with pytest.raises(ValueError):
        Reference(name="x", location="-", loader=inline_loader, priority=200)
    with pytest.raises(ValueError):
        Reference(name="x", location="-", loader=inline_loader, estimated_tokens=-1)

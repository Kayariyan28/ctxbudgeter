"""Tests for async compilation, async compressors, and compression retry."""

from __future__ import annotations

from ctxbudgeter import ContextPack, Reference
from ctxbudgeter.loaders import inline_loader


async def test_acompile_basic() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="x", required=True)
    compiled = await pack.acompile()
    assert compiled.used_tokens > 0
    assert {it.name for it in compiled.included_items} >= {"sys", "task"}


async def test_acompile_with_async_compressor() -> None:
    pack = ContextPack(token_budget=1_500, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add(name="huge", content="y" * 30_000, priority=80, compressible=True)

    async def compress(item, target):
        return f"summary of {item.name}"

    pack.set_compressor(compress)
    compiled = await pack.acompile()
    by = {d.name: d for d in compiled.decisions}
    assert by["huge"].status == "compressed"
    assert by["huge"].tokens < by["huge"].original_tokens


async def test_acompile_resolves_async_references_concurrently() -> None:
    import asyncio

    async def slow_loader(ref):
        await asyncio.sleep(0.01)
        return f"loaded {ref.name}"

    pack = ContextPack(token_budget=10_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add_reference(name="ref1", location="-", loader=slow_loader, priority=70)
    pack.add_reference(name="ref2", location="-", loader=slow_loader, priority=70)
    pack.add_reference(name="ref3", location="-", loader=slow_loader, priority=70)

    compiled = await pack.acompile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["ref1"] == "included"
    assert statuses["ref2"] == "included"
    assert statuses["ref3"] == "included"


def test_sync_compile_warns_on_async_compressor() -> None:
    pack = ContextPack(token_budget=1_500, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add(name="huge", content="x" * 30_000, priority=80, compressible=True)

    async def compress(item, target):
        return "summary"

    pack.set_compressor(compress)
    compiled = pack.compile()
    # Should NOT crash; should warn and exclude
    assert any("acompile" in w for w in compiled.warnings)


def test_compressor_retry_on_overshoot() -> None:
    pack = ContextPack(token_budget=1_500, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add(name="huge", content="y" * 30_000, priority=80, compressible=True)

    calls: list[int] = []

    def compress(item, target):
        calls.append(target)
        if len(calls) == 1:
            return "x" * 8_000  # overshoot
        return "small summary"

    pack.set_compressor(compress)
    compiled = pack.compile()
    assert len(calls) == 2, f"expected retry on overshoot; got {len(calls)} call(s)"
    assert calls[1] < calls[0], "retry target should be tighter than first attempt"


def test_compressor_retry_disabled_when_config_off() -> None:
    from ctxbudgeter import CompilerConfig

    pack = ContextPack(token_budget=1_500, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add(name="huge", content="y" * 30_000, priority=80, compressible=True)

    calls: list[int] = []

    def compress(item, target):
        calls.append(target)
        return "x" * 8_000

    pack.set_compressor(compress)
    pack._config.compression_retry = False
    pack.compile()
    assert len(calls) == 1, "retry should be disabled"

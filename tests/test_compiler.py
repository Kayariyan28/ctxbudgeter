"""Tests for the deterministic compiler."""

from __future__ import annotations

import pytest

from ctxbudgeter import (
    BudgetExceededError,
    CompilerConfig,
    ContextItem,
    ContextPack,
    compile_items,
)


def _pack(budget: int = 5_000, reserved: int = 500) -> ContextPack:
    return ContextPack(model="claude-sonnet-4.6", token_budget=budget, reserved_output_tokens=reserved)


def test_required_items_included_first() -> None:
    pack = _pack()
    pack.add(name="sys", content="rules", kind="system", priority=100, required=True, cache_policy="stable")
    pack.add(name="task", content="do thing", kind="task", priority=95, required=True)
    pack.add(name="extra", content="x" * 50, kind="other", priority=10)
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["sys"] == "included"
    assert statuses["task"] == "included"


def test_excluded_when_over_budget() -> None:
    pack = _pack(budget=1500, reserved=500)
    # Total available is 1000 tokens. We'll fill with required, then add huge optional.
    pack.add(name="task", content="x" * 200, required=True)  # ~50 tokens
    pack.add(name="huge", content="y" * 20_000, priority=10)  # too big
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["task"] == "included"
    assert statuses["huge"] == "excluded"


def test_higher_score_wins_when_budget_tight() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(name="task", content="x", required=True)
    # Two items, only one will fit. Same size, different priority.
    pack.add(name="hi_pri", content="y" * 1500, priority=90)
    pack.add(name="lo_pri", content="z" * 1500, priority=10)
    compiled = pack.compile()
    by = {d.name: d for d in compiled.decisions}
    assert by["hi_pri"].status == "included"
    assert by["lo_pri"].status == "excluded"


def test_required_over_budget_raises_without_compression() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(name="huge", content="x" * 50_000, required=True)
    with pytest.raises(BudgetExceededError):
        pack.compile()


def test_required_with_precomputed_compressed_content_fits() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(
        name="huge",
        content="x" * 50_000,
        required=True,
        compressed_content="compressed summary",
    )
    compiled = pack.compile()
    by = {d.name: d for d in compiled.decisions}
    assert by["huge"].status == "compressed"
    assert by["huge"].tokens < by["huge"].original_tokens


def test_compressor_callback_used_when_needed() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(
        name="huge",
        content="y" * 30_000,
        priority=80,
        compressible=True,
    )
    calls = []

    def compress(item, target_tokens):
        calls.append((item.name, target_tokens))
        return f"summary of {item.name}"

    pack.set_compressor(compress)
    pack.add(name="task", content="task text", required=True)
    compiled = pack.compile()
    by = {d.name: d for d in compiled.decisions}
    assert calls, "compressor should have been invoked"
    assert by["huge"].status in ("compressed", "excluded")
    if by["huge"].status == "compressed":
        assert by["huge"].tokens < by["huge"].original_tokens


def test_compressor_not_called_when_item_fits() -> None:
    pack = _pack()
    pack.add(name="task", content="task text", required=True)
    pack.add(name="small", content="short text", compressible=True)
    calls = []
    pack.set_compressor(lambda it, n: calls.append(it.name) or "summary")
    pack.compile()
    assert not calls


def test_truncation_when_enabled_for_required() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(name="huge", content="x" * 50_000, required=True)
    cfg = CompilerConfig(allow_truncation=True)
    pack.set_config(cfg)
    compiled = pack.compile()
    by = {d.name: d for d in compiled.decisions}
    assert by["huge"].status == "truncated"
    assert by["huge"].tokens > 0


def test_compilation_is_deterministic() -> None:
    def build() -> ContextPack:
        p = _pack(budget=2000, reserved=500)
        p.add(name="task", content="t", required=True)
        p.add(name="a", content="alpha", priority=70)
        p.add(name="b", content="beta", priority=70)  # same priority — name breaks tie
        p.add(name="c", content="gamma", priority=70)
        return p

    r1 = build().compile()
    r2 = build().compile()
    order1 = [it.name for it in r1.included_items]
    order2 = [it.name for it in r2.included_items]
    assert order1 == order2


def test_duplicate_item_name_rejected_at_add_time() -> None:
    pack = _pack()
    pack.add(name="x", content="a")
    with pytest.raises(ValueError):
        pack.add(name="x", content="b")


def test_duplicate_names_in_compile_items_rejected() -> None:
    a = ContextItem(name="dup", content="1")
    b = ContextItem(name="dup", content="2")
    with pytest.raises(ValueError, match="Duplicate"):
        compile_items([a, b], model="claude-sonnet-4.6", token_budget=1000, reserved_output_tokens=100)


def test_included_order_stable_then_dynamic_then_ephemeral() -> None:
    pack = _pack(budget=10_000, reserved=500)
    pack.add(name="z_stable", content="a", cache_policy="stable", priority=10)
    pack.add(name="a_dynamic", content="b", cache_policy="dynamic", priority=10)
    pack.add(name="m_ephemeral", content="c", cache_policy="ephemeral", priority=10)
    compiled = pack.compile()
    names = [it.name for it in compiled.included_items]
    # stable comes first, then dynamic, then ephemeral
    assert names.index("z_stable") < names.index("a_dynamic")
    assert names.index("a_dynamic") < names.index("m_ephemeral")


def test_cacheable_prefix_only_counts_consecutive_stable_from_top() -> None:
    pack = _pack(budget=10_000, reserved=500)
    pack.add(name="sys", content="rules" * 50, kind="system", priority=100, cache_policy="stable", required=True)
    pack.add(name="doc", content="docs" * 50, kind="project_doc", priority=80, cache_policy="stable")
    pack.add(name="task", content="task", priority=95, cache_policy="dynamic", required=True)
    compiled = pack.compile()
    # cacheable prefix should equal sys + doc tokens
    sys_toks = compiled.included_tokens["sys"]
    doc_toks = compiled.included_tokens["doc"]
    assert compiled.cacheable_prefix_tokens == sys_toks + doc_toks


def test_decisions_have_one_row_per_input_item() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(name="task", content="t", required=True)
    pack.add(name="a", content="a")
    pack.add(name="b", content="b" * 20_000, priority=10)  # excluded
    compiled = pack.compile()
    names = {d.name for d in compiled.decisions}
    assert names == {"task", "a", "b"}


def test_health_score_in_range() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(name="task", content="task", required=True)
    pack.add(name="extra", content="x" * 100, priority=90)
    compiled = pack.compile()
    assert 0 <= compiled.health_score <= 100


def test_health_zero_when_nothing_included() -> None:
    # Edge case: only optional items, none fit
    pack = _pack(budget=1000, reserved=900)  # only 100 tokens available
    pack.add(name="big", content="x" * 5000, priority=10)
    compiled = pack.compile()
    assert compiled.used_tokens == 0
    assert compiled.health_score == 0


def test_health_penalized_when_high_priority_excluded() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(name="task", content="task", required=True)
    pack.add(name="big_hi", content="x" * 20_000, priority=95)  # high priority, excluded
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["big_hi"] == "excluded"
    assert compiled.health_score < 100


def test_prompt_property_concatenates_included() -> None:
    pack = _pack()
    pack.add(name="sys", content="SYSTEM", kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="TASK", required=True)
    compiled = pack.compile()
    assert "SYSTEM" in compiled.prompt
    assert "TASK" in compiled.prompt


def test_content_for_returns_compressed_if_compressed() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(
        name="huge",
        content="x" * 50_000,
        required=True,
        compressed_content="small",
    )
    compiled = pack.compile()
    assert compiled.content_for("huge") == "small"


def test_content_for_returns_original_when_not_compressed() -> None:
    pack = _pack()
    pack.add(name="x", content="hello", required=True)
    compiled = pack.compile()
    assert compiled.content_for("x") == "hello"


def test_content_for_missing_raises_keyerror() -> None:
    pack = _pack()
    pack.add(name="x", content="hello", required=True)
    compiled = pack.compile()
    with pytest.raises(KeyError):
        compiled.content_for("missing")


def test_pack_get_and_remove() -> None:
    pack = _pack()
    pack.add(name="x", content="hello")
    assert "x" in pack
    assert pack.get("x") is not None
    assert pack.remove("x") is True
    assert "x" not in pack
    assert pack.remove("x") is False


def test_compressor_returning_garbage_does_not_crash() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(name="huge", content="x" * 30_000, priority=80, compressible=True)
    pack.add(name="task", content="t", required=True)
    pack.set_compressor(lambda item, n: None)  # type: ignore[return-value,arg-type]
    compiled = pack.compile()
    by = {d.name: d.status for d in compiled.decisions}
    assert by["huge"] == "excluded"


def test_compressor_that_raises_does_not_crash() -> None:
    pack = _pack(budget=1500, reserved=500)
    pack.add(name="huge", content="x" * 30_000, priority=80, compressible=True)
    pack.add(name="task", content="t", required=True)

    def boom(item, n):
        raise RuntimeError("oops")

    pack.set_compressor(boom)
    compiled = pack.compile()
    by = {d.name: d.status for d in compiled.decisions}
    assert by["huge"] == "excluded"

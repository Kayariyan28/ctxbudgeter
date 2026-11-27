"""Rubric, diagnostics, planner, summarizer, cache cost model, analysis helpers."""

from __future__ import annotations

import pytest

from ctxbudgeter.runtime import (
    CompactionRubric,
    ContextEvent,
    ContextRuntime,
    ExtractiveSummarizer,
    PrefillCostModel,
)
from ctxbudgeter.runtime.analysis import retention, salient_terms
from ctxbudgeter.runtime.diagnostics import diagnose
from ctxbudgeter.runtime.planner import build_plan


def ev(tag: str, content: str, tokens: int = 100, kind: str = "tool_result", category: str = "working",
       **kw) -> ContextEvent:
    return ContextEvent(tag=tag, kind=kind, content=content, tokens=tokens, category=category, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("usage", "phase", "decision"),
    [
        (0.97, "mid_derivation", "compact_now"),  # hard limit beats phase
        (0.30, "subtask_complete", "block"),      # anti-anxiety floor
        (0.70, "mid_derivation", "defer"),
        (0.70, "stuck", "defer"),
        (0.90, None, "compact_now"),
        (0.70, "subtask_complete", "allow"),
        (0.70, "exploring", "defer"),
    ],
)
def test_rubric_decisions(usage: float, phase: str | None, decision: str) -> None:
    assert CompactionRubric().evaluate(usage, phase).decision == decision


def test_rubric_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        CompactionRubric(min_usage=0.9, trigger=0.5)


def test_salient_terms_and_retention() -> None:
    t = salient_terms("Module AUTH-3 sets ttl_s to 900 in src/auth/config.py; see https://x.io/a and `max_retries`.")
    assert {"AUTH-3", "ttl_s", "900", "src/auth/config.py", "https://x.io/a", "max_retries"} <= t
    assert retention({"AUTH-3", "900"}, "AUTH-3 uses 900") == 1.0
    assert retention({"AUTH-3", "900"}, "AUTH-3 changed") == 0.5
    assert retention(set(), "anything") == 1.0


def test_diagnose_finds_each_problem_kind() -> None:
    log = "\n".join(["2026-10-01 INFO fetch page=1 status=200"] * 60)
    events = [
        ev("1", "rules", kind="system", category="protected"),
        ev("2", "same text " * 20),
        ev("3", "value for X is 10", metadata={"key": "X"}),
        ev("4", log, tokens=400),
        ev("5", "huge " * 50, tokens=900),
        ev("6", "AKIAABCDEFGHIJKLMNOP is the key"),
        ev("7", "same text " * 20),
        ev("8", "value for X is 12", metadata={"key": "X"}),
    ]
    kinds = {d.kind: d for d in diagnose(events, available_tokens=4000, current_step=0)}
    assert kinds["duplicate"].tags == ["2"]
    assert kinds["conflict"].tags == ["3"]
    assert kinds["repetitive"].tags == ["4"]
    assert kinds["oversized"].tags == ["5"]
    assert kinds["sensitive"].severity == "critical"
    stale = diagnose([ev("1", "old output", step=0), ev("2", "new", step=20)], available_tokens=4000,
                     current_step=20)
    assert [d.kind for d in stale] == ["stale"]
    referenced = diagnose([ev("1", "old output", step=0), ev("2", "as shown in [e1]", step=20)],
                          available_tokens=4000, current_step=20)
    assert not [d for d in referenced if d.kind == "stale"]


def test_planner_respects_protection_and_reaches_target() -> None:
    events = [ev("1", "sys", 200, "system", "protected")] + [ev(str(i), f"doc {i}", 300) for i in range(2, 12)]
    events[4].pinned = True
    rubric = CompactionRubric().evaluate(0.9, None)
    plan = build_plan(events, available_tokens=3500, target_usage=0.5, diagnoses=[], timing=rubric,
                      protected_categories=("protected",), keep_recent=2)
    touched = set()
    for op in plan.ops:
        a, b = int(op.args["start"]), int(op.args["end"])
        touched |= {str(i) for i in range(a, b + 1)}
    assert "1" not in touched and events[4].tag not in touched and not touched & {"10", "11"}
    assert plan.projected_usage < plan.current_usage
    assert plan.timing is not None and plan.timing.decision == "compact_now"


def test_planner_is_cache_aware_and_avoids_resummarizing() -> None:
    events = ([ev("1", "sys", 100, "system", "protected"), ev("2..5", "summary", 300, "summary")]
              + [ev(str(i), f"doc {i}", 400) for i in range(6, 12)])
    events.insert(5, ev("x", "user said", 50, "user_message", "task_critical"))  # splits runs
    timing = CompactionRubric().evaluate(0.95, None)

    def plan_for(remaining: int):
        return build_plan(events, available_tokens=3000, target_usage=0.8, diagnoses=[], timing=timing,
                          protected_categories=("protected",), keep_recent=1,
                          expected_remaining_requests=remaining)

    long_horizon, short_horizon = plan_for(10), plan_for(1)
    for plan in (long_horizon, short_horizon):
        assert plan.ops and all(op.args.get("start") != "2..5" for op in plan.ops)
        assert any("invalidates the KV prefix" in n for n in plan.notes)
    # Many requests left: the bigger, earlier run amortizes its prefix recompute.
    assert long_horizon.ops[0].args["start"] == "6"
    # One request left: the later run (cheap to recompute after) wins.
    assert short_horizon.ops[0].args["start"] == "9"


def test_extractive_summarizer_keeps_facts_over_template_rows() -> None:
    rows = "\n".join(f"Line {i}: account {i % 7} adjustments {i * 3} ; offsets {i * 5}" for i in range(40))
    fact = "Ledger note: entity ALPHA-7 reported receipts for FY2019 of 3,461 million USD."
    events = [ev("3", rows + "\n" + fact, 900), ev("4", "Correction: ALPHA-7 revised to 3,512. This supersedes it.", 30)]
    out = ExtractiveSummarizer()(events, 80)
    assert "ALPHA-7" in out and "3,461" in out and "3,512" in out
    assert out.startswith("Summary of events 3..4")


def test_prefill_cost_model_prefix_vs_suffix() -> None:
    m = PrefillCostModel()
    a, b, c = ev("1", "A", 100), ev("2", "B", 200), ev("3", "C", 300)
    m.record([a, b, c], 0)
    s = ev("2..2", "S", 50)
    rc = m.record([a, s, c], 1)
    assert rc.prefix_reused == 100
    assert rc.prefill_prefix_cache == 350  # summary + unchanged C recomputed
    assert rc.prefill_suffix_reuse == 50   # only the summary is new
    t = m.totals()
    assert t["requests"] == 2 and t["prefill_prefix_cache"] == 600 + 350
    assert 0 < t["suffix_reuse_saving"] < 1
    assert PrefillCostModel.edit_recompute_cost([a, b, c], 1) == 500


def test_runtime_records_suffix_reuse_savings_after_compaction() -> None:
    rt = ContextRuntime(budget=4000)
    rt.add("sys", kind="system")
    for i in range(6):
        rt.add(f"doc {i} " + "word " * 120)
    rt.step()
    rt.policy = rt.governor.policy = rt.policy.with_overrides(min_usage_to_compact=0.0, compaction_trigger=0.0,
                                                               hard_limit=0.0)
    rt.compact("2", "3")
    rt.step()
    t = rt.cost_model.totals()
    assert t["prefill_suffix_reuse"] < t["prefill_prefix_cache"]

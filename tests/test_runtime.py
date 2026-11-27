"""ContextRuntime: the governed context API, autonomy levels, budget enforcement."""

from __future__ import annotations

import json

import pytest

from ctxbudgeter.governance import AutonomyLevel, ContextEditPolicy
from ctxbudgeter.runtime import ContextRuntime, JSONArchive, TraceLog


def doc(i: int, n: int = 12) -> str:
    return (f"Doc {i}: entity ALPHA-{i} reported receipts for FY2019 of {1000 + i * 37} million USD. "
            + "Filler prose about appropriations and outlays for the bureau. " * n)


def make_rt(budget: int = 2600, docs: int = 12, **kw) -> ContextRuntime:
    rt = ContextRuntime(budget=budget, reserved_output_tokens=200, **kw)
    rt.add("You are a research agent. Never fabricate numbers.", kind="system")
    rt.add("Report ALPHA-7 receipts for FY2019 in USD.", kind="task")
    for i in range(docs):
        rt.add(doc(i), kind="tool_result", source=f"docs/d{i}.txt")
        rt.step()
    return rt


def test_add_assigns_sequential_tags_and_default_categories() -> None:
    rt = ContextRuntime(budget=1000)
    a = rt.add("sys", kind="system")
    b = rt.add("hello", kind="user_message")
    c = rt.add("out", kind="tool_result")
    assert (a.tag, b.tag, c.tag) == ("1", "2", "3")
    assert (a.category, b.category, c.category) == ("protected", "task_critical", "working")
    with pytest.raises(ValueError):
        rt.add("x", category="nonsense")  # type: ignore[arg-type]


def test_budget_and_inspect_report_composition() -> None:
    rt = make_rt()
    b = rt.budget()
    assert b["available"] == 2400
    assert b["used"] == rt.used_tokens and 0 < b["usage"] == pytest.approx(rt.used_tokens / 2400, abs=1e-3)
    assert rt.stats["overflow_events_dropped"] == 0
    assert b["pressure"] in ("moderate", "high", "critical")
    ins = rt.inspect()
    assert ins["composition"]["by_category"]["protected"]["events"] == 2
    assert len(ins["events"]) == len(rt) == 14
    assert {"tag", "kind", "category", "tokens", "preview"} <= set(ins["events"][0])


def test_compact_archives_originals_and_recall_returns_them() -> None:
    rt = make_rt()
    before = rt.used_tokens
    r = rt.compact("3", "8")
    assert r.status == "applied", r.message
    assert r.details["summary_tag"] == "3..8"
    assert r.details["covers"] == ["3", "4", "5", "6", "7", "8"]
    assert rt.used_tokens < before
    assert "3" not in rt and "3..8" in rt
    summary = rt.get("3..8")
    assert summary is not None and "[ctx-provenance] covers events 3..8" in summary.content
    rec = rt.recall("5")
    assert rec.status == "applied" and rec.details["content"] == doc(2)
    assert rt.evaluate()["reopen_rate"] == 1.0  # the compaction was reopened


def test_nested_compaction_uses_outer_range_tag_and_expands() -> None:
    rt = make_rt(policy=ContextEditPolicy(min_usage_to_compact=0.0, compaction_trigger=0.0, hard_limit=0.0))
    assert rt.compact("3", "5").status == "applied"
    r = rt.compact("3..5", "7")
    assert r.status == "applied" and r.details["summary_tag"] == "3..7"
    rec = rt.recall("3..5")
    assert rec.details["originals"] == ["3", "4", "5"]


def test_protected_pinned_and_recent_context_are_denied() -> None:
    rt = make_rt()
    r = rt.archive("1")
    assert r.status == "denied" and r.decision["violations"][0]["rule"] == "protected_context"
    assert r.decision["violations"][0]["severity"] == "critical"
    assert rt.pin("5", "evidence").status == "applied"
    assert rt.compact("4", "6").status == "denied"
    last = rt.events[-1].tag
    r = rt.compact("9", last)
    assert any(v["rule"] == "recent_context" for v in r.decision["violations"])
    assert "1" in rt and "5" in rt


def test_premature_compaction_is_blocked_but_duplicates_can_be_archived() -> None:
    rt = ContextRuntime(budget=100_000)
    rt.add("sys", kind="system")
    for i in range(5):
        rt.add(doc(i), kind="tool_result")
    rt.add(doc(0), kind="tool_result")  # duplicate of tag 2
    rt.add("latest", kind="tool_result")
    rt.add("latest2", kind="tool_result")
    r = rt.compact("2", "4")
    assert r.status == "denied" and r.decision["violations"][0]["rule"] == "premature_compaction"
    dup = [d for d in rt.diagnose() if d.kind == "duplicate"]
    assert dup and dup[0].tags == ["2"]
    assert rt.archive(*dup[0].tags).status == "applied"


def test_pin_budget_enforced() -> None:
    rt = make_rt(policy=ContextEditPolicy(max_pinned_fraction=0.1))
    results = [rt.pin(t).status for t in ("3", "4", "5", "6")]
    assert "denied" in results


def test_restore_respects_hard_limit_and_reinserts_in_order() -> None:
    rt = make_rt(policy=ContextEditPolicy(hard_limit=0.3, compaction_trigger=0.0, min_usage_to_compact=0.0))
    assert rt.compact("3", "6").status == "applied"
    r = rt.recall("4", restore=True)  # usage is already above the 30% hard limit
    assert r.status == "denied" and r.decision["violations"][0]["rule"] == "restore_over_limit"
    assert rt.recall("4").status == "applied"  # plain recall is always allowed
    rt2 = make_rt(budget=6000)
    rt2.policy = rt2.governor.policy = ContextEditPolicy(min_usage_to_compact=0.0, compaction_trigger=0.0,
                                                          hard_limit=1.0)
    rt2.compact("3", "6")
    r2 = rt2.recall("4", restore=True)
    assert r2.status == "applied" and r2.details["restored"]
    tags = [e.tag for e in rt2.events]
    assert tags.index("3..6") < tags.index("4") < tags.index("7")


def test_checkpoint_rollback_and_auto_checkpoint() -> None:
    rt = make_rt()
    snap = [e.tag for e in rt.events]
    assert rt.compact("3", "8").status == "applied"
    assert rt.rollback().status == "applied"  # auto checkpoint taken before the compaction
    assert [e.tag for e in rt.events] == snap
    cp = rt.checkpoint("manual").details["checkpoint_id"]
    rt.archive("9")
    rt.rollback(cp)
    assert "9" in rt
    assert rt.rollback("nope").status == "not_found"


def test_autonomy_observe_and_recommend_never_mutate() -> None:
    for level, status in ((AutonomyLevel.OBSERVE, "denied"), (AutonomyLevel.RECOMMEND, "recommended")):
        rt = make_rt(autonomy=level)
        before = [e.tag for e in rt.events]
        r = rt.compact("3", "8")
        assert r.status == status
        assert [e.tag for e in rt.events] == before
        assert rt.recall("3").status == "noop"  # read-only calls still work


def test_supervised_queue_approve_revalidates_and_reject() -> None:
    rt = make_rt(autonomy="L2")
    r = rt.compact("3", "8")
    assert r.status == "pending" and r.proposal_id
    assert len(rt.pending) == 1 and "3" in rt
    ok = rt.approve(r.proposal_id)
    assert ok.status == "applied" and "3..8" in rt
    r2 = rt.archive("9")
    assert rt.reject(r2.proposal_id, "no").status == "denied"
    assert rt.approve("missing").status == "not_found"


def test_supervised_with_approver_callback() -> None:
    seen = []
    rt = make_rt(autonomy=AutonomyLevel.SUPERVISED, approver=lambda p: seen.append(p.op) or p.op != "archive")
    assert rt.compact("3", "8").status == "applied"
    assert rt.archive("9").status == "denied"
    assert seen == ["compact", "archive"]


def test_step_overflow_drops_oldest_unprotected_events() -> None:
    rt = ContextRuntime(budget=600)
    rt.add("system rules", kind="system")
    for i in range(8):
        rt.add(doc(i, 8), kind="tool_result")
    rep = rt.step()
    assert rep.overflow_dropped and rep.overflow_dropped[0] == "2"
    assert "1" in rt and rt.used_tokens <= rt.available_tokens
    assert rt.stats["overflow_events_dropped"] == len(rep.overflow_dropped)
    assert rt.archive_store.get("2") is None  # dropped, not archived — irrecoverable


def test_fallback_compaction_keeps_facts_recoverable() -> None:
    rt = ContextRuntime(budget=900, policy=ContextEditPolicy(fallback_compaction=True))
    rt.add("system rules", kind="system")
    for i in range(12):
        rt.add(doc(i, 6), kind="tool_result")
        rep = rt.step()
    assert rt.stats["fallback_compactions"] >= 1
    assert rt.stats["overflow_events_dropped"] == 0
    assert rt.archive_store.get("2") is not None
    assert not rep.failed


def test_pressure_warning_appears_above_warn_threshold() -> None:
    rt = make_rt()
    w = rt.pressure_warning()
    assert (w is not None) == (rt.usage >= rt.policy.warn_at)
    if w:
        assert "context_compact" in w and "not a reason to stop" in w


def test_messages_rendering_openai_and_anthropic() -> None:
    rt = make_rt(docs=2)
    rt.add("thinking", kind="assistant_message")
    oa = rt.to_messages("openai")
    assert oa[0]["role"] == "system" and "[e3 tool_result" in oa[1]["content"]
    assert oa[-1]["role"] == "assistant"
    an = rt.to_messages("anthropic")
    assert "research agent" in an["system"] and an["messages"][0]["role"] == "user"
    with pytest.raises(ValueError):
        rt.to_messages("cohere")


def test_bridge_to_context_pack_compiles_with_bom() -> None:
    rt = make_rt(docs=3)
    pack = rt.to_context_pack()
    compiled = pack.compile()
    names = {it.name for it in compiled.included_items}
    assert {"e1", "e2"} <= names
    assert {i["name"] for i in compiled.bom.to_dict()["included_items"]} >= {"e1", "e2"}


def test_round_trip_to_dict_from_dict(tmp_path) -> None:
    arch = JSONArchive(tmp_path / "a.json")
    rt = make_rt(archive=arch)
    rt.compact("3", "8")
    data = json.loads(json.dumps(rt.to_dict()))
    rt2 = ContextRuntime.from_dict(data, archive=JSONArchive(tmp_path / "a.json"))
    assert [e.tag for e in rt2.events] == [e.tag for e in rt.events]
    assert rt2.recall("4").details["content"] == doc(1)
    assert rt2.add("new").tag == str(rt._next_tag)


def test_trace_log_persists_and_evaluate_single_decision(tmp_path) -> None:
    log = TraceLog(tmp_path / "t.jsonl")
    rt = make_rt(trace=log)
    r = rt.compact("3", "8")
    ev = rt.evaluate(r.trace_id)
    assert ev["op"] == "compact" and ev["compression_factor"] > 1
    assert 0 <= ev["fact_retention"] <= 1
    end = rt.end_session(correct=True)
    assert end["correct"] is True
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["op"] == "session_end"
    with pytest.raises(KeyError):
        rt.evaluate("nope")


def test_secret_in_summary_is_redacted_not_leaked() -> None:
    rt = make_rt()
    key = "sk-proj-ABCDEFGHIJKLMNOPQRSTUV1234"
    r = rt.compact("3", "8", summary=f"Summary. The key is {key}. ALPHA-3 1111.")
    assert r.status == "applied" and r.decision["verdict"] == "modify"
    assert key not in rt.render_text()


def test_unrestricted_policy_commits_violations() -> None:
    rt = make_rt(policy=ContextEditPolicy.unrestricted())
    r = rt.archive("1")
    assert r.status == "applied"
    assert rt.evaluate()["critical_violations"] == 1


def test_recall_hints_find_archived_facts_even_when_question_is_active() -> None:
    rt = make_rt()
    rt.compact("3", "12", summary="Read documents; details archived.")
    q = "What were ALPHA-7 receipts for FY2019?"
    rt.add(q, kind="user_message")
    hints = rt.recall_hints(q)
    assert hints and hints[0]["tag"] == "10"  # Doc 7 is tag 10

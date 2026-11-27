"""CTXBench: tasks, reader, scoring, runner modes, agents (scripted and LLM)."""

from __future__ import annotations

import json

import pytest

from ctxbudgeter.bench import (
    FAMILIES,
    BenchRunner,
    LLMContextAgent,
    TaskGenerator,
    answer_correct,
    context_intelligence_score,
    make_agent,
    read_facts,
    run_episode,
    to_anthropic_conversation,
    value_present,
)
from ctxbudgeter.runtime import TraceLog


@pytest.fixture(scope="module")
def tasks():
    return TaskGenerator(seed=11).suite(["research", "coding"], n=1, stream_tokens=6000)


def test_generator_is_deterministic_and_plants_facts() -> None:
    for fam in FAMILIES:
        a = TaskGenerator(seed=5).generate(fam, 1, stream_tokens=4000)[0]
        b = TaskGenerator(seed=5).generate(fam, 1, stream_tokens=4000)[0]
        assert [e.content for e in a.events] == [e.content for e in b.events]
        stream = "\n".join(e.content for e in a.events)
        for k, v in a.expected.items():
            assert value_present(stream, k, v), (fam, k)
        assert a.events[0].category == "protected" and a.events[1].category == "protected"
        assert len(a.superseded) == 1
    assert TaskGenerator(seed=6).generate("research", 1)[0].question != \
        TaskGenerator(seed=5).generate("research", 1)[0].question
    with pytest.raises(ValueError):
        TaskGenerator().generate("poetry")


def test_reader_prefers_explicit_corrections() -> None:
    old = "Ledger note: entity ALPHA-7 reported receipts for FY2019 of 3,461 million USD."
    new = "Correction: ledger note: entity ALPHA-7 reported receipts for FY2019 of 3,512 million USD. " \
          "This supersedes the earlier figure."
    assert read_facts(old + "\n" + new)["ALPHA-7|receipts|FY2019"] == "3512"
    assert read_facts(new + "\n" + old)["ALPHA-7|receipts|FY2019"] == "3512"  # order-independent
    assert answer_correct("It is 3,512 million.", {"k": "3512"}, {"k": "3461"}) == (True, False)
    assert answer_correct("3461 then 3512", {"k": "3512"}, {"k": "3461"}) == (False, True)


def test_cis_formula_override_and_renormalization() -> None:
    cis, raw, grade = context_intelligence_score(accuracy=0.9, retention=0.85, efficiency=0.75, recovery=0.8)
    assert cis == raw == pytest.approx(84.25) and grade == "competent"
    cis, raw, grade = context_intelligence_score(accuracy=1, retention=1, efficiency=1, recovery=1,
                                                 critical_violations=1)
    assert cis == 0 and raw == 100 and grade == "fail"
    cis, _, _ = context_intelligence_score(accuracy=1, retention=1, efficiency=0.5, recovery=None)
    assert cis == pytest.approx(100 * (0.4 + 0.25 + 0.1) / 0.85, abs=0.01)
    with pytest.raises(ValueError):
        context_intelligence_score(accuracy=2, retention=1, efficiency=1, recovery=1)
    with pytest.raises(ValueError):
        context_intelligence_score(accuracy=1, retention=1, efficiency=1, recovery=1, weights={"x": 1})


def test_harness_mode_never_calls_the_agent(tasks) -> None:
    class Boom:
        name = "boom"

        def act(self, rt, obs):
            raise AssertionError("agent called in harness mode")

        def prepare_answer(self, rt, q, obs):
            raise AssertionError("agent called in harness mode")

    r = run_episode(tasks[0], Boom(), mode="harness", budget=2000)
    assert r.mode == "harness" and r.recovery is None


def test_competent_agent_beats_harness_baseline(tasks) -> None:
    rep = BenchRunner(tasks, lambda: make_agent("competent"), modes=["harness", "guided"], budgets=[2000]).run()
    s = rep.summary
    assert s["guided"]["retention"] >= s["harness"]["retention"]
    assert s["guided"]["cis"] > s["harness"]["cis"]
    assert s["guided"]["critical_violations"] == 0


def test_governance_eliminates_reckless_violations(tasks) -> None:
    rep = BenchRunner(tasks, lambda: make_agent("reckless"), modes=["unrestricted", "adaptive"], budgets=[2000],
                      optimize=False).run()
    s = rep.summary
    assert s["unrestricted"]["critical_violations"] > 0 and s["unrestricted"]["cis"] == 0
    assert s["adaptive"]["critical_violations"] == 0 and s["adaptive"]["cis"] > 0
    assert "unsafe" in rep.adaptive_policy["behaviors"]


def test_adaptive_rescues_avoidant_agent(tasks) -> None:
    rep = BenchRunner(tasks, lambda: make_agent("avoidant"), modes=["unrestricted", "adaptive"],
                      budgets=[2000]).run()
    s = rep.summary
    assert s["adaptive"]["retention"] > s["unrestricted"]["retention"]
    assert s["adaptive"]["overflow_events_dropped"] < s["unrestricted"]["overflow_events_dropped"]
    assert rep.adaptive_policy["status"] in ("validated", "candidate")
    assert "avoidant" in rep.adaptive_policy["behaviors"]
    md = rep.to_markdown()
    assert "| adaptive |" in md and "## Adaptive policy" in md
    data = json.loads(rep.to_json())
    assert data["episodes"] and data["by_mode_budget"]
    ev = rep.evidence("adaptive")
    assert ev.episodes == len(tasks)


def test_runner_writes_traces_and_rejects_unknown_modes(tasks, tmp_path) -> None:
    log = TraceLog(tmp_path / "t.jsonl")
    BenchRunner(tasks[:1], lambda: make_agent("competent"), modes=["guided"], budgets=[2000], trace=log).run()
    assert any(t.op == "session_end" for t in log)
    assert (tmp_path / "t.jsonl").read_text().strip()
    with pytest.raises(ValueError):
        BenchRunner(tasks, lambda: make_agent("competent"), modes=["chaos"])
    with pytest.raises(ValueError):
        make_agent("genius")


def test_answer_fn_path(tasks) -> None:
    t = tasks[0]
    right = " ".join(t.expected.values())
    r = run_episode(t, make_agent("competent"), mode="guided", budget=50_000, answer_fn=lambda ctx, q: right)
    assert r.correct
    r2 = run_episode(t, make_agent("competent"), mode="guided", budget=50_000, answer_fn=lambda ctx, q: "unsure")
    assert not r2.correct


def test_llm_agent_drives_context_tools_with_fake_model(tasks) -> None:
    """A scripted 'model' that compacts under pressure and recalls before answering."""
    state = {"compacted": False}

    def complete(messages, tools):
        assert any(t["function"]["name"] == "context_compact" for t in tools)
        last = messages[-1]
        if last["role"] == "tool":
            return []  # one tool call per turn
        prompt = last["content"]
        if "Context maintenance turn" in prompt:
            data = json.loads(prompt.splitlines()[-1])
            evs = [e for e in data["events"] if e["category"] not in ("protected",) and not e["pinned"]][:-3]
            if len(evs) >= 2:
                state["compacted"] = True
                return [{"id": "c1", "name": "context_compact",
                         "arguments": {"start": evs[0]["tag"], "end": evs[-1]["tag"]}}]
            return []
        if "about to answer" in prompt:
            return [{"id": "r1", "name": "context_search_archive", "arguments": {"query": "x"}}]
        return []

    agent = LLMContextAgent(complete, act_from_usage=0.6)
    r = run_episode(tasks[0], agent, mode="guided", budget=2000)
    assert state["compacted"] and agent.calls_made >= 2 and r.compactions >= 1
    assert not agent.errors


def test_anthropic_conversation_conversion() -> None:
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "context_budget", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "{\"usage\": 0.5}"}]
    system, conv = to_anthropic_conversation(msgs)
    assert system == "S"
    assert [m["role"] for m in conv] == ["user", "assistant", "user"]
    assert conv[1]["content"][0] == {"type": "tool_use", "id": "t1", "name": "context_budget", "input": {}}
    assert conv[2]["content"][0]["type"] == "tool_result" and conv[2]["content"][0]["tool_use_id"] == "t1"

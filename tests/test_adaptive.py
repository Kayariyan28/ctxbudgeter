"""Behavioural profiling, priors, tuning, versioned policy store, optimization."""

from __future__ import annotations

import pytest

from ctxbudgeter.adaptive import (
    PAPER_PRIORS,
    AdaptivePolicy,
    BehaviorProfiler,
    ModelProfile,
    PolicyStore,
    PolicyTuner,
    classify,
    prior_for,
    resolve_profile,
)
from ctxbudgeter.runtime import DecisionTrace


def tr(sid: str, op: str, usage: float, *, verdict: str = "applied", before: int = 400, after: int = 100,
       model: str = "m", outcome: dict | None = None, violations: list | None = None, step: int = 1) -> DecisionTrace:
    return DecisionTrace(trace_id=f"{sid}-{op}-{step}-{usage}", session_id=sid, agent_id="a", model=model, step=step,
                         op=op, args={}, autonomy=3, verdict=verdict, usage_before=usage, tokens_before=before,
                         tokens_after=after, outcome=outcome or {"actor": "agent"}, violations=violations or [])


def test_profiler_computes_paper_metrics() -> None:
    traces = [
        tr("s1", "compact", 0.25, outcome={"actor": "agent", "reopened": True,
                                           "decision_notes": {"fact_retention": 0.4}}),
        tr("s2", "compact", 0.35, before=800, after=100, outcome={"actor": "agent",
                                                                  "decision_notes": {"fact_retention": 0.5}}),
        tr("s3", "session_end", 0.9, verdict="observed",
           outcome={"actor": "harness", "overflow_events_dropped": 3, "correct": False, "retention": 0.5}),
    ]
    p = BehaviorProfiler().profile(traces)["m"]
    assert p.sessions == 3
    assert p.compaction_session_rate == pytest.approx(2 / 3, abs=1e-3)
    assert p.median_first_compaction_usage == pytest.approx(0.30)
    assert p.median_compression_factor == pytest.approx(6.0)
    assert p.reopen_rate == 0.5 and p.early_compaction_rate == 1.0
    assert p.outcome_retention == 0.5 and p.accuracy == 0.0
    assert "anxious" in p.behaviors and "lossy" in p.behaviors


def test_classify_each_behavior() -> None:
    assert "avoidant" in classify(ModelProfile("m", compaction_session_rate=0.05, warning_answer_rate=0.9))
    assert "over_compressing" in classify(ModelProfile("m", median_compression_factor=6, reopen_rate=0.4))
    assert "under_recovering" in classify(ModelProfile("m", recovery_rate=0.2))
    assert "stale_keeping" in classify(ModelProfile("m", stale_answer_rate=0.5))
    assert "unsafe" in classify(ModelProfile("m", critical_violation_rate=0.5))
    assert classify(ModelProfile("m", compaction_session_rate=0.5, median_first_compaction_usage=0.85)) == []


def test_paper_priors_reproduce_reported_behaviours() -> None:
    assert PAPER_PRIORS["gpt-6.1-sol"].behaviors == []
    assert "anxious" in PAPER_PRIORS["kimi-k3"].behaviors
    assert "anxious" in PAPER_PRIORS["deepseek-v4-pro"].behaviors
    assert {"avoidant", "anxious"} <= set(PAPER_PRIORS["claude-opus-5.5"].behaviors)
    assert prior_for("Claude-Opus-5-5") is not None and prior_for("unknown-model") is None


def test_resolve_profile_prefers_observed_data() -> None:
    assert resolve_profile("kimi-k3").source == "paper_prior"
    traces = [tr(f"s{i}", "compact", 0.9, model="kimi-k3") for i in range(3)]
    assert resolve_profile("kimi-k3", traces).source == "observed"
    assert resolve_profile("nobody").model == "nobody"


@pytest.mark.parametrize(
    ("behavior", "check"),
    [
        ("avoidant", lambda a: a.policy.fallback_compaction and "compaction_expected" in a.guidance),
        ("anxious", lambda a: a.policy.min_usage_to_compact >= 0.7 and "timing_instruction" in a.guidance),
        ("over_compressing", lambda a: a.policy.max_compression_ratio == 4.0 and "preserve_detail" in a.guidance),
        ("lossy", lambda a: a.policy.min_fact_retention >= 0.6 and "pin_before_compact" in a.guidance),
        ("under_recovering", lambda a: "use_archive_search" in a.guidance),
        ("stale_keeping", lambda a: "resolve_conflicts" in a.guidance),
        ("unsafe", lambda a: a.autonomy_cap == 2 and a.policy.keep_recent_events >= 4),
    ],
)
def test_tuner_rules(behavior: str, check) -> None:
    a = PolicyTuner().tune(ModelProfile("m", behaviors=[behavior]))
    assert check(a), a.rules_applied
    assert a.status == "candidate" and a.version == 1
    if behavior not in ("under_recovering", "stale_keeping"):
        assert a.policy.fallback_compaction  # tightening always ships the safe path
    a.render_guidance(autonomy=4)  # guidance keys are all valid


def test_tuner_balanced_profile_keeps_base_policy() -> None:
    a = PolicyTuner().tune(ModelProfile("m"))
    assert a.rules_applied == ["no failure behaviours detected → base policy retained"]
    assert not a.policy.fallback_compaction


def test_policy_round_trip_and_transfer() -> None:
    a = PolicyTuner().tune(ModelProfile("kimi-k3", behaviors=["anxious"]))
    b = AdaptivePolicy.from_dict(a.to_dict())
    assert b.policy_id == a.policy_id and b.policy == a.policy
    t = a.transfer_to("deepseek-v4-pro")
    assert t.status == "transfer_candidate" and t.parent_id == a.policy_id and t.model == "deepseek-v4-pro"


def test_policy_store_versions_and_activation(tmp_path) -> None:
    store = PolicyStore(tmp_path)
    tuner = PolicyTuner()
    p1 = store.save(tuner.tune(ModelProfile("m", behaviors=["anxious"])))
    p2 = store.save(tuner.tune(ModelProfile("m", behaviors=["lossy"]), parent=p1))
    assert [p.version for p in store.history("m")] == [1, 2]
    assert store.active("m") is None and store.latest("m").policy_id == p2.policy_id
    store.record_validation(p1, {"passed": True, "delta_cis": 3.0})
    assert store.active("m").policy_id == p1.policy_id
    store.record_validation(p2, {"passed": False})
    assert store.get("m", p2.policy_id).status == "rejected"


def test_optimizer_selects_and_validates() -> None:
    from ctxbudgeter.adaptive.optimizer import PolicyOptimizer, validate_policy
    from ctxbudgeter.bench import TaskGenerator, make_agent

    tasks = TaskGenerator(seed=3).generate("research", 1, stream_tokens=5000)
    prof = ModelProfile("bench-model", behaviors=["avoidant", "lossy"])
    opt = PolicyOptimizer(lambda: make_agent("avoidant"), tasks, budgets=[1800], model="bench-model")
    names = [n for n, _ in opt.candidates(prof)]
    assert names[0] == "tuned" and "base" in names and "tuned_without_lossy" in names
    res = opt.optimize(prof)
    assert res.best.evidence["optimization"]["selected"] in names
    assert res.leaderboard[0]["cis"] >= res.leaderboard[-1]["cis"]
    val = validate_policy(res.best, lambda: make_agent("avoidant"), tasks, budgets=[1800], model="bench-model")
    assert {"passed", "delta_cis", "candidate", "baseline"} <= set(val)

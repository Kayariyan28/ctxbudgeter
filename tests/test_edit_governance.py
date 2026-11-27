"""EditGovernor rules, ContextEditPolicy, graduated autonomy and certification."""

from __future__ import annotations

import pytest

from ctxbudgeter.governance import (
    PROVENANCE_PREFIX,
    AutonomyCertifier,
    AutonomyLevel,
    AutonomyRegistry,
    ContextEditPolicy,
    EditGovernor,
    EditProposal,
    Evidence,
    enforce_policy,  # backward-compat import path still works
)
from ctxbudgeter.runtime import ContextEvent


def e(tag: str, content: str = "x", tokens: int = 100, category: str = "working", **kw) -> ContextEvent:
    return ContextEvent(tag=tag, kind="tool_result", content=content, tokens=tokens, category=category, **kw)  # type: ignore[arg-type]


def rules(decision) -> set[str]:
    return {v.rule for v in decision.violations}


def test_backcompat_symbols() -> None:
    assert callable(enforce_policy)


def test_compact_happy_path_adds_provenance_footer() -> None:
    gov = EditGovernor()
    p = EditProposal("compact", [e("3", "ALPHA-7 is 3461", source="a.md"), e("4")], usage=0.9,
                     summary="ALPHA-7 is 3461", summary_tokens=10)
    d = gov.review(p)
    assert d.verdict == "modify" and d.approved
    assert PROVENANCE_PREFIX in (d.summary or "") and "sources: a.md" in (d.summary or "")
    assert d.notes["fact_retention"] == 1.0


@pytest.mark.parametrize(
    ("policy", "proposal", "rule"),
    [
        (ContextEditPolicy(), EditProposal("compact", [e("1", category="protected")], usage=0.9, summary="s",
                                           summary_tokens=5), "protected_context"),
        (ContextEditPolicy(), EditProposal("compact", [e("3", pinned=True)], usage=0.9, summary="s",
                                           summary_tokens=5), "pinned_context"),
        (ContextEditPolicy(), EditProposal("compact", [e("9")], usage=0.9, summary="s", summary_tokens=5,
                                           recent_tags=("9",)), "recent_context"),
        (ContextEditPolicy(), EditProposal("compact", [e("3")], usage=0.2, summary="s", summary_tokens=5),
         "premature_compaction"),
        (ContextEditPolicy(max_compression_ratio=4), EditProposal("compact", [e("3", tokens=1000)], usage=0.9,
                                                                  summary="s", summary_tokens=10), "over_compression"),
        (ContextEditPolicy(min_fact_retention=0.6),
         EditProposal("compact", [e("3", "ALPHA-7 3461 FY2019", category="task_critical")], usage=0.9,
                      summary="something", summary_tokens=3), "critical_fact_loss"),
        (ContextEditPolicy(min_summary_retention=0.5),
         EditProposal("compact", [e("3", "ALPHA-7 3461 FY2019 BRAVO-2 77")], usage=0.9, summary="ALPHA-7",
                      summary_tokens=3), "summary_detail_loss"),
        (ContextEditPolicy(max_compaction_span_tokens=150),
         EditProposal("compact", [e("3"), e("4")], usage=0.9, summary="s", summary_tokens=5), "compaction_span"),
        (ContextEditPolicy(), EditProposal("archive", [e("3", category="task_critical")], usage=0.2),
         "premature_archive"),
        (ContextEditPolicy(max_pinned_fraction=0.1), EditProposal("pin", [e("3", tokens=200)], available_tokens=1000),
         "pin_budget"),
        (ContextEditPolicy(), EditProposal("restore", [e("3")], used_tokens=950, restore_tokens=100,
                                           available_tokens=1000), "restore_over_limit"),
        (ContextEditPolicy(max_mutations_per_step=2), EditProposal("archive", [e("3")], usage=0.9,
                                                                   mutations_this_step=2), "mutation_rate"),
    ],
)
def test_governor_denies(policy: ContextEditPolicy, proposal: EditProposal, rule: str) -> None:
    d = EditGovernor(policy).review(proposal)
    assert d.verdict == "deny" and rule in rules(d)


def test_unrestricted_detects_but_allows() -> None:
    d = EditGovernor(ContextEditPolicy.unrestricted()).review(
        EditProposal("archive", [e("1", category="protected")], usage=0.9))
    assert d.approved and d.critical and "protected_context" in rules(d)


def test_secret_in_summary_redacted_or_blocked() -> None:
    secret = "ghp_" + "A" * 30
    p = EditProposal("compact", [e("3")], usage=0.9, summary=f"token {secret}", summary_tokens=10)
    d = EditGovernor().review(p)
    assert d.verdict == "modify" and secret not in (d.summary or "")
    d2 = EditGovernor(ContextEditPolicy(redact_summary_secrets=False)).review(p)
    assert d2.verdict == "deny" and "secret_in_summary" in rules(d2)


def test_unknown_op_and_policy_validation() -> None:
    assert EditGovernor().review(EditProposal("explode")).verdict == "deny"
    with pytest.raises(ValueError):
        ContextEditPolicy(min_usage_to_compact=0.9, compaction_trigger=0.5)
    with pytest.raises(ValueError):
        ContextEditPolicy.from_dict({"nope": 1})
    with pytest.raises(ValueError):
        ContextEditPolicy().with_overrides(nope=1)
    pol = ContextEditPolicy.strict()
    assert ContextEditPolicy.from_dict(pol.to_dict()) == pol


def test_autonomy_parse() -> None:
    assert AutonomyLevel.parse("L3") is AutonomyLevel.GOVERNED
    assert AutonomyLevel.parse("adaptive") is AutonomyLevel.ADAPTIVE
    assert AutonomyLevel.parse(2) is AutonomyLevel.SUPERVISED


def test_certifier_promotes_holds_and_demotes() -> None:
    c = AutonomyCertifier()
    good = Evidence(episodes=12, mean_cis=70, critical_violations=0, mean_retention=0.9)
    assert c.evaluate(AutonomyLevel.RECOMMEND, good).action == "promote"
    assert c.evaluate(AutonomyLevel.SUPERVISED, good).action == "hold"  # needs 20 eps / CIS 75 for L3
    not_held_out = Evidence(episodes=50, mean_cis=99, critical_violations=0, mean_retention=1, held_out=False)
    assert c.evaluate(AutonomyLevel.RECOMMEND, not_held_out).action == "hold"
    bad = Evidence(episodes=30, mean_cis=95, critical_violations=1, mean_retention=1)
    r = c.evaluate(AutonomyLevel.ADAPTIVE, bad)
    assert r.action == "demote" and r.new == AutonomyLevel.SUPERVISED
    slump = Evidence(episodes=25, mean_cis=60, critical_violations=0, mean_retention=0.9)
    assert c.evaluate(AutonomyLevel.GOVERNED, slump).new == AutonomyLevel.SUPERVISED


def test_registry_persists_history(tmp_path) -> None:
    path = tmp_path / "aut.json"
    reg = AutonomyRegistry(path)
    assert reg.level("a", "m") == AutonomyLevel.RECOMMEND
    reg.certify("a", "m", Evidence(episodes=12, mean_cis=70, critical_violations=0, mean_retention=0.9))
    reg2 = AutonomyRegistry(path)
    assert reg2.level("a", "m") == AutonomyLevel.SUPERVISED
    assert reg2.history("a", "m")[0]["action"] == "promote"
    reg2.set_level("a", "m", AutonomyLevel.OBSERVE, "incident")
    assert AutonomyRegistry(path).level("a", "m") == AutonomyLevel.OBSERVE

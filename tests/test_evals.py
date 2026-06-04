"""Tests for ContextEval / EvalSuite."""

from __future__ import annotations

import json

from ctxbudgeter import ContextEval, ContextPack, ContextPolicy, EvalSuite


def _compiled():
    p = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    p.add(name="refund_policy.md", content="Refunds within 30 days.", kind="project_doc",
          required=True, source="docs/refund_policy.md")
    p.add(name="task", content="answer refund question", kind="task", required=True)
    return p.compile()


def test_required_source_present_passes() -> None:
    res = ContextEval(name="t", required_sources=["refund_policy"]).run(_compiled())
    assert res.passed


def test_required_source_missing_fails() -> None:
    res = ContextEval(name="t", required_sources=["nonexistent_doc"]).run(_compiled())
    assert not res.passed
    assert any(not c.passed for c in res.checks)


def test_forbidden_source_present_fails() -> None:
    p = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    p.add(name="payroll.csv", content="salaries", kind="data", required=True, source="hr/payroll.csv")
    p.add(name="task", content="x", kind="task", required=True)
    res = ContextEval(name="t", forbidden_sources=["payroll.csv"]).run(p.compile())
    assert not res.passed


def test_max_tokens_enforced() -> None:
    res = ContextEval(name="t", max_tokens=1).run(_compiled())
    assert not res.passed


def test_block_secrets_passes_when_redacted() -> None:
    policy = ContextPolicy(max_tokens=20000, reserved_output_tokens=2000,
                           block_secrets=True, redact_sensitive=True)
    p = ContextPack(policy=policy)
    p.add(name="task", content="do", kind="task", required=True)
    p.add(name="leak", content="key sk-ABCDEF1234567890GHIJKL", kind="data", priority=50, source="t.txt")
    compiled = p.compile()
    res = ContextEval(name="t", block_secrets=True).run(compiled)
    assert res.passed, "redacted secrets should satisfy block_secrets"


def test_min_health_score() -> None:
    res = ContextEval(name="t", min_context_health_score=200).run(_compiled())
    assert not res.passed


def test_suite_from_yaml(tmp_path) -> None:
    import pytest

    pytest.importorskip("yaml")
    p = tmp_path / "evals.yaml"
    p.write_text(
        "evals:\n  - name: refund_eval\n    required_sources: [refund_policy]\n    max_tokens: 16000\n"
    )
    suite = EvalSuite.from_file(str(p))
    assert suite.names == ["refund_eval"]
    results = suite.run(_compiled())
    assert results[0].passed


def test_suite_from_json(tmp_path) -> None:
    p = tmp_path / "evals.json"
    p.write_text(json.dumps({"evals": [{"name": "e1", "required_sources": ["refund_policy"]}]}))
    suite = EvalSuite.from_file(str(p))
    assert suite.run(_compiled())[0].passed


def test_eval_runs_on_bom() -> None:
    bom = _compiled().bom
    res = ContextEval(name="t", required_sources=["refund_policy"]).run(bom)
    assert res.passed

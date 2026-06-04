"""Tests for ContextPolicy + governance enforcement."""

from __future__ import annotations

import json

import pytest

from ctxbudgeter import ContextPack, ContextPolicy
from ctxbudgeter.governance import PolicyViolationError


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        ContextPolicy(max_tokens=0)
    with pytest.raises(ValueError):
        ContextPolicy(max_tokens=100, reserved_output_tokens=-1)
    with pytest.raises(ValueError):
        ContextPolicy(max_tokens=100, reserved_output_tokens=100)
    with pytest.raises(ValueError):
        ContextPolicy(max_tokens=100, reserved_output_tokens=10, max_item_tokens=0)


def test_policy_available_tokens() -> None:
    p = ContextPolicy(max_tokens=24000, reserved_output_tokens=4000)
    assert p.available_context_tokens == 20000


def test_policy_source_matching() -> None:
    p = ContextPolicy(allowed_sources=["docs", "repo"], forbidden_sources=[".env", "payroll"])
    assert p.is_source_forbidden("config/.env")
    assert not p.is_source_forbidden("docs/readme.md")
    assert p.is_source_allowed("docs/readme.md")
    assert not p.is_source_allowed("secrets/x.txt")
    # No allow-list => everything allowed
    assert ContextPolicy().is_source_allowed("anything")


def test_policy_from_dict_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown policy keys"):
        ContextPolicy.from_dict({"max_tokens": 100, "reserved_output_tokens": 10, "bogus": 1})


def test_policy_from_yaml(tmp_path) -> None:
    p = tmp_path / "pack.yaml"
    p.write_text(
        "max_tokens: 20000\nreserved_output_tokens: 2000\nblock_secrets: true\n"
        "forbidden_sources:\n  - .env\n"
    )
    pytest.importorskip("yaml")
    pol = ContextPolicy.from_yaml(str(p))
    assert pol.max_tokens == 20000
    assert pol.forbidden_sources == [".env"]


def test_policy_from_json_no_yaml_needed(tmp_path) -> None:
    p = tmp_path / "pack.json"
    p.write_text(json.dumps({"max_tokens": 15000, "reserved_output_tokens": 3000}))
    pol = ContextPolicy.from_yaml(str(p))  # .json branch needs no pyyaml
    assert pol.max_tokens == 15000


def test_policy_nested_layout(tmp_path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"policy": {"max_tokens": 12000, "reserved_output_tokens": 2000}}))
    pol = ContextPolicy.from_yaml(str(p))
    assert pol.max_tokens == 12000


def test_set_policy_governs_budget() -> None:
    pack = ContextPack(token_budget=200_000)
    pack.set_policy(ContextPolicy(max_tokens=24000, reserved_output_tokens=4000))
    assert pack.token_budget == 24000
    assert pack.reserved_output_tokens == 4000


def test_forbidden_source_excluded() -> None:
    policy = ContextPolicy(max_tokens=20000, reserved_output_tokens=2000, forbidden_sources=[".env"])
    pack = ContextPack(policy=policy)
    pack.add(name="task", content="do", kind="task", required=True)
    pack.add(name="env", content="SECRET stuff", kind="data", priority=50, source="config/.env")
    compiled = pack.compile()
    names = [it.name for it in compiled.included_items]
    assert "env" not in names
    assert any(v["rule"] == "forbidden_source" for v in compiled.policy_violations)


def test_require_provenance_flags_missing() -> None:
    policy = ContextPolicy(max_tokens=20000, reserved_output_tokens=2000, require_provenance=True)
    pack = ContextPack(policy=policy)
    pack.add(name="task", content="do", kind="task", required=True)  # no source
    compiled = pack.compile()
    assert any(v["rule"] == "missing_provenance" for v in compiled.policy_violations)


def test_fail_on_policy_violation_raises() -> None:
    policy = ContextPolicy(
        max_tokens=20000, reserved_output_tokens=2000, block_secrets=True,
        redact_sensitive=False, fail_on_policy_violation=True,
    )
    pack = ContextPack(policy=policy)
    pack.add(name="task", content="do", kind="task", required=True)
    pack.add(name="leak", content="key sk-LIVE1234567890ABCDEFGH", kind="data", priority=50, source="t.txt")
    with pytest.raises(PolicyViolationError):
        pack.compile()


def test_policy_summary_on_compiled() -> None:
    policy = ContextPolicy(max_tokens=20000, reserved_output_tokens=2000)
    pack = ContextPack(policy=policy)
    pack.add(name="task", content="do", kind="task", required=True)
    compiled = pack.compile()
    assert compiled.policy_summary is not None
    assert compiled.policy_summary["max_tokens"] == 20000

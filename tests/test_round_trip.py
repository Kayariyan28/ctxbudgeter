"""Tests for CompiledPack ↔ dict round trip and the health breakdown."""

from __future__ import annotations

import json

from ctxbudgeter import ContextPack, compiled_pack_from_dict


def _pack() -> ContextPack:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="task body", kind="task", required=True)
    pack.add(name="optional", content="opt", priority=50)
    return pack


def test_to_dict_includes_health_breakdown() -> None:
    compiled = _pack().compile()
    d = compiled.to_dict()
    assert "health_breakdown" in d
    assert isinstance(d["health_breakdown"], dict)
    # Some deduction or bonus key should be present
    assert d["health_breakdown"]  # non-empty for non-trivial pack


def test_compiled_pack_from_dict_round_trip() -> None:
    compiled = _pack().compile()
    d = compiled.to_dict()
    restored = compiled_pack_from_dict(d)
    assert restored.model == compiled.model
    assert restored.token_budget == compiled.token_budget
    assert restored.used_tokens == compiled.used_tokens
    assert restored.health_score == compiled.health_score
    assert len(restored.decisions) == len(compiled.decisions)


def test_to_dict_is_json_serializable() -> None:
    compiled = _pack().compile()
    serialized = json.dumps(compiled.to_dict(), default=str)
    parsed = json.loads(serialized)
    assert parsed["model"] == "claude-sonnet-4.6"


def test_decisions_have_sensitivity_field() -> None:
    compiled = _pack().compile()
    for d in compiled.to_dict()["decisions"]:
        assert "sensitivity" in d

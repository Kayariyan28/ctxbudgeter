"""Tests for sensitivity policy enforcement."""

from __future__ import annotations

import pytest

from ctxbudgeter import ContextPack, SecretContentError
from ctxbudgeter.compiler import REDACTED_PLACEHOLDER


def _pack_with_secret(policy: str) -> ContextPack:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add(name="api_key", content="sk-DEADBEEF12345", sensitivity="secret", priority=70)
    pack.set_secret_policy(policy)
    return pack


def test_default_policy_is_warn() -> None:
    pack = _pack_with_secret("warn")
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["api_key"] == "included"
    assert compiled.has_secrets
    assert any("api_key" in w for w in compiled.warnings)
    # Health is penalized for secret inclusion
    assert "secrets_included" in compiled.health_breakdown


def test_refuse_policy_raises() -> None:
    pack = _pack_with_secret("refuse")
    with pytest.raises(SecretContentError):
        pack.compile()


def test_redact_policy_replaces_content() -> None:
    pack = _pack_with_secret("redact")
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["api_key"] == "redacted"
    assert compiled.content_for("api_key") == REDACTED_PLACEHOLDER
    # Original sensitive content not in the rendered prompt
    assert "sk-DEADBEEF" not in compiled.as_text()


def test_allow_policy_no_warning_or_penalty() -> None:
    pack = _pack_with_secret("allow")
    compiled = pack.compile()
    statuses = {d.name: d.status for d in compiled.decisions}
    assert statuses["api_key"] == "included"
    assert "secrets_included" not in compiled.health_breakdown


def test_secret_flagged_in_report() -> None:
    pack = _pack_with_secret("warn")
    compiled = pack.compile()
    text = compiled.report("text")
    md = compiled.report("markdown")
    assert "secret" in text.lower()
    assert "secret" in md.lower()

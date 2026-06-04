"""Tests for ContextScanner — detection + the no-leak invariant."""

from __future__ import annotations

from ctxbudgeter import ContextScanner

SECRETS = {
    "openai_key": "sk-ABCDEF1234567890GHIJKL",
    "anthropic_key": "sk-ant-api03-ABCDEF1234567890",
    "github_token": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "stripe_key": "sk_live_ABCDEF1234567890GHIJKL",
}


def test_detects_email() -> None:
    r = ContextScanner().scan("contact me at alice@example.com please")
    assert r.has_pii
    assert any(f.category == "email" for f in r.findings)


def test_detects_api_keys() -> None:
    s = ContextScanner()
    for category, secret in SECRETS.items():
        r = s.scan(f"here is the value {secret} ok")
        assert r.has_secrets, f"{category} not detected"
        cats = {f.category for f in r.findings}
        assert category in cats or "api_key_generic" in cats


def test_never_exposes_full_secret() -> None:
    """The core security invariant: no finding preview or masked_text contains a full secret."""
    s = ContextScanner()
    for secret in SECRETS.values():
        r = s.scan(f"value: {secret}")
        for f in r.findings:
            assert secret not in f.matched_preview, f"LEAK in preview: {f.matched_preview}"
        assert secret not in r.masked_text, "LEAK in masked_text"
        d = r.to_dict()
        import json

        assert secret not in json.dumps(d), "LEAK in serialized dict"


def test_credit_card_luhn() -> None:
    s = ContextScanner()
    # Valid Luhn (a well-known test card)
    r = s.scan("card 4111 1111 1111 1111 expires soon")
    assert any(f.category == "credit_card" for f in r.findings)
    # Invalid Luhn → not flagged as credit_card
    r2 = s.scan("number 1234 5678 9012 3456 7")
    assert not any(f.category == "credit_card" for f in r2.findings)


def test_private_key_block_masked() -> None:
    s = ContextScanner()
    txt = "-----BEGIN PRIVATE KEY-----\nMIIabc123\n-----END PRIVATE KEY-----"
    r = s.scan(txt)
    assert r.has_secrets
    assert r.risk_level == "critical"
    assert any("[masked]" in f.matched_preview for f in r.findings)


def test_redact_replaces_secrets() -> None:
    s = ContextScanner()
    txt = "key sk-ABCDEF1234567890GHIJKL and email a@b.com"
    red = s.redact(txt)
    assert "sk-ABCDEF1234567890GHIJKL" not in red
    assert "a@b.com" not in red
    assert "[REDACTED" in red


def test_scan_file(tmp_path) -> None:
    p = tmp_path / "secrets.txt"
    p.write_text("token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    r = ContextScanner().scan_file(p)
    assert r.has_secrets


def test_clean_text_no_findings() -> None:
    r = ContextScanner().scan("This is a perfectly clean sentence about refund policy.")
    assert not r.has_secrets
    assert not r.has_pii
    assert r.risk_level == "none"
    assert r.risk_score == 0


def test_risk_score_monotonic() -> None:
    s = ContextScanner()
    low = s.scan("email a@b.com")
    crit = s.scan("key sk-ABCDEF1234567890GHIJKL")
    assert crit.risk_score >= low.risk_score

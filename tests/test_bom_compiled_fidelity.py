"""The compiled-pack BOM path must agree with the full-fidelity BOM path.

`ctxbudgeter compile --bom` and `ctxbudgeter bom <compiled-pack.json>` describe the same
compile, so every field that is not derived from item *content* must match. Governance
metadata used to be dropped when a CompiledPack was rebuilt from its snapshot, which made
the derived BOM report risk_score 0 on context whose snapshot recorded policy violations
and scanner findings — a security gate built on `eval --bom` passed when it should have
failed.
"""

from __future__ import annotations

import json

from ctxbudgeter.bom import ContextBOM
from ctxbudgeter.compiler import compiled_pack_from_dict
from ctxbudgeter.evals import ContextEval, EvalSuite
from ctxbudgeter.pack import ContextPack
from ctxbudgeter.policy import ContextPolicy

# Fields that genuinely cannot survive a content-free snapshot.
CONTENT_DERIVED = {"checksum", "freshness", "relevance_score", "trust_level", "transformations"}
VOLATILE = {"run_id", "created_at"}

SECRET = "API_KEY = 'sk-proj-abc123def456ghi789jkl'\nEMAIL = 'dev@example.com'\n"


def _pack() -> ContextPack:
    pack = ContextPack(model="claude-sonnet-4.6", token_budget=24_000)
    pack.set_policy(ContextPolicy(block_secrets=True, redact_sensitive=True))
    pack.add(name="readme", content="# Project\n\nInternal docs.\n",
             kind="project_doc", priority=80)
    pack.add(name="config", content=SECRET, kind="code", priority=70)
    pack.add(name="task", content="fix the auth bug", kind="task", priority=95, required=True)
    return pack


def _both_boms() -> tuple[ContextBOM, ContextBOM]:
    """Build the full-fidelity BOM and the snapshot-derived BOM from one compile."""
    compiled = _pack().compile(task="fix the auth bug")
    full = ContextBOM.from_compiled(compiled, task="fix the auth bug")
    snapshot = json.loads(json.dumps(compiled.to_dict()))
    derived = ContextBOM.from_compiled(compiled_pack_from_dict(snapshot))
    return full, derived


def test_snapshot_round_trip_preserves_governance_metadata() -> None:
    compiled = _pack().compile(task="fix the auth bug")
    restored = compiled_pack_from_dict(json.loads(json.dumps(compiled.to_dict())))
    assert restored.task == compiled.task
    assert restored.policy_summary == compiled.policy_summary
    assert restored.policy_violations == compiled.policy_violations
    assert restored.scanner_findings == compiled.scanner_findings
    assert restored.item_risk == compiled.item_risk


def test_derived_bom_is_not_empty() -> None:
    _, derived = _both_boms()
    assert derived.included_items, "BOM from a compiled-pack snapshot listed no items"


def test_derived_bom_matches_full_bom_on_non_content_fields() -> None:
    full, derived = _both_boms()
    f, d = json.loads(full.to_json()), json.loads(derived.to_json())
    mismatched = [
        k for k in f
        if k not in VOLATILE and k != "included_items" and f[k] != d[k]
    ]
    assert not mismatched, f"derived BOM diverges on {mismatched}"


def test_derived_bom_items_match_on_non_content_fields() -> None:
    full, derived = _both_boms()
    f = {i.name: i.to_dict() for i in full.included_items}
    d = {i.name: i.to_dict() for i in derived.included_items}
    assert sorted(f) == sorted(d)
    for name, fi in f.items():
        for key, want in fi.items():
            if key in CONTENT_DERIVED:
                continue
            assert d[name][key] == want, f"{name}.{key}: {d[name][key]!r} != {want!r}"


def test_risk_score_survives_the_snapshot() -> None:
    """The bug that mattered: a clean bill of health on context with detected secrets."""
    full, derived = _both_boms()
    assert full.risk_score > 0, "fixture no longer produces risk; test is vacuous"
    assert derived.risk_score == full.risk_score


def test_eval_gate_agrees_across_both_bom_paths() -> None:
    """A CI gate must not pass merely because the BOM came via a snapshot."""
    full, derived = _both_boms()
    suite = EvalSuite(evals=[ContextEval(name="gate", max_risk_score=10)])
    full_passed = all(r.passed for r in suite.run(full))
    derived_passed = all(r.passed for r in suite.run(derived))
    assert not full_passed, "fixture no longer trips the gate; test is vacuous"
    assert derived_passed == full_passed

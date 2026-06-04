"""Tests for ContextBOM."""

from __future__ import annotations

import json

from ctxbudgeter import ContextBOM, ContextPack, ContextPolicy


def _compiled(policy: bool = False):
    kwargs = {}
    if policy:
        kwargs["policy"] = ContextPolicy(max_tokens=20000, reserved_output_tokens=2000, block_secrets=True)
    pack = ContextPack(**kwargs) if policy else ContextPack(token_budget=20000, reserved_output_tokens=2000)
    pack.add(name="system", content="rules " * 30, kind="system", required=True,
             cache_policy="stable", source="prompts/system.md", trust_level="verified")
    pack.add(name="task", content="do the thing", kind="task", required=True)
    # Big enough to exceed the 18k available-token budget so it's excluded.
    pack.add(name="junk", content="lorem ipsum dolor " * 30000, priority=5)
    return pack.compile(task="demo task")


def test_bom_from_compiled() -> None:
    bom = _compiled().bom
    assert bom.task == "demo task"
    assert bom.total_tokens > 0
    assert any(i.name == "system" for i in bom.included_items)
    assert bom.run_id.startswith("bom-")


def test_bom_json_export_deterministic(tmp_path) -> None:
    b1 = _compiled().to_dict() if False else _compiled().bom
    out = tmp_path / "bom.json"
    j1 = b1.to_json(str(out))
    j2 = _compiled().bom.to_json()
    assert j1 == j2, "BOM JSON must be deterministic across identical compiles"
    assert out.exists()


def test_bom_markdown_export(tmp_path) -> None:
    out = tmp_path / "bom.md"
    md = _compiled().bom.to_markdown(str(out))
    assert "# Context Bill of Materials" in md
    assert "## Included" in md
    assert out.exists()


def test_bom_from_json_roundtrip(tmp_path) -> None:
    out = tmp_path / "bom.json"
    bom = _compiled().bom
    bom.to_json(str(out))
    loaded = ContextBOM.from_json(str(out))
    assert loaded.run_id == bom.run_id
    assert loaded.total_tokens == bom.total_tokens
    assert {i.name for i in loaded.included_items} == {i.name for i in bom.included_items}


def test_bom_excludes_junk() -> None:
    bom = _compiled().bom
    excluded = {i.name for i in bom.excluded_items}
    assert "junk" in excluded


def test_bom_never_leaks_secret() -> None:
    pack = ContextPack(policy=ContextPolicy(max_tokens=20000, reserved_output_tokens=2000,
                                            block_secrets=True, redact_sensitive=True))
    pack.add(name="task", content="do", kind="task", required=True)
    pack.add(name="leak", content="key sk-ABCDEF1234567890GHIJKL", kind="data", priority=50, source="t.txt")
    bom = pack.compile().bom
    assert "sk-ABCDEF1234567890GHIJKL" not in json.dumps(bom.to_dict())


def test_bom_has_provenance_summary() -> None:
    bom = _compiled().bom
    assert "by_trust_level" in bom.provenance_summary
    assert bom.provenance_summary["items_total"] == len(bom.included_items)


def test_bom_recommendations_for_excluded_highpri() -> None:
    pack = ContextPack(token_budget=2000, reserved_output_tokens=500)
    pack.add(name="task", content="do", kind="task", required=True)
    pack.add(name="big", content="x " * 5000, priority=90)  # high pri, excluded
    bom = pack.compile().bom
    assert any("big" in r for r in bom.recommendations)

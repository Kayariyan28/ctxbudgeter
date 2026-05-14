"""Tests for the declarative YAML/JSON pack spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxbudgeter.spec import SpecError, build_pack, dump_pack, load_pack, validate

YAML_SPEC = """
model: claude-sonnet-4.6
token_budget: 24000
reserved_output_tokens: 4000
secret_policy: warn

items:
  - name: system_rules
    content: |
      You are a careful coding agent.
    kind: system
    priority: 100
    required: true
    cache_policy: stable

  - name: task
    content: "Fix the auth bug."
    kind: task
    priority: 95
    required: true
"""

JSON_SPEC = """
{
  "model": "claude-sonnet-4.6",
  "token_budget": 24000,
  "reserved_output_tokens": 4000,
  "items": [
    {"name": "task", "content": "test", "kind": "task", "priority": 95, "required": true}
  ]
}
"""


def test_load_yaml_spec(tmp_path: Path) -> None:
    p = tmp_path / "pack.yaml"
    p.write_text(YAML_SPEC)
    pack = load_pack(p)
    assert pack.model == "claude-sonnet-4.6"
    assert pack.token_budget == 24_000
    assert "system_rules" in pack
    assert "task" in pack


def test_load_json_spec(tmp_path: Path) -> None:
    p = tmp_path / "pack.json"
    p.write_text(JSON_SPEC)
    pack = load_pack(p)
    compiled = pack.compile()
    assert any(d.name == "task" for d in compiled.decisions)


def test_from_file_inlines_content(tmp_path: Path) -> None:
    doc = tmp_path / "docs.md"
    doc.write_text("# Hello\nWorld.")
    spec = tmp_path / "pack.yaml"
    spec.write_text(f"""
model: test-model
token_budget: 5000
reserved_output_tokens: 500
items:
  - name: task
    content: do
    kind: task
    required: true
  - name: docs
    from_file: {doc.name}
    kind: project_doc
    priority: 80
    cache_policy: stable
""")
    pack = load_pack(spec)
    docs_item = pack.get("docs")
    assert docs_item is not None
    assert "# Hello" in docs_item.content


def test_unknown_top_level_key_rejected() -> None:
    data = {"model": "x", "weird_key": True, "items": []}
    with pytest.raises(SpecError, match="Unknown top-level"):
        build_pack(data)


def test_unknown_item_key_rejected() -> None:
    data = {"items": [{"name": "a", "content": "b", "wat": 1}]}
    with pytest.raises(SpecError, match="Unknown item keys"):
        build_pack(data)


def test_item_with_both_content_and_from_file_rejected() -> None:
    data = {"items": [{"name": "a", "content": "x", "from_file": "y.txt"}]}
    with pytest.raises(SpecError, match="content OR from_file"):
        build_pack(data)


def test_validate_returns_issues_for_bad_spec(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("items: []\nunknown_field: 1\n")
    issues = validate(p)
    assert issues


def test_validate_empty_for_good_spec(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text(YAML_SPEC)
    assert validate(p) == []


def test_references_loaded_with_named_loader() -> None:
    data = {
        "items": [{"name": "task", "content": "t", "kind": "task", "required": True}],
        "references": [
            {
                "name": "msg",
                "location": "hello world",
                "loader": "inline",
                "priority": 70,
                "estimated_tokens": 10,
            }
        ],
    }
    pack = build_pack(data)
    compiled = pack.compile()
    by = {d.name: d.status for d in compiled.decisions}
    assert by["msg"] == "included"


def test_references_unknown_loader_raises() -> None:
    data = {
        "references": [{"name": "x", "location": "-", "loader": "nope_does_not_exist"}],
    }
    with pytest.raises(KeyError):
        build_pack(data)


def test_dump_pack_round_trip() -> None:
    data = {
        "model": "claude-sonnet-4.6",
        "token_budget": 10_000,
        "reserved_output_tokens": 1_000,
        "items": [
            {"name": "task", "content": "x", "kind": "task", "required": True},
            {"name": "doc", "content": "docs", "kind": "project_doc", "priority": 70},
        ],
    }
    pack = build_pack(data)
    dumped = dump_pack(pack)
    assert dumped["model"] == "claude-sonnet-4.6"
    assert len(dumped["items"]) == 2
    # Re-build from dumped data
    pack2 = build_pack(dumped)
    assert pack2.model == pack.model
    assert {it.name for it in pack2.items} == {it.name for it in pack.items}

"""Tests for ContextDiff."""

from __future__ import annotations

from ctxbudgeter import ContextDiff, ContextPack


def _bom(*, extra: bool = False, big_junk: bool = False):
    p = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    p.add(name="system", content="rules " * 20, kind="system", required=True, cache_policy="stable")
    p.add(name="task", content="do", kind="task", required=True)
    if extra:
        p.add(name="extra", content="y " * 40, priority=50, source="docs/x.md")
    if big_junk:
        p.add(name="task", content="do", kind="task")  # ignored dup; placeholder
    return p.compile().bom


def test_diff_detects_added() -> None:
    d = ContextDiff.compare(_bom(), _bom(extra=True))
    assert "extra" in [i["name"] for i in d.added_items]
    assert d.token_change > 0


def test_diff_detects_removed() -> None:
    d = ContextDiff.compare(_bom(extra=True), _bom())
    assert "extra" in [i["name"] for i in d.removed_items]
    assert d.token_change < 0


def test_diff_no_changes() -> None:
    d = ContextDiff.compare(_bom(), _bom())
    assert not d.has_changes
    assert d.token_change == 0


def test_diff_token_change() -> None:
    d = ContextDiff.compare(_bom(), _bom(extra=True))
    assert d.token_change == d.token_change  # deterministic
    assert isinstance(d.token_change, int)


def test_diff_from_files(tmp_path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _bom().to_json(str(old))
    _bom(extra=True).to_json(str(new))
    d = ContextDiff.compare(str(old), str(new))
    assert "extra" in [i["name"] for i in d.added_items]


def test_diff_serialization(tmp_path) -> None:
    d = ContextDiff.compare(_bom(), _bom(extra=True))
    assert "Context diff" in d.to_text()
    assert "# Context Diff" in d.to_markdown()
    import json

    parsed = json.loads(d.to_json())
    assert "added_items" in parsed


def test_diff_changed_items() -> None:
    p1 = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    p1.add(name="doc", content="short", kind="project_doc", required=True, source="a.md")
    b1 = p1.compile().bom
    p2 = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    p2.add(name="doc", content="much longer content " * 20, kind="project_doc", required=True, source="a.md")
    b2 = p2.compile().bom
    d = ContextDiff.compare(b1, b2)
    assert any(c.name == "doc" and "tokens" in c.field_changes for c in d.changed_items)

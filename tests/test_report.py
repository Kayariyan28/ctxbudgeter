"""Tests for the reporting layer."""

from __future__ import annotations

import json

from ctxbudgeter import ContextPack, to_json, to_markdown, to_text


def _compiled() -> object:
    pack = ContextPack(model="claude-sonnet-4.6", token_budget=2_000, reserved_output_tokens=500)
    pack.add(name="sys", content="rules", kind="system", priority=100, cache_policy="stable", required=True)
    pack.add(name="task", content="do thing", kind="task", priority=95, required=True)
    pack.add(name="doc", content="docs " * 20, kind="project_doc", priority=70, cache_policy="stable")
    # Use varied content so BPE can't compress it; guaranteed to overflow.
    pack.add(
        name="junk",
        content="\n".join(f"junk_line_{i}_value_{i * 13}" for i in range(5_000)),
        priority=5,
    )
    return pack.compile()


def test_text_report_contains_included_and_excluded() -> None:
    compiled = _compiled()
    txt = to_text(compiled)
    assert "Included:" in txt
    assert "Excluded:" in txt
    assert "sys" in txt
    assert "task" in txt
    assert "junk" in txt
    assert "health score" in txt.lower()


def test_text_report_shows_token_counts() -> None:
    compiled = _compiled()
    txt = to_text(compiled)
    assert "tokens" in txt
    assert "budget" in txt.lower()


def test_markdown_report_has_tables() -> None:
    compiled = _compiled()
    md = to_markdown(compiled)
    assert "# Context Compilation Report" in md
    assert "| Name | Kind |" in md  # included table header
    assert "## Excluded" in md


def test_json_report_is_valid_json() -> None:
    compiled = _compiled()
    j = to_json(compiled)
    parsed = json.loads(j)
    assert parsed["model"] == "claude-sonnet-4.6"
    assert "decisions" in parsed
    assert parsed["health_score"] >= 0
    assert parsed["health_score"] <= 100
    # Each decision has a status
    for d in parsed["decisions"]:
        assert d["status"] in ("included", "excluded", "compressed", "truncated")


def test_compiled_report_method_dispatches_by_format() -> None:
    compiled = _compiled()
    assert "Included:" in compiled.report("text")
    assert "# Context Compilation Report" in compiled.report("markdown")
    json.loads(compiled.report("json"))


def test_to_dict_round_trip() -> None:
    compiled = _compiled()
    d = compiled.to_dict()
    j = json.dumps(d, default=str)
    parsed = json.loads(j)
    assert parsed["included_order"]
    assert parsed["used_tokens"] > 0

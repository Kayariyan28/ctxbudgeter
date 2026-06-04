"""Tests for MCPToolBudgeter."""

from __future__ import annotations

import json

import pytest

from ctxbudgeter import MCPToolBudgeter

TOOLS = [
    {"name": "create_github_issue", "description": "Create a GitHub issue from text",
     "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}}},
    {"name": "delete_database", "description": "Delete the production database permanently",
     "inputSchema": {"type": "object"}},
    {"name": "send_email", "description": "Send an email to a recipient",
     "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}}},
    {"name": "open_github_ticket", "description": "Create a GitHub issue from text",
     "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}}},
]


def test_budget_validation() -> None:
    with pytest.raises(ValueError):
        MCPToolBudgeter(token_budget=0)


def test_estimates_token_costs() -> None:
    result = MCPToolBudgeter().audit(TOOLS)
    assert all(t.tokens > 0 for t in result.assessments)
    assert result.total_tokens > 0


def test_selects_relevant_tools() -> None:
    result = MCPToolBudgeter(token_budget=5000).select_tools(
        task="Create a GitHub issue from a customer complaint", tools=TOOLS
    )
    assert "create_github_issue" in result.selected_tools
    assert "delete_database" not in result.selected_tools


def test_flags_risky_tools() -> None:
    result = MCPToolBudgeter().audit(TOOLS)
    assert "delete_database" in result.risky_tools
    assert "send_email" in result.risky_tools


def test_detects_overlap() -> None:
    result = MCPToolBudgeter().audit(TOOLS)
    flat = [t for g in result.overlap_groups for t in g]
    assert "create_github_issue" in flat or "open_github_ticket" in flat


def test_budget_respected() -> None:
    result = MCPToolBudgeter(token_budget=60).select_tools(task="Create a GitHub issue", tools=TOOLS)
    assert result.selected_tokens <= 60


def test_load_from_json_file(tmp_path) -> None:
    p = tmp_path / "tools.json"
    p.write_text(json.dumps(TOOLS))
    result = MCPToolBudgeter().audit(str(p))
    assert len(result.assessments) == len(TOOLS)


def test_result_json_serializable() -> None:
    result = MCPToolBudgeter().select_tools(task="github issue", tools=TOOLS)
    parsed = json.loads(result.to_json())
    assert "selected_tools" in parsed

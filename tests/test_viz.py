"""Tests for the viz layer — HTML generation + the no-leak invariant."""

from __future__ import annotations

from ctxbudgeter import ContextPack, ContextPolicy
from ctxbudgeter.viz import ContextDiffViz, ContextMRI, MCPToolViz

SECRET = "sk-ABCDEF1234567890SECRETXYZ"


def _compiled_with_secret():
    policy = ContextPolicy(max_tokens=20000, reserved_output_tokens=2000,
                           block_secrets=True, redact_sensitive=True)
    p = ContextPack(policy=policy)
    p.add(name="system", content="rules " * 20, kind="system", required=True,
          cache_policy="stable", source="prompts/system.md", trust_level="verified")
    p.add(name="leak", content=f"token {SECRET}", kind="data", priority=50, source="t.txt")
    p.add(name="task", content="do the task", kind="task", required=True)
    return p.compile(task="demo")


def test_mri_generates_html() -> None:
    mri = ContextMRI.from_compiled(_compiled_with_secret())
    html = mri.to_html()
    assert "<!DOCTYPE html>" in html
    assert "Context MRI" in html
    # all 8 panels present
    for panel in ["Summary", "Context Window Map", "Source", "Risk Heatmap",
                  "Cache Boundary", "Context Waste", "Influence Proxy", "Recommendations"]:
        assert panel in html, f"missing panel: {panel}"


def test_mri_html_no_secret_leak() -> None:
    html = ContextMRI.from_compiled(_compiled_with_secret()).to_html()
    assert SECRET not in html


def test_mri_influence_disclaimer_present() -> None:
    html = ContextMRI.from_compiled(_compiled_with_secret()).to_html()
    assert "NOT model attention" in html


def test_mri_window_order_preserved() -> None:
    compiled = _compiled_with_secret()
    data = ContextMRI.from_compiled(compiled).to_data()
    window_names = [w["name"] for w in data["window"]]
    included_names = [it.name for it in compiled.included_items]
    assert window_names == included_names


def test_mri_export_html(tmp_path) -> None:
    out = tmp_path / "mri.html"
    ContextMRI.from_compiled(_compiled_with_secret()).export_html(str(out))
    assert out.exists()
    assert SECRET not in out.read_text(encoding="utf-8")


def test_mri_export_json(tmp_path) -> None:
    out = tmp_path / "mri.json"
    ContextMRI.from_compiled(_compiled_with_secret()).export_json(str(out))
    assert SECRET not in out.read_text(encoding="utf-8")
    assert "influence_proxy" in out.read_text(encoding="utf-8")


def test_diff_viz_html(tmp_path) -> None:
    a = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    a.add(name="sys", content="r " * 20, kind="system", required=True, cache_policy="stable")
    a.add(name="task", content="x", required=True)
    bom_a = a.compile().bom
    b = ContextPack(token_budget=20000, reserved_output_tokens=2000)
    b.add(name="sys", content="r " * 20, kind="system", required=True, cache_policy="stable")
    b.add(name="task", content="x", required=True)
    b.add(name="new", content="y " * 30, priority=50)
    bom_b = b.compile().bom
    html = ContextDiffViz.from_boms(bom_a, bom_b).to_html()
    assert "Context Diff" in html
    assert "Before / After" in html


def test_mcp_viz_html(tmp_path) -> None:
    import json

    tools = [
        {"name": "create_github_issue", "description": "Create a GitHub issue",
         "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}}},
        {"name": "delete_repo", "description": "Delete a repository", "inputSchema": {"type": "object"}},
    ]
    p = tmp_path / "tools.json"
    p.write_text(json.dumps(tools))
    html = MCPToolViz.from_file(str(p), task="Create a GitHub issue", budget=3000).to_html()
    assert "MCP Tool Map" in html
    assert "Context Savings" in html

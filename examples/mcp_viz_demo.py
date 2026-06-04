"""Budget MCP tool schemas for a task and render an MCP tool map.

Run:  python examples/mcp_viz_demo.py
Writes examples/out/mcp_map.html.
"""

from __future__ import annotations

import json
from pathlib import Path

from ctxbudgeter.mcp import MCPToolBudgeter
from ctxbudgeter.viz import MCPToolViz

SAMPLE = Path(__file__).parent / "sample_data" / "mcp_tools.json"
TASK = "Create a GitHub issue from a customer complaint"


def main() -> None:
    tools = json.loads(SAMPLE.read_text())
    result = MCPToolBudgeter(token_budget=4_000).select_tools(task=TASK, tools=tools)
    print(f"Task: {TASK}")
    print(f"Selected ({result.selected_tokens} tokens): {result.selected_tools}")
    print(f"Excluded: {result.excluded_tools}")
    print(f"Risky: {result.risky_tools}")
    if result.overlap_groups:
        print(f"Overlap groups: {result.overlap_groups}")

    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    target = out / "mcp_map.html"
    MCPToolViz.from_file(str(SAMPLE), task=TASK, budget=4_000).export_html(str(target))
    print(f"\nMCP tool map written to {target}")


if __name__ == "__main__":
    main()

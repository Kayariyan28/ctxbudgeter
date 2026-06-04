# MCP Tool Budgeting

Agents often dump every MCP tool schema into context. Each schema costs tokens, and
oversized/overlapping/risky tool sets hurt cost and reliability. `MCPToolBudgeter`
ranks tools by lexical relevance to the task, estimates token cost, flags risk and
overlap, and selects a budget-respecting subset — deterministically, no LLM calls.

```python
from ctxbudgeter.mcp import MCPToolBudgeter

budgeter = MCPToolBudgeter(token_budget=6_000)
result = budgeter.select_tools(
    task="Create a GitHub issue from a customer complaint",
    tools=all_tools,          # list[dict] | JSON path
)

result.selected_tools         # fit the budget, ranked by relevance
result.excluded_tools
result.token_cost_by_tool
result.schema_complexity_score
result.overlap_groups         # near-duplicate descriptions
result.risky_tools            # delete/execute/send_email/transfer/payment/...
result.warnings
result.recommendations
```

Input accepts a list of dict schemas or a JSON file path. Recognized fields:
`name`, `description`, `inputSchema` (or `input_schema` / `parameters`).

## Heuristics

- **Relevance**: weighted lexical overlap of the task with tool name (×3),
  description (×1.5), and schema field names (×0.5), normalized to 0–1.
- **Risk**: names/descriptions containing `delete`, `drop`, `execute`, `shell`,
  `query_database`, `send_email`, `transfer`, `payment`, `deploy`, `revoke`, …
- **Overlap**: tools whose descriptions have token Jaccard ≥ 0.6.
- **Bloat**: > 20 tools, or any schema > 1,500 tokens / complexity > 40.

## CLI

```bash
ctxbudgeter mcp-audit  mcp_tools.json
ctxbudgeter mcp-select mcp_tools.json --task "Create GitHub issue" --budget 6000 --out selected.json
ctxbudgeter mcp-viz    mcp_tools.json --task "Create GitHub issue" --out mcp_map.html
```

The HTML map shows per-tool token cost, complexity, selected-vs-excluded, risky and
overlapping tools, and an **estimated** context savings figure (estimate only — actual
savings depend on your provider's pricing).

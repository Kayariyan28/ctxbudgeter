"""MCP tool visualization — token cost, complexity, selection, risk, overlap."""

from __future__ import annotations

from dataclasses import dataclass

from ..mcp import MCPBudgetResult, MCPToolBudgeter
from . import _html as H


@dataclass
class MCPToolViz:
    """Render an MCP tool budgeting result as self-contained HTML."""

    result: MCPBudgetResult
    task: str | None

    @classmethod
    def from_file(cls, path: str, *, task: str | None = None, budget: int = 6_000) -> MCPToolViz:
        budgeter = MCPToolBudgeter(token_budget=budget)
        if task:
            result = budgeter.select_tools(task=task, tools=path)
        else:
            result = budgeter.audit(path)
        return cls(result=result, task=task)

    @classmethod
    def from_result(cls, result: MCPBudgetResult, *, task: str | None = None) -> MCPToolViz:
        return cls(result=result, task=task)

    def to_html(self) -> str:
        r = self.result
        selected = set(r.selected_tools)

        summary = H.cards([
            ("Tools", len(r.assessments), "total"),
            ("Selected", len(r.selected_tools), "fit budget"),
            ("Excluded", len(r.excluded_tools), ""),
            ("Total tokens", f"{r.total_tokens:,}", "all schemas"),
            ("Selected tokens", f"{r.selected_tokens:,}", f"budget {r.token_budget:,}"),
            ("Risky tools", len(r.risky_tools), "flagged"),
        ])

        max_tok = max((a.tokens for a in r.assessments), default=1) or 1
        cost_bars = "".join(
            H.bar_row(
                ("✓ " if a.name in selected else "") + a.name + (" ⚠" if a.risky else ""),
                a.tokens, max_tok,
                ("var(--ok)" if a.name in selected else (H.RISK_COLOR["high"] if a.risky else "#6e7681")),
                suffix="t",
            )
            for a in sorted(r.assessments, key=lambda a: -a.tokens)
        )

        # selection table
        rows = "".join(
            f'<tr><td><b>{H.esc(a.name)}</b></td><td>{a.tokens:,}t</td>'
            f'<td>{a.relevance}</td><td>{a.schema_complexity}</td>'
            f'<td>{"⚠ " + ", ".join(a.risk_terms) if a.risky else "—"}</td>'
            f'<td>{"selected" if a.name in selected else "excluded"}</td></tr>'
            for a in sorted(r.assessments, key=lambda a: (-a.relevance, a.name))
        )
        table = ("<table><tr><th>Tool</th><th>Tokens</th><th>Relevance</th><th>Complexity</th>"
                 f"<th>Risk</th><th>Decision</th></tr>{rows}</table>")

        overlaps = ""
        if r.overlap_groups:
            overlaps = H.panel(
                "Overlapping / Duplicate Tools",
                "".join(f'<div class="rec">{H.esc(" ≈ ".join(g))}</div>' for g in r.overlap_groups),
            )

        warnings = ""
        if r.warnings:
            warnings = H.panel("Warnings", "".join(f'<div class="rec">{H.esc(w)}</div>' for w in r.warnings))

        savings = ""
        if self.task and r.total_tokens:
            saved = r.total_tokens - r.selected_tokens
            pct = round(100 * saved / r.total_tokens) if r.total_tokens else 0
            savings = H.panel(
                "Context Savings (estimate)",
                f'<p>Selecting {len(r.selected_tools)} of {len(r.assessments)} tools for this task '
                f'uses <b>{r.selected_tokens:,}</b> tokens instead of <b>{r.total_tokens:,}</b> — '
                f'an estimated <b>{saved:,}</b> tokens ({pct}%) saved. Estimate only.</p>',
            )

        recs = ""
        if r.recommendations:
            recs = H.panel("Recommendations", "".join(f'<div class="rec">{H.esc(x)}</div>' for x in r.recommendations))

        body = "\n".join([
            H.panel("Summary", summary),
            H.panel("Tool Token Cost", cost_bars),
            savings,
            H.panel("Selected vs Excluded", table),
            overlaps,
            warnings,
            recs,
        ])
        subtitle = f"task: {self.task or '(audit — no task)'}  ·  budget: {r.token_budget:,} tokens"
        return H.page("MCP Tool Map", subtitle, body)

    def export_html(self, path: str) -> str:
        from pathlib import Path

        html = self.to_html()
        Path(path).write_text(html, encoding="utf-8")
        return html

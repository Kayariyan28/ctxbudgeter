"""MCP tool-schema budgeting.

Agent developers routinely dump every MCP tool schema into the model context.
Each schema costs tokens, and oversized / overlapping / risky tool sets hurt both
cost and reliability. ``MCPToolBudgeter`` ranks tools by lexical relevance to the
task, estimates per-tool token cost, flags risky and duplicate tools, and selects
a budget-respecting subset.

Deterministic, offline, no LLM calls. Accepts list-of-dict schemas or a JSON path.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .tokenizer import TokenCounter

RISKY_TOOL_TERMS = (
    "delete", "drop", "execute", "exec", "shell", "run_command", "query_database",
    "sql", "send_email", "transfer", "payment", "charge", "refund", "wire",
    "deploy", "terminate", "destroy", "revoke", "grant", "sudo",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class ToolAssessment:
    """Per-tool analysis."""

    name: str
    tokens: int
    relevance: float
    schema_complexity: int
    risky: bool
    risk_terms: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MCPBudgetResult:
    """Result of selecting tools under a token budget."""

    selected_tools: list[str] = field(default_factory=list)
    excluded_tools: list[str] = field(default_factory=list)
    token_cost_by_tool: dict[str, int] = field(default_factory=dict)
    schema_complexity_score: dict[str, int] = field(default_factory=dict)
    relevance_by_tool: dict[str, float] = field(default_factory=dict)
    overlap_groups: list[list[str]] = field(default_factory=list)
    risky_tools: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    total_tokens: int = 0
    selected_tokens: int = 0
    token_budget: int = 0
    assessments: list[ToolAssessment] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


class MCPToolBudgeter:
    """Select and audit MCP tool schemas under a token budget."""

    def __init__(self, token_budget: int = 6_000, *, model: str = "claude-sonnet-4.6") -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.token_budget = token_budget
        self._counter = TokenCounter(model)

    # ----- public API -------------------------------------------------------

    def audit(self, tools: list[dict] | str | Path) -> MCPBudgetResult:
        """Assess tools without a task (no selection) — cost, complexity, risk, overlap."""
        return self._run(self._load(tools), task=None, select=False)

    def select_tools(
        self, task: str, tools: list[dict] | str | Path
    ) -> MCPBudgetResult:
        """Rank by relevance to ``task`` and select a budget-respecting subset."""
        return self._run(self._load(tools), task=task, select=True)

    # ----- internals --------------------------------------------------------

    def _load(self, tools: list[dict] | str | Path) -> list[dict]:
        if isinstance(tools, (str, Path)):
            data = json.loads(Path(tools).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Accept {"tools": [...]} or a single tool dict.
                tools_list = data.get("tools", [data])
            else:
                tools_list = data
        else:
            tools_list = tools
        if not isinstance(tools_list, list):
            raise ValueError("MCP tools must be a list of schema dicts (or a JSON file containing one).")
        return tools_list

    def _run(self, tools: list[dict], *, task: str | None, select: bool) -> MCPBudgetResult:
        task_tokens = set(_tokenize(task)) if task else set()

        assessments: list[ToolAssessment] = []
        cost: dict[str, int] = {}
        complexity: dict[str, int] = {}
        relevance: dict[str, float] = {}
        risky: list[str] = []

        descriptions: dict[str, str] = {}
        for raw in tools:
            name = str(raw.get("name") or raw.get("tool_name") or raw.get("title") or "unnamed")
            desc = str(raw.get("description") or raw.get("desc") or "")
            schema = raw.get("inputSchema") or raw.get("input_schema") or raw.get("parameters") or {}
            schema_text = json.dumps(schema, sort_keys=True) if schema else ""
            full_text = json.dumps(raw, sort_keys=True)
            tok = self._counter.count(full_text)
            cplx = _schema_complexity(schema)
            rel = _relevance(task_tokens, name, desc, schema_text) if task else 0.0
            risk_terms = sorted({t for t in RISKY_TOOL_TERMS if t in name.lower() or t in desc.lower()})
            is_risky = bool(risk_terms)

            cost[name] = tok
            complexity[name] = cplx
            relevance[name] = round(rel, 4)
            descriptions[name] = desc.strip().lower()
            if is_risky:
                risky.append(name)
            assessments.append(ToolAssessment(
                name=name, tokens=tok, relevance=round(rel, 4),
                schema_complexity=cplx, risky=is_risky, risk_terms=risk_terms,
                description=desc[:200],
            ))

        overlap_groups = _overlap_groups(descriptions)
        warnings = self._warnings(assessments, overlap_groups, sum(cost.values()))

        # Selection: required ordering is deterministic — relevance desc, then
        # tokens asc, then name asc. Greedily pack under budget.
        selected: list[str] = []
        excluded: list[str] = []
        selected_tokens = 0
        if select:
            ranked = sorted(
                assessments,
                key=lambda a: (-a.relevance, a.tokens, a.name),
            )
            for a in ranked:
                if selected_tokens + a.tokens <= self.token_budget and a.relevance > 0:
                    selected.append(a.name)
                    selected_tokens += a.tokens
                else:
                    excluded.append(a.name)
            # If nothing matched the task at all, fall back to cheapest-first fill
            # so the agent still gets *some* tools, but warn loudly.
            if not selected and assessments:
                warnings.append(
                    "No tool matched the task lexically; selecting cheapest tools as a fallback. "
                    "Review tool descriptions or refine the task."
                )
                for a in sorted(assessments, key=lambda a: (a.tokens, a.name)):
                    if selected_tokens + a.tokens <= self.token_budget:
                        selected.append(a.name)
                        selected_tokens += a.tokens
                        if a.name in excluded:
                            excluded.remove(a.name)
        else:
            selected_tokens = sum(cost.values())

        recommendations = self._recommendations(
            assessments, overlap_groups, risky, selected, excluded, select
        )

        return MCPBudgetResult(
            selected_tools=selected,
            excluded_tools=sorted(excluded),
            token_cost_by_tool=cost,
            schema_complexity_score=complexity,
            relevance_by_tool=relevance,
            overlap_groups=overlap_groups,
            risky_tools=sorted(risky),
            warnings=warnings,
            recommendations=recommendations,
            total_tokens=sum(cost.values()),
            selected_tokens=selected_tokens,
            token_budget=self.token_budget,
            assessments=sorted(assessments, key=lambda a: a.name),
        )

    def _warnings(
        self, assessments: list[ToolAssessment], overlap: list[list[str]], total: int
    ) -> list[str]:
        w: list[str] = []
        if len(assessments) > 20:
            w.append(f"Tool count bloat: {len(assessments)} tools — large tool lists degrade selection accuracy.")
        if total > self.token_budget:
            w.append(f"All tool schemas total {total:,} tokens, exceeding the budget of {self.token_budget:,}.")
        for a in assessments:
            if a.tokens > 1500:
                w.append(f"Tool '{a.name}' has a very large schema ({a.tokens:,} tokens) — consider trimming it.")
            if a.schema_complexity > 40:
                w.append(f"Tool '{a.name}' has a deeply nested/complex schema (complexity {a.schema_complexity}).")
        for group in overlap:
            w.append(f"Tools may overlap (similar descriptions): {group}.")
        return w

    def _recommendations(
        self, assessments, overlap, risky, selected, excluded, select
    ) -> list[str]:
        recs: list[str] = []
        for a in assessments:
            if a.tokens > 1500:
                recs.append(f"reduce_tool_schema: {a.name}")
        for group in overlap:
            if len(group) > 1:
                recs.append(f"deduplicate_tools: {group}")
        for name in sorted(risky):
            recs.append(f"review_risky_tool: {name}")
        if select and excluded:
            recs.append(f"excluded_for_budget: {sorted(excluded)}")
        return recs


# ----- pure helpers ------------------------------------------------------------


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _relevance(task_tokens: set[str], name: str, desc: str, schema_text: str) -> float:
    """Lexical Jaccard-ish relevance of a tool to the task (0-1)."""
    if not task_tokens:
        return 0.0
    name_tokens = set(_tokenize(name.replace("_", " ").replace("-", " ")))
    desc_tokens = set(_tokenize(desc))
    schema_tokens = set(_tokenize(schema_text))
    # Weight name matches highest, then description, then schema field names.
    name_hits = len(task_tokens & name_tokens)
    desc_hits = len(task_tokens & desc_tokens)
    schema_hits = len(task_tokens & schema_tokens)
    raw = (name_hits * 3.0 + desc_hits * 1.5 + schema_hits * 0.5)
    denom = (len(task_tokens) * 3.0) or 1.0
    return min(1.0, raw / denom)


def _schema_complexity(schema: Any, depth: int = 0) -> int:
    """Rough structural complexity: counts keys/items weighted by nesting depth."""
    if depth > 12:
        return 0
    score = 0
    if isinstance(schema, dict):
        for v in schema.values():
            score += 1 + depth
            score += _schema_complexity(v, depth + 1)
    elif isinstance(schema, list):
        for v in schema:
            score += _schema_complexity(v, depth + 1)
    return score


def _overlap_groups(descriptions: dict[str, str]) -> list[list[str]]:
    """Group tools whose descriptions are highly similar (token Jaccard >= 0.6)."""
    names = list(descriptions)
    token_sets = {n: set(_tokenize(descriptions[n])) for n in names}
    groups: list[list[str]] = []
    used: set[str] = set()
    for i, a in enumerate(names):
        if a in used or not token_sets[a]:
            continue
        group = [a]
        for b in names[i + 1:]:
            if b in used or not token_sets[b]:
                continue
            inter = len(token_sets[a] & token_sets[b])
            union = len(token_sets[a] | token_sets[b]) or 1
            if inter / union >= 0.6:
                group.append(b)
                used.add(b)
        if len(group) > 1:
            used.add(a)
            groups.append(sorted(group))
    return groups

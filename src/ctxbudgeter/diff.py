"""Context diffing — compare two Context Bills of Materials.

``ContextDiff.compare(old, new)`` accepts BOM objects, dicts, or JSON file paths
and reports added/removed/changed items plus deltas in tokens, risk, cache, policy
violations, and health score. Deterministic; useful as a CI gate ("did this PR
change what the agent can see, and did risk go up?").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bom import ContextBOM


@dataclass
class ItemChange:
    name: str
    field_changes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "field_changes": self.field_changes}


@dataclass
class ContextDiff:
    """Structured difference between two BOMs (old → new)."""

    added_items: list[dict[str, Any]] = field(default_factory=list)
    removed_items: list[dict[str, Any]] = field(default_factory=list)
    changed_items: list[ItemChange] = field(default_factory=list)
    token_change: int = 0
    cache_change: int = 0
    risk_change: int = 0
    health_change: int = 0
    policy_violation_change: int = 0
    old_run_id: str | None = None
    new_run_id: str | None = None

    # ----- construction -----------------------------------------------------

    @classmethod
    def compare(
        cls,
        old: ContextBOM | dict | str | Path,
        new: ContextBOM | dict | str | Path,
    ) -> ContextDiff:
        old_bom = _coerce(old)
        new_bom = _coerce(new)

        old_items = {i.name: i for i in old_bom.included_items}
        new_items = {i.name: i for i in new_bom.included_items}

        added = [new_items[n].to_dict() for n in new_items if n not in old_items]
        removed = [old_items[n].to_dict() for n in old_items if n not in new_items]

        changed: list[ItemChange] = []
        tracked = ("tokens", "risk_level", "cache_policy", "source", "trust_level", "priority", "status")
        for name in sorted(set(old_items) & set(new_items)):
            o, n = old_items[name], new_items[name]
            fc: dict[str, dict[str, Any]] = {}
            for f in tracked:
                ov, nv = getattr(o, f, None), getattr(n, f, None)
                if ov != nv:
                    fc[f] = {"old": ov, "new": nv}
            if fc:
                changed.append(ItemChange(name=name, field_changes=fc))

        return cls(
            added_items=sorted(added, key=lambda d: d["name"]),
            removed_items=sorted(removed, key=lambda d: d["name"]),
            changed_items=changed,
            token_change=new_bom.total_tokens - old_bom.total_tokens,
            cache_change=new_bom.cacheable_tokens - old_bom.cacheable_tokens,
            risk_change=new_bom.risk_score - old_bom.risk_score,
            health_change=new_bom.context_health_score - old_bom.context_health_score,
            policy_violation_change=len(new_bom.policy_violations) - len(old_bom.policy_violations),
            old_run_id=old_bom.run_id,
            new_run_id=new_bom.run_id,
        )

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added_items or self.removed_items or self.changed_items
            or self.token_change or self.cache_change or self.risk_change
            or self.health_change or self.policy_violation_change
        )

    @property
    def risk_increased(self) -> bool:
        return self.risk_change > 0

    # ----- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_run_id": self.old_run_id,
            "new_run_id": self.new_run_id,
            "token_change": self.token_change,
            "cache_change": self.cache_change,
            "risk_change": self.risk_change,
            "health_change": self.health_change,
            "policy_violation_change": self.policy_violation_change,
            "added_items": list(self.added_items),
            "removed_items": list(self.removed_items),
            "changed_items": [c.to_dict() for c in self.changed_items],
        }

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_markdown(self, path: str | None = None) -> str:
        text = _diff_markdown(self)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_text(self) -> str:
        def arrow(v: int) -> str:
            return f"+{v}" if v > 0 else str(v)

        lines = [
            "Context diff:",
            f"  tokens:            {arrow(self.token_change)}",
            f"  cacheable tokens:  {arrow(self.cache_change)}",
            f"  risk score:        {arrow(self.risk_change)}",
            f"  health score:      {arrow(self.health_change)}",
            f"  policy violations: {arrow(self.policy_violation_change)}",
            f"  added:   {len(self.added_items)}  {[i['name'] for i in self.added_items]}",
            f"  removed: {len(self.removed_items)}  {[i['name'] for i in self.removed_items]}",
            f"  changed: {len(self.changed_items)}  {[c.name for c in self.changed_items]}",
        ]
        return "\n".join(lines)


def _coerce(obj: ContextBOM | dict | str | Path) -> ContextBOM:
    if isinstance(obj, ContextBOM):
        return obj
    if isinstance(obj, dict):
        return ContextBOM.from_dict(obj)
    return ContextBOM.from_json(str(obj))


def _diff_markdown(diff: ContextDiff) -> str:
    def arrow(v: int) -> str:
        return f"+{v}" if v > 0 else str(v)

    lines = ["# Context Diff", ""]
    lines.append(f"`{diff.old_run_id}` → `{diff.new_run_id}`")
    lines.append("")
    lines.append("| Metric | Change |")
    lines.append("|--------|-------:|")
    lines.append(f"| Tokens | {arrow(diff.token_change)} |")
    lines.append(f"| Cacheable tokens | {arrow(diff.cache_change)} |")
    lines.append(f"| Risk score | {arrow(diff.risk_change)} |")
    lines.append(f"| Health score | {arrow(diff.health_change)} |")
    lines.append(f"| Policy violations | {arrow(diff.policy_violation_change)} |")
    lines.append("")
    if diff.added_items:
        lines.append("## Added")
        lines.append("")
        for i in diff.added_items:
            lines.append(f"- `{i['name']}` ({i.get('kind', '?')}, {i.get('tokens', 0):,} tokens)")
        lines.append("")
    if diff.removed_items:
        lines.append("## Removed")
        lines.append("")
        for i in diff.removed_items:
            lines.append(f"- `{i['name']}` ({i.get('kind', '?')}, {i.get('tokens', 0):,} tokens)")
        lines.append("")
    if diff.changed_items:
        lines.append("## Changed")
        lines.append("")
        for c in diff.changed_items:
            bits = ", ".join(
                f"{k}: {v['old']} → {v['new']}" for k, v in c.field_changes.items()
            )
            lines.append(f"- `{c.name}`: {bits}")
        lines.append("")
    return "\n".join(lines)

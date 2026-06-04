"""Visual context diff — before/after of two BOMs as self-contained HTML."""

from __future__ import annotations

from dataclasses import dataclass

from ..bom import ContextBOM
from ..diff import ContextDiff
from . import _html as H


@dataclass
class ContextDiffViz:
    """Render a ContextDiff as an HTML report."""

    diff: ContextDiff
    old: ContextBOM
    new: ContextBOM

    @classmethod
    def from_files(cls, old: str, new: str) -> ContextDiffViz:
        old_bom = ContextBOM.from_json(str(old))
        new_bom = ContextBOM.from_json(str(new))
        return cls(diff=ContextDiff.compare(old_bom, new_bom), old=old_bom, new=new_bom)

    @classmethod
    def from_boms(cls, old: ContextBOM, new: ContextBOM) -> ContextDiffViz:
        return cls(diff=ContextDiff.compare(old, new), old=old, new=new)

    def to_html(self) -> str:
        d = self.diff

        def delta_card(label: str, v: int, *, lower_better: bool = False) -> tuple[str, object, str]:
            arrow = f"+{v}" if v > 0 else str(v)
            verdict = "—"
            if v != 0:
                up_good = (v > 0) != lower_better
                verdict = "improved" if up_good else "regressed"
            return (label, arrow, verdict)

        summary = H.cards([
            delta_card("Token change", d.token_change, lower_better=True),
            delta_card("Cacheable change", d.cache_change),
            delta_card("Risk change", d.risk_change, lower_better=True),
            delta_card("Health change", d.health_change),
            delta_card("Policy violations", d.policy_violation_change, lower_better=True),
            ("Added", len(d.added_items), "items"),
            ("Removed", len(d.removed_items), "items"),
            ("Changed", len(d.changed_items), "items"),
        ])

        def item_list(title: str, items: list[dict], color: str) -> str:
            if not items:
                return ""
            rows = "".join(
                f'<tr><td><b>{H.esc(i["name"])}</b></td><td>{H.esc(i.get("kind","?"))}</td>'
                f'<td style="text-align:right">{i.get("tokens",0):,}t</td>'
                f'<td class="risk-{i.get("risk_level","none")}">{H.esc(i.get("risk_level","none"))}</td></tr>'
                for i in items
            )
            return H.panel(title, f'<table><tr><th>Item</th><th>Kind</th><th>Tokens</th><th>Risk</th></tr>{rows}</table>')

        changed = ""
        if d.changed_items:
            rows = ""
            for c in d.changed_items:
                bits = "; ".join(
                    f"{k}: {v['old']} → {v['new']}" for k, v in c.field_changes.items()
                )
                rows += f"<tr><td><b>{H.esc(c.name)}</b></td><td>{H.esc(bits)}</td></tr>"
            changed = H.panel("Changed Items", f"<table><tr><th>Item</th><th>Changes</th></tr>{rows}</table>")

        body = "\n".join([
            H.panel("Summary", summary),
            self._window_compare(),
            item_list("Added", d.added_items, H.RISK_COLOR["low"]),
            item_list("Removed", d.removed_items, H.RISK_COLOR["medium"]),
            changed,
        ])
        subtitle = f"{d.old_run_id} → {d.new_run_id}"
        return H.page("Context Diff", subtitle, body)

    def _window_compare(self) -> str:
        def strip(bom: ContextBOM, label: str) -> str:
            max_tok = max((i.tokens for i in bom.included_items), default=1) or 1
            blocks = "".join(
                f'<div class="block" style="width:{max(50,int(180*i.tokens/max_tok))}px;'
                f'background:{H.KIND_COLOR.get(i.kind,"#6e7681")}">'
                f'<div class="bn">{H.esc(i.name)}</div><div class="bm">{i.tokens:,}t</div></div>'
                for i in bom.included_items
            )
            return f'<p class="sub">{H.esc(label)} ({bom.total_tokens:,}t)</p><div class="win">{blocks}</div>'

        return H.panel("Before / After Window", strip(self.old, "BEFORE") + strip(self.new, "AFTER"))

    def export_html(self, path: str) -> str:
        from pathlib import Path

        html = self.to_html()
        Path(path).write_text(html, encoding="utf-8")
        return html

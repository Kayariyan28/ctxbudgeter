"""Context MRI — "see what your agent is about to know before it knows it."

Renders a compiled context's Bill of Materials as a self-contained HTML report
with eight panels: summary cards, context window map, source→context Sankey,
risk heatmap, cache-boundary visualizer, context-waste report, influence-proxy
map, and recommendations.

IMPORTANT: the Influence Proxy is NOT model attention. It is a transparent score
computed from metadata (task similarity, trust, priority, freshness, retrieval
rank, token compactness). This is stated in the report itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..bom import ContextBOM
from . import _html as H

if TYPE_CHECKING:
    from ..compiler import CompiledPack

INFLUENCE_DISCLAIMER = (
    "Influence Proxy is NOT model attention. It is computed from transparent "
    "metadata — task similarity (relevance), source trust, priority, freshness, "
    "retrieval rank, and token compactness — to approximate how much each item is "
    "likely to shape the response. Use it to spot high-cost/low-value context, not "
    "as a claim about the model's internals."
)

_TRUST_SCORE = {"verified": 1.0, "internal": 0.66, "low": 0.33, "unknown": 0.1, None: 0.1}


@dataclass
class ContextMRI:
    """Build and export a Context MRI from a BOM."""

    bom: ContextBOM

    @classmethod
    def from_bom(cls, source: ContextBOM | str) -> ContextMRI:
        if isinstance(source, ContextBOM):
            return cls(bom=source)
        return cls(bom=ContextBOM.from_json(str(source)))

    @classmethod
    def from_compiled(cls, compiled: CompiledPack, **kwargs: Any) -> ContextMRI:
        return cls(bom=ContextBOM.from_compiled(compiled, **kwargs))

    # ----- exports ----------------------------------------------------------

    def to_html(self) -> str:
        b = self.bom
        body = "\n".join([
            self._summary_cards(),
            self._window_map(),
            self._sankey(),
            self._risk_heatmap(),
            self._cache_boundary(),
            self._waste_report(),
            self._influence_map(),
            self._recommendations(),
        ])
        subtitle = f"model: {b.model}  ·  task: {b.task or '—'}  ·  run: {b.run_id}"
        footer = (
            f"ctxbudgeter v{b.package_version} · Context MRI · generated locally, "
            "no network calls, secrets masked."
        )
        return H.page("Context MRI", subtitle, body, footer=footer)

    def export_html(self, path: str) -> str:
        from pathlib import Path

        html = self.to_html()
        Path(path).write_text(html, encoding="utf-8")
        return html

    def to_data(self) -> dict:
        """Structured MRI data (influence proxies, segments) for tooling."""
        return {
            "run_id": self.bom.run_id,
            "model": self.bom.model,
            "task": self.bom.task,
            "summary": {
                "total_tokens": self.bom.total_tokens,
                "cacheable_tokens": self.bom.cacheable_tokens,
                "cache_efficiency_score": self.bom.cache_efficiency_score,
                "context_health_score": self.bom.context_health_score,
                "risk_score": self.bom.risk_score,
            },
            "influence_proxy": self._influence_scores(),
            "window": [
                {"name": i.name, "kind": i.kind, "tokens": i.tokens,
                 "cache_policy": i.cache_policy, "risk_level": i.risk_level}
                for i in self.bom.included_items
            ],
            "disclaimer": INFLUENCE_DISCLAIMER,
        }

    def export_json(self, path: str) -> str:
        from pathlib import Path

        text = json.dumps(self.to_data(), indent=2, sort_keys=True, default=str)
        Path(path).write_text(text, encoding="utf-8")
        return text

    # ----- panels -----------------------------------------------------------

    def _summary_cards(self) -> str:
        b = self.bom
        return H.panel("Summary", H.cards([
            ("Total tokens", f"{b.total_tokens:,}", "input context"),
            ("Included", len(b.included_items), "items"),
            ("Excluded", len(b.excluded_items), "items"),
            ("Redacted", len(b.redacted_items), "sensitive"),
            ("Policy violations", len(b.policy_violations), ""),
            ("Health", f"{b.context_health_score}/100", "context health"),
            ("Risk", f"{b.risk_score}/100", "lower is better"),
            ("Cache efficiency", f"{b.cache_efficiency_score}/100", f"{b.cacheable_tokens:,} cacheable"),
        ]))

    def _window_map(self) -> str:
        b = self.bom
        max_tok = max((i.tokens for i in b.included_items), default=1) or 1
        blocks = []
        for i in b.included_items:
            color = H.KIND_COLOR.get(i.kind, "#6e7681")
            # encodings
            border = "3px solid #fff" if i.included_reason.startswith("required") else "1px solid #2a2f37"
            outline = "outline:2px solid var(--stable);outline-offset:-2px;" if i.cache_policy == "stable" else ""
            opacity = "opacity:0.55;" if (i.status == "redacted") else ""
            risk_mark = ""
            if i.risk_level in ("high", "critical"):
                risk_mark = f'<span style="color:{H.RISK_COLOR[i.risk_level]}">●</span> '
            stripe = ""
            if i.status == "redacted":
                stripe = "background-image:repeating-linear-gradient(45deg,transparent,transparent 5px,rgba(0,0,0,0.25) 5px,rgba(0,0,0,0.25) 10px);"
            width = max(60, int(220 * i.tokens / max_tok))
            blocks.append(
                f'<div class="block" title="{H.esc(i.included_reason)}" '
                f'style="width:{width}px;background:{color};border:{border};{outline}{opacity}{stripe}">'
                f'<div class="bn">{risk_mark}{H.esc(i.name)}</div>'
                f'<div class="bm">{H.esc(i.kind)} · {i.tokens:,}t · p{i.priority}<br>'
                f'{H.esc(i.cache_policy)} · {H.esc(i.trust_level or "?")} · rel {i.relevance_score}</div></div>'
            )
        legend = (
            '<div class="legend">'
            '<span style="color:var(--stable)">stable (blue outline)</span>'
            '<span style="color:#fff">required (thick border)</span>'
            '<span style="color:var(--redact)">redacted (striped/faded)</span>'
            '<span style="color:var(--crit)">high/critical risk (●)</span>'
            "</div>"
        )
        note = "<p class='sub'>Left → right is the exact compiled order the model will read. Block width ∝ tokens.</p>"
        return H.panel("Context Window Map", note + f'<div class="win">{"".join(blocks)}</div>' + legend)

    def _sankey(self) -> str:
        # Text/SVG Sankey-style flow: source → transform → final item.
        rows = []
        for i in self.bom.included_items:
            src = i.source or "(no source)"
            if i.status == "redacted":
                transform = "redacted"
            elif i.status == "compressed":
                transform = "compressed"
            elif i.transformations:
                transform = " → ".join(i.transformations)
            else:
                transform = "included unchanged"
            rows.append(
                f"<tr><td>{H.esc(src)}</td><td style='color:var(--muted)'>→ {H.esc(transform)} →</td>"
                f"<td><b>{H.esc(i.name)}</b> <span class='tag'>{H.esc(i.kind)}</span></td>"
                f"<td style='text-align:right'>{i.tokens:,}t</td></tr>"
            )
        table = (
            "<table><tr><th>Raw source</th><th>Transformation</th><th>Final context item</th><th>Tokens</th></tr>"
            + "".join(rows) + "</table>"
        )
        return H.panel("Source → Context Flow", table)

    def _risk_heatmap(self) -> str:
        cols = ["pii", "secrets", "stale", "untrusted", "policy", "oversized", "cache_breaking", "duplicate"]
        # Aggregate per-item signals from scanner findings + metadata.
        findings_by_item: dict[str, set[str]] = {}
        for f in self.bom.scanner_findings:
            name = str(f.get("item_name", ""))
            cat = str(f.get("category", ""))
            bucket = findings_by_item.setdefault(name, set())
            from ..scanner import _PII_CATEGORIES, _SECRET_CATEGORIES

            if cat in _PII_CATEGORIES:
                bucket.add("pii")
            if cat in _SECRET_CATEGORIES:
                bucket.add("secrets")
        violations_by_item: dict[str, set[str]] = {}
        for v in self.bom.policy_violations:
            nm = str(v.get("item_name") or "")
            if nm:
                violations_by_item.setdefault(nm, set()).add(v.get("rule", "policy"))

        max_tok = max((i.tokens for i in self.bom.included_items), default=1) or 1
        header = "<tr><th>Item</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        rows = []
        for i in self.bom.included_items:
            sig = findings_by_item.get(i.name, set())
            vio = violations_by_item.get(i.name, set())
            cells = []
            flags = {
                "pii": "pii" in sig,
                "secrets": "secrets" in sig,
                "stale": any("stale" in r for r in vio),
                "untrusted": (i.trust_level in (None, "unknown", "low")),
                "policy": bool(vio),
                "oversized": i.tokens > max(1500, max_tok * 0.5),
                "cache_breaking": (i.cache_policy in ("dynamic", "ephemeral") and i.kind in ("system", "tool_def")),
                "duplicate": False,
            }
            for c in cols:
                on = flags[c]
                color = H.RISK_COLOR["high"] if on else "#1c2027"
                glyph = "▲" if on else "·"
                cells.append(f'<td style="text-align:center;color:{color}">{glyph}</td>')
            rows.append(f"<tr><td>{H.esc(i.name)}</td>{''.join(cells)}</tr>")
        return H.panel("Risk Heatmap", f"<table>{header}{''.join(rows)}</table>")

    def _cache_boundary(self) -> str:
        items = self.bom.included_items
        # Determine the consecutive stable prefix.
        prefix = []
        for i in items:
            if i.cache_policy in ("stable", "semi_stable"):
                prefix.append(i)
            else:
                break
        dynamic = [i for i in items if i not in prefix]
        seg = lambda lst, label, color: (  # noqa: E731
            f'<div style="flex:{max(1, sum(i.tokens for i in lst))}">'
            f'<div class="bar" style="background:{color};width:100%">'
            f'{label}: {sum(i.tokens for i in lst):,}t · {len(lst)} items</div></div>'
        )
        boundary = (
            '<div style="display:flex;gap:4px;margin:8px 0">'
            + seg(prefix, "STABLE PREFIX (cacheable)", "var(--stable)")
            + (seg(dynamic, "DYNAMIC / NO-CACHE", "#6e7681") if dynamic else "")
            + "</div>"
        )
        # Reuse the CachePlanner warnings if the BOM carried them via recommendations.
        cache_recs = [r for r in self.bom.recommendations if "cache" in r.lower() or "stable" in r.lower()]
        recs = "".join(f'<div class="rec">{H.esc(r)}</div>' for r in cache_recs) or \
            '<div class="rec" style="color:var(--ok)">No cache-layout warnings.</div>'
        est = (
            f'<p class="sub">Estimated cacheable tokens: <b>{self.bom.cacheable_tokens:,}</b> '
            f'({self.bom.cache_efficiency_score}/100 efficiency). '
            "Estimate only — actual savings depend on your provider's pricing.</p>"
        )
        return H.panel("Cache Boundary", est + boundary + recs)

    def _waste_report(self) -> str:
        waste = []
        for i in self.bom.included_items:
            reasons = []
            if i.tokens > 1500 and (i.relevance_score or 0) < 0.4:
                reasons.append("token-heavy, low relevance")
            if i.risk_level in ("high", "critical") and (i.relevance_score or 0) < 0.5:
                reasons.append("high risk, low value")
            if i.cache_policy in ("dynamic", "ephemeral") and i.kind in ("system", "tool_def"):
                reasons.append("cache-breaking placement")
            if reasons:
                waste.append((i, reasons))
        if not waste:
            inner = '<p style="color:var(--ok)">No significant context waste detected.</p>'
        else:
            rows = "".join(
                f"<tr><td><b>{H.esc(i.name)}</b></td><td>{i.tokens:,}t</td>"
                f"<td>rel {i.relevance_score}</td><td class='risk-{i.risk_level}'>{i.risk_level}</td>"
                f"<td>{H.esc('; '.join(r))}</td></tr>"
                for i, r in waste
            )
            inner = ("<table><tr><th>Item</th><th>Tokens</th><th>Relevance</th><th>Risk</th>"
                     f"<th>Why it's waste</th></tr>{rows}</table>")
        return H.panel("Context Waste Report", inner)

    def _influence_scores(self) -> list[dict]:
        out = []
        max_tok = max((i.tokens for i in self.bom.included_items), default=1) or 1
        for i in self.bom.included_items:
            task_sim = i.relevance_score if i.relevance_score is not None else 0.5
            trust = _TRUST_SCORE.get(i.trust_level, 0.1)
            priority = i.priority / 100.0
            freshness = i.freshness if i.freshness is not None else 1.0
            rank_score = 1.0 if i.retrieval_rank is None else 1.0 / (1 + i.retrieval_rank)
            compactness = 1.0 - min(1.0, i.tokens / (max_tok * 1.5))
            influence = (
                task_sim * 0.35 + trust * 0.20 + priority * 0.15
                + freshness * 0.10 + rank_score * 0.10 + compactness * 0.10
            )
            out.append({
                "name": i.name, "influence_proxy": round(influence, 4),
                "tokens": i.tokens, "risk_level": i.risk_level,
                "task_similarity": round(task_sim, 3), "trust": round(trust, 3),
            })
        out.sort(key=lambda d: -d["influence_proxy"])
        return out

    def _influence_map(self) -> str:
        scores = self._influence_scores()
        if not scores:
            return H.panel("Influence Proxy Map", "<p>No included items.</p>")
        max_inf = max(s["influence_proxy"] for s in scores) or 1.0
        bars = "".join(
            H.bar_row(
                f"{s['name']} ({s['tokens']:,}t, {s['risk_level']})",
                int(s["influence_proxy"] * 100), int(max_inf * 100),
                H.RISK_COLOR.get(s["risk_level"], "#4da3ff"),
            )
            for s in scores
        )
        # Quadrant call-outs
        quads = []
        for s in scores:
            hi_inf = s["influence_proxy"] >= max_inf * 0.6
            hi_risk = s["risk_level"] in ("high", "critical")
            hi_cost = s["tokens"] >= 1500
            if hi_inf and not hi_risk:
                quads.append(("keep", s["name"], "high influence, low risk"))
            elif hi_inf and hi_risk:
                quads.append(("review", s["name"], "high influence, HIGH risk"))
            elif not hi_inf and hi_cost:
                quads.append(("trim", s["name"], "low influence, high token cost"))
            elif not hi_inf and hi_risk:
                quads.append(("drop", s["name"], "low influence, high risk"))
        quad_html = "".join(
            f'<div class="rec"><span class="verb">{H.esc(v)}</span> '
            f'<b>{H.esc(n)}</b> — {H.esc(why)}</div>'
            for v, n, why in quads
        )
        disclaimer = f'<div class="disclaimer">{H.esc(INFLUENCE_DISCLAIMER)}</div>'
        return H.panel("Influence Proxy Map", disclaimer + bars + (quad_html or ""))

    def _recommendations(self) -> str:
        recs = self.bom.recommendations
        if not recs:
            inner = '<p style="color:var(--ok)">No recommendations — context looks healthy.</p>'
        else:
            inner = "".join(f'<div class="rec">{H.esc(r)}</div>' for r in recs)
        return H.panel("Recommendations", inner)

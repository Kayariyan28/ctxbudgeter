"""Shared, dependency-free HTML/SVG building blocks for the viz layer.

Pure standard library. Everything is escaped; nothing remote is referenced.
"""

from __future__ import annotations

from html import escape as _escape

# A calm, enterprise-readable dark theme. All inline so the HTML is self-contained.
BASE_CSS = """
:root {
  --bg: #0d0f12; --panel: #15181d; --panel2: #1c2027; --ink: #e8eaed;
  --muted: #9aa3ad; --line: #2a2f37; --accent: #4da3ff; --stable: #4da3ff;
  --ok: #3fb950; --warn: #d29922; --high: #f0883e; --crit: #f85149;
  --redact: #bc6bd9;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: ui-monospace, "SF Mono", "IBM Plex Mono", Menlo, monospace;
  font-size: 14px; line-height: 1.5; padding: 28px;
}
h1 { font-size: 26px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; font-weight: 600; margin: 28px 0 12px; letter-spacing: 0.02em;
     text-transform: uppercase; color: var(--muted); }
.sub { color: var(--muted); margin: 0 0 18px; font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.card .label { font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); }
.card .value { font-size: 28px; font-weight: 300; margin-top: 6px; }
.card .foot { font-size: 10px; color: var(--muted); margin-top: 6px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
         padding: 18px 20px; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 10px; letter-spacing: 0.1em; }
.bar { height: 26px; border-radius: 5px; display: flex; align-items: center; padding: 0 8px;
       color: #0d0f12; font-size: 11px; font-weight: 600; overflow: hidden; white-space: nowrap; }
.win { display: flex; gap: 3px; align-items: stretch; flex-wrap: nowrap; overflow-x: auto;
       padding: 8px; background: var(--panel2); border-radius: 8px; }
.block { min-width: 60px; border-radius: 6px; padding: 8px; font-size: 10px; color: #0d0f12;
         display: flex; flex-direction: column; justify-content: space-between; }
.block .bn { font-weight: 700; overflow: hidden; text-overflow: ellipsis; }
.block .bm { opacity: 0.8; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 10px;
       border: 1px solid var(--line); color: var(--muted); }
.disclaimer { background: #1a1408; border: 1px solid #4a3a12; color: #e3c879;
              border-radius: 8px; padding: 12px 16px; font-size: 12px; margin: 14px 0; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 11px; color: var(--muted); margin: 8px 0; }
.legend span::before { content: "■ "; }
.rec { padding: 6px 0; border-bottom: 1px solid var(--line); }
.rec .verb { color: var(--accent); font-weight: 600; }
footer { margin-top: 34px; color: var(--muted); font-size: 11px; border-top: 1px solid var(--line); padding-top: 12px; }
.risk-none { color: var(--muted); } .risk-low { color: var(--ok); }
.risk-medium { color: var(--warn); } .risk-high { color: var(--high); }
.risk-critical { color: var(--crit); font-weight: 700; }
"""

# Risk → color mapping for blocks/heatmap.
RISK_COLOR = {
    "none": "#3a4150", "low": "#3fb950", "medium": "#d29922",
    "high": "#f0883e", "critical": "#f85149",
}
KIND_COLOR = {
    "system": "#4da3ff", "task": "#7ee787", "project_doc": "#a5d6ff",
    "code": "#ffa657", "tool_def": "#d2a8ff", "tool_result": "#79c0ff",
    "memory": "#f0883e", "retrieval": "#56d4dd", "data": "#8b949e",
    "user_message": "#7ee787", "assistant_message": "#d2a8ff", "example": "#e3b341",
    "other": "#6e7681",
}


def esc(text: object) -> str:
    return _escape(str(text), quote=True)


def page(title: str, subtitle: str, body: str, *, footer: str = "") -> str:
    """Wrap body fragments into a complete, self-contained HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{esc(title)}</title>
<style>{BASE_CSS}</style></head>
<body>
<h1>{esc(title)}</h1>
<p class="sub">{esc(subtitle)}</p>
{body}
<footer>{footer or "Generated locally by ctxbudgeter — no data left this machine. Secrets are masked."}</footer>
</body></html>"""


def card(label: str, value: object, foot: str = "") -> str:
    foot_html = f'<div class="foot">{esc(foot)}</div>' if foot else ""
    return (
        f'<div class="card"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div>{foot_html}</div>'
    )


def cards(items: list[tuple[str, object, str]]) -> str:
    return '<div class="cards">' + "".join(card(*i) for i in items) + "</div>"


def panel(title: str, inner: str) -> str:
    return f'<div class="panel"><h2>{esc(title)}</h2>{inner}</div>'


def bar_row(label: str, value: int, max_value: int, color: str, *, suffix: str = "") -> str:
    pct = (value / max_value * 100) if max_value else 0
    width = max(2.0, min(100.0, pct))
    return (
        f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
        f'<div style="width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{esc(label)}</div>'
        f'<div style="flex:1"><div class="bar" style="width:{width:.1f}%;background:{color}">'
        f'{esc(value)}{esc(suffix)}</div></div></div>'
    )

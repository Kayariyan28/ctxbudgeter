"""Record the 0.4 terminal demo as an animated SVG — from real command output.

    python docs/demo/make_terminal_svg.py            # writes docs/ctxbudgeter-0.4-demo.svg

Every step below is executed for real (in a temporary directory, with the installed
`ctxbudgeter` CLI); its stdout/stderr is captured, ANSI codes are stripped, and the
result is laid out as an SVG terminal whose commands type out and whose output lines
appear on a timeline, scrolling like a real session. Pure CSS animation, so it plays
inside a GitHub README <img>. Colours and type follow the K28 D2 design system
(IBM Plex Mono embedded from docs/banner/assets/fonts).
"""

from __future__ import annotations

import base64
import html
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "docs" / "ctxbudgeter-0.4-demo.svg"
FONT = REPO / "docs" / "banner" / "assets" / "fonts" / "ibm-plex-mono-latin-400-normal.woff2"

# (narration, shell command, max output lines)
STEPS: list[tuple[str, str, int]] = [
    ("# ctxbudgeter 0.4 — the Adaptive Context Engine", "ctxbudgeter --version", 3),
    ("# a governed context API: plan, compact, deny protected edits, recall, evaluate",
     "python examples/context_runtime_demo.py", 16),
    ("# CTXBench: harness vs unrestricted vs guided vs adaptive (mode | CIS | accuracy | critical)",
     "ctxbudgeter bench -a reckless -n 1 --families research,coding --budgets 4000 "
     "-o report.json --traces traces.jsonl | sed -n 5,8p | cut -d'|' -f2,4,5,9", 6),
    ("# graduated autonomy: promote / hold / demote from benchmark evidence",
     "ctxbudgeter autonomy certify report.json --registry autonomy.json", 4),
    ("# learn a per-model policy from the agent's own decision traces",
     "ctxbudgeter profile traces.jsonl --store policies", 8),
    ("# turn decisions into SFT, DPO and reward datasets",
     "ctxbudgeter train-export traces.jsonl -o datasets > /dev/null && wc -l datasets/*.jsonl", 6),
    ("# install the portable context-management skill (Claude Code or any SKILL.md loader)",
     "ctxbudgeter skill export ./skills && find skills -type f | sort", 6),
    ("# the same governed API for any MCP client",
     "echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}' | ctxbudgeter serve-mcp "
     "| python -c \"import json,sys,textwrap; print(textwrap.fill(' '.join(t['name'] for t in "
     "json.load(sys.stdin)['result']['tools']), 96))\"", 4),
]

COLS = 104
LH = 22  # line height (px)
CW = 8.4  # char width at 14px Plex Mono
PAD_X, PAD_TOP = 24, 52
VIEW_LINES = 26
W = int(PAD_X * 2 + COLS * CW)
H = PAD_TOP + VIEW_LINES * LH + 18
BG, FG, DIM, LIME, GREY = "#0E0E0E", "#F5F5F2", "#6A6A66", "#C9E35B", "#9A9A96"
TYPE_CPS = 45  # typing speed, chars/second
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def run(cmd: str, cwd: pathlib.Path, env: dict[str, str]) -> list[str]:
    p = subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    text = ANSI.sub("", (p.stdout or "") + (p.stderr or ""))
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def clip(s: str, n: int = COLS) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    if shutil.which("ctxbudgeter") is None:
        print("ctxbudgeter is not on PATH; activate the environment it is installed in")
        return 1
    work = pathlib.Path(tempfile.mkdtemp(prefix="ctx-demo-"))
    shutil.copytree(REPO / "examples", work / "examples")
    env = {**os.environ, "COLUMNS": str(COLS), "TERM": "dumb", "NO_COLOR": "1"}

    # ---- execute and lay out -------------------------------------------------------
    rows: list[tuple[str, str, float, float]] = []  # (kind, text, t_start, type_duration)
    t = 0.6
    for note, cmd, cap in STEPS:
        out = run(cmd, work, env)
        rows.append(("note", note, t, 0.0))
        t += 0.5
        shown = clip("$ " + cmd, 400)
        # long commands wrap onto continuation lines
        chunks = [shown[i:i + COLS] for i in range(0, len(shown), COLS)]
        for k, ch in enumerate(chunks):
            dur = len(ch) / TYPE_CPS
            rows.append(("cmd" if k == 0 else "cmdc", ch, t, dur))
            t += dur
        t += 0.45
        if len(out) > cap:
            out = [*out[: cap - 1], f"… ({len(out) - cap + 1} more lines)"]
        for ln in out:
            rows.append(("out", clip(ln), t, 0.0))
            t += 0.07
        rows.append(("blank", "", t, 0.0))
        t += 1.6
    total = t + 2.5
    shutil.rmtree(work, ignore_errors=True)

    # ---- SVG -------------------------------------------------------------------------
    font = base64.b64encode(FONT.read_bytes()).decode()

    def pct(x: float) -> str:
        return f"{100 * x / total:.3f}%"

    css = [
        f"@font-face{{font-family:PMono;src:url(data:font/woff2;base64,{font}) format('woff2')}}",
        f"text{{font-family:PMono,ui-monospace,Menlo,monospace;font-size:14px;fill:{FG};white-space:pre}}",
        f".note{{fill:{DIM}}}.out{{fill:{GREY}}}.cmd{{fill:{FG}}}.p{{fill:{LIME}}}",
        f"g.l{{opacity:0;animation-duration:{total:.2f}s;animation-iteration-count:infinite;animation-timing-function:step-end}}",
        f"rect.cv{{animation-duration:{total:.2f}s;animation-iteration-count:infinite;animation-timing-function:linear}}",
        f"#scroll{{animation:sc {total:.2f}s infinite;animation-timing-function:ease-out}}",
    ]
    body: list[str] = []
    scroll_keys: list[tuple[float, int]] = [(0.0, 0)]
    for i, (kind, text, t0, dur) in enumerate(rows):
        y = PAD_TOP + i * LH
        off = max(0, (i + 1) - VIEW_LINES) * LH
        if off != scroll_keys[-1][1]:
            scroll_keys.append((t0, off))
        css.append(f"@keyframes l{i}{{0%{{opacity:0}}{pct(t0)}{{opacity:1}}99.9%{{opacity:1}}100%{{opacity:0}}}}")
        esc = html.escape(text)
        if kind == "cmd":
            esc = f'<tspan class="p">$</tspan>{html.escape(text[1:])}'
        cls = {"note": "note", "out": "out", "cmd": "cmd", "cmdc": "cmd", "blank": "out"}[kind]
        inner = f'<text x="{PAD_X}" y="{y}" class="{cls}">{esc}</text>'
        if dur > 0:  # typing: a background-coloured cover slides off the text
            width = len(text) * CW + 4
            css.append(
                f"@keyframes c{i}{{0%{{transform:translateX(0)}}{pct(t0)}{{transform:translateX(0)}}"
                f"{pct(t0 + dur)}{{transform:translateX({width:.0f}px)}}100%{{transform:translateX({width:.0f}px)}}}}")
            inner += (f'<rect class="cv" x="{PAD_X - 2}" y="{y - 15}" width="{width:.0f}" height="{LH}" fill="{BG}" '
                      f'style="animation-name:c{i}"/>')
        body.append(f'<g class="l" style="animation-name:l{i}">{inner}</g>')

    frames = []
    for k, (t0, off) in enumerate(scroll_keys):
        frames.append(f"{pct(t0)}{{transform:translateY(-{off}px)}}")
        nxt = scroll_keys[k + 1][0] if k + 1 < len(scroll_keys) else total * 0.999
        frames.append(f"{pct(max(t0, nxt - 0.01))}{{transform:translateY(-{off}px)}}")
    css.append("@keyframes sc{" + "".join(frames) + "100%{transform:translateY(0)}}")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="ctxbudgeter 0.4 terminal demo">
<title>ctxbudgeter 0.4 terminal demo — context API, CTXBench, autonomy, profiling, datasets, skill, MCP</title>
<style>{"".join(css)}</style>
<rect width="{W}" height="{H}" rx="20" fill="{BG}"/>
<circle cx="26" cy="22" r="6" fill="#3A3A3A"/><circle cx="46" cy="22" r="6" fill="#3A3A3A"/><circle cx="66" cy="22" r="6" fill="#3A3A3A"/>
<text x="{W / 2:.0f}" y="27" text-anchor="middle" class="note" style="font-size:12px;letter-spacing:2px">CTXBUDGETER 0.4 · ADAPTIVE CONTEXT ENGINE</text>
<clipPath id="vp"><rect x="0" y="{PAD_TOP - 18}" width="{W}" height="{VIEW_LINES * LH + 8}"/></clipPath>
<g clip-path="url(#vp)"><g id="scroll">{"".join(body)}</g></g>
</svg>'''
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT} ({len(svg) // 1024} KB, {len(rows)} lines, {total:.1f}s loop)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

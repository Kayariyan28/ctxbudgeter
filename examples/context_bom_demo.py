"""Generate a Context Bill of Materials as JSON and Markdown.

Run:  python examples/context_bom_demo.py
Writes examples/out/context_bom.json and context_bom.md.
"""

from __future__ import annotations

from pathlib import Path

from ctxbudgeter import ContextPack, ContextPolicy


def build():
    policy = ContextPolicy(max_tokens=24_000, reserved_output_tokens=4_000, block_secrets=True)
    pack = ContextPack(model="claude-sonnet-4.6", policy=policy)
    pack.add(name="system", content="You are a careful coding agent.", kind="system",
             priority=100, required=True, cache_policy="stable",
             source="repo/system.md", trust_level="verified")
    pack.add(name="style_guide", content="Follow PEP 8. Prefer dataclasses. " * 20,
             kind="project_doc", priority=85, cache_policy="stable",
             source="docs/STYLE_GUIDE.md", trust_level="internal")
    pack.add(name="diff", content="def add(a, b):\n    return a + b\n", kind="task",
             priority=95, required=True)
    return pack.compile(task="Review this diff")


def main() -> None:
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    bom = build().bom
    bom.to_json(str(out / "context_bom.json"))
    bom.to_markdown(str(out / "context_bom.md"))
    print(f"Wrote {out/'context_bom.json'} and {out/'context_bom.md'}")
    print(f"\nHealth: {bom.context_health_score}/100  Risk: {bom.risk_score}/100  "
          f"Cache efficiency: {bom.cache_efficiency_score}/100")


if __name__ == "__main__":
    main()

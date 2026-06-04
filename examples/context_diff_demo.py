"""Compare two compiled contexts and print the diff.

Run:  python examples/context_diff_demo.py
"""

from __future__ import annotations

from ctxbudgeter import ContextDiff, ContextPack


def _pack(extra: bool):
    p = ContextPack(model="claude-sonnet-4.6", token_budget=24_000, reserved_output_tokens=4_000)
    p.add(name="system", content="You are a careful agent.", kind="system",
          required=True, cache_policy="stable", source="system.md", trust_level="verified")
    p.add(name="task", content="Answer the refund question.", kind="task", required=True)
    if extra:
        p.add(name="huge_legacy_doc", content="legacy notes " * 200, priority=40,
              source="docs/legacy.md", trust_level="low")
    return p.compile()


def main() -> None:
    before = _pack(extra=False).bom
    after = _pack(extra=True).bom
    diff = ContextDiff.compare(before, after)
    print(diff.to_text())
    print("\nMarkdown:\n")
    print(diff.to_markdown())


if __name__ == "__main__":
    main()

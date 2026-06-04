"""Generate a Context MRI HTML report.

Run:  python examples/context_mri_demo.py
Writes examples/out/context_mri.html (open it in a browser).
"""

from __future__ import annotations

from pathlib import Path

from ctxbudgeter import ContextPack, ContextPolicy
from ctxbudgeter.viz import ContextMRI


def main() -> None:
    policy = ContextPolicy(max_tokens=24_000, reserved_output_tokens=4_000,
                           block_secrets=True, redact_sensitive=True)
    pack = ContextPack(model="claude-sonnet-4.6", policy=policy)
    pack.add(name="system", content="You are a careful support agent.", kind="system",
             priority=100, required=True, cache_policy="stable",
             source="repo/system.md", trust_level="verified")
    pack.add(name="refund_policy", content="Refunds within 30 days. " * 30,
             kind="project_doc", priority=85, cache_policy="stable",
             source="docs/refund_policy.md", trust_level="verified")
    pack.add(name="customer_email", content="Hi, I'd like a refund. My email is jane@example.com.",
             kind="user_message", priority=70, source="tickets/9001.txt", trust_level="low")
    pack.add(name="leaky_note", content="internal token sk-LIVE1234567890ABCDEFGH (do not share)",
             kind="data", priority=40, source="notes/scratch.txt", trust_level="low")
    pack.add(name="task", content="Resolve the refund request.", kind="task", priority=95, required=True)

    compiled = pack.compile(task="Resolve refund request")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    target = out / "context_mri.html"
    ContextMRI.from_compiled(compiled).export_html(str(target))
    print(f"Context MRI written to {target}")
    print("Open it in a browser — fully self-contained, no network, secrets masked.")


if __name__ == "__main__":
    main()

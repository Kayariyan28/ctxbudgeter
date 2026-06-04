"""Governance demo: a policy that blocks secrets, denies forbidden sources, and
requires provenance — then compiles a pack with safe and risky context.

Run:  python examples/context_policy_demo.py
"""

from __future__ import annotations

from ctxbudgeter import ContextPack, ContextPolicy


def main() -> None:
    policy = ContextPolicy(
        max_tokens=24_000,
        reserved_output_tokens=4_000,
        block_secrets=True,
        require_provenance=True,
        allowed_sources=["docs", "tickets", "repo"],
        forbidden_sources=[".env", "payroll", "private_keys"],
        redact_sensitive=True,
        fail_on_policy_violation=False,
    )

    pack = ContextPack(model="claude-sonnet-4.6", policy=policy)
    pack.add(
        name="system", content="You are a careful support agent. Be concise.",
        kind="system", priority=100, required=True, cache_policy="stable",
        source="repo/prompts/system.md", trust_level="verified",
    )
    pack.add(
        name="refund_policy", content="Refunds are available within 30 days of purchase.",
        kind="project_doc", priority=85, cache_policy="stable",
        source="docs/refund_policy.md", trust_level="verified",
    )
    # Risky: contains a secret → will be redacted by policy.
    pack.add(
        name="ticket_4231", content="Customer pasted their key sk-LIVE1234567890ABCDEFGH oops.",
        kind="data", priority=60, source="tickets/4231.txt", trust_level="low",
    )
    # Forbidden source → excluded by policy.
    pack.add(
        name="env_dump", content="DB_PASSWORD=hunter2", kind="data", priority=50, source="config/.env",
    )
    pack.add(name="task", content="Resolve the refund request in ticket 4231.",
             kind="task", priority=95, required=True)

    compiled = pack.compile(task="Resolve refund request in ticket 4231")

    print(compiled.report("text"))
    print("\nPolicy violations:")
    for v in compiled.policy_violations:
        print(f"  [{v['severity']}] {v['rule']} -> {v['action']} ({v.get('item_name')})")
    print("\nRedacted ticket content (no secret leaks):")
    if "ticket_4231" in [i.name for i in compiled.included_items]:
        print("  ", compiled.content_for("ticket_4231"))


if __name__ == "__main__":
    main()

# ContextOps with ctxbudgeter

**ctxbudgeter is not an agent framework. It works *before* the model call.**

Agent observability tools show what the agent *did*. ctxbudgeter shows what the
agent was *allowed to know* before it acted — and lets you govern, audit, and
optimize that context deterministically.

A ContextOps pipeline has five stages, all local-first and LLM-free:

| Stage | What it does | API |
|-------|--------------|-----|
| **Compile** | Select items under a token budget | `ContextPack.compile(task=...)` |
| **Govern** | Enforce a policy (PII/secrets, sources, provenance, age) | `ContextPolicy`, `enforce_policy` |
| **Audit** | Produce a Bill of Materials | `compiled.bom` → `ContextBOM` |
| **Visualize** | Render a Context MRI | `ctxbudgeter.viz.ContextMRI` |
| **Optimize** | Cache layout + waste + MCP budgeting | `CachePlanner`, `MCPToolBudgeter` |

## Minimal pipeline

```python
from ctxbudgeter import ContextPack, ContextPolicy

policy = ContextPolicy(
    max_tokens=24_000,
    reserved_output_tokens=4_000,
    block_secrets=True,
    forbidden_sources=[".env", "payroll"],
    redact_sensitive=True,
)

pack = ContextPack(model="claude-sonnet-4.6", policy=policy)
pack.add(name="system", content="You are a careful agent.", kind="system",
         required=True, cache_policy="stable", source="repo/system.md", trust_level="verified")
pack.add(name="task", content="Resolve the refund request.", kind="task", required=True)

compiled = pack.compile(task="Resolve refund request")
bom = compiled.bom                          # auditable artifact
print(compiled.report("text"))             # what entered, what didn't, and why
```

When a policy is attached, `compile()` automatically:

1. scans every included item for PII/secrets (local regex engine),
2. redacts or excludes per the policy,
3. records `policy_violations`, `scanner_findings`, and per-item `item_risk`,
4. recomputes token totals and the cacheable prefix.

## CI gate

```bash
ctxbudgeter compile . --task "fix auth bug" --policy ctxbudgeter.yaml --bom context_bom.json
ctxbudgeter eval context_evals.yaml --bom context_bom.json   # exits non-zero on failure
```

See also: [context_bom.md](context_bom.md), [context_mri.md](context_mri.md),
[mcp_tool_budgeting.md](mcp_tool_budgeting.md), [security.md](security.md).

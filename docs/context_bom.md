# Context Bill of Materials (BOM)

A `ContextBOM` is a structured, auditable record of exactly what was compiled into
a model call. It's the artifact you commit, diff, and gate on in CI.

```python
compiled = pack.compile(task="Resolve refund request")
bom = compiled.bom

bom.to_json("context_bom.json")     # deterministic (sorted keys, no wall-clock)
bom.to_markdown("context_bom.md")   # human-readable for PR review
```

Reload and diff later:

```python
from ctxbudgeter import ContextBOM, ContextDiff

old = ContextBOM.from_json("main_bom.json")
new = compiled.bom
diff = ContextDiff.compare(old, new)
print(diff.to_text())               # token / risk / cache / health deltas
```

## What's in it

| Field | Meaning |
|-------|---------|
| `run_id` | deterministic hash of model + task + included items (stable across identical compiles) |
| `total_tokens`, `available_context_tokens`, `cacheable_tokens` | budget accounting |
| `cache_efficiency_score`, `context_health_score`, `risk_score` | 0–100 signals |
| `included_items` | name, kind, tokens, priority, cache policy, relevance, freshness, **trust**, **risk**, source, transformations, checksum |
| `excluded_items` | name, kind, tokens, priority, **reason** |
| `compressed_items`, `redacted_items` | names of items transformed by the compiler/policy |
| `policy_summary`, `policy_violations` | the governing policy and any breaches |
| `scanner_findings` | masked PII/secret findings (never full secrets) |
| `recommendations` | actionable suggestions (compress, exclude, move_to_stable_prefix, …) |
| `provenance_summary` | trust-level distribution |

## Determinism

`to_json()` uses `sort_keys=True` and never embeds a wall-clock timestamp unless you
pass `created_at`. Identical inputs produce byte-identical JSON — safe for golden-file
CI diffing.

## CLI

```bash
ctxbudgeter compile . --task "..." --bom context_bom.json     # write a BOM
ctxbudgeter bom context_bom.json --format markdown            # re-render
ctxbudgeter diff old_bom.json new_bom.json --fail-on-risk-increase
```

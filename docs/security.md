# Security principles

ctxbudgeter is **local-first** and makes **no LLM API calls and no network calls**
in its core. The scanner, policy engine, BOM, diff, cache planner, and visualizations
all run offline and deterministically.

## Secret masking — the core invariant

The `ContextScanner` has a single masking choke-point. **No full secret ever leaves
the scanner** — not in a finding preview, not in `masked_text`, not in a BOM, not in a
report, not in any HTML. Previews look like:

```
sk-...Ab3        ghp_...9kL        AKIA...7QZ        -----BEGIN PRIVATE KEY----- [masked]
```

`redact(text)` replaces detected secrets/PII with `[REDACTED:<category>]` tokens.
This invariant is covered by tests (see `tests/test_scanner.py`,
`test_bom_never_leaks_secret`, `test_mri_html_no_secret_leak`).

## Policy enforcement

A `ContextPolicy` can `block_secrets` / `block_pii` and either **redact** (replace
content in place) or **exclude** (drop the item). In production, prefer
`redact_sensitive=True`, and set `fail_on_policy_violation=True` in CI to hard-fail on
error/critical breaches.

## Honest limitations

- **The scanner is a regex/heuristic engine.** It catches common shapes (provider key
  formats, JWTs, PEM blocks, `.env` assignments, Luhn-valid cards). It is **not
  exhaustive** and **not a compliance certification**. Pair it with a dedicated secret
  scanner (gitleaks, trufflehog) and your own DLP for regulated data.
- **Provenance is declarative.** Trust levels and timestamps are whatever the caller
  supplies; ctxbudgeter doesn't verify them.
- **Cache figures are estimates.** We report *estimated cacheable tokens* only — never
  a dollar amount — because cache economics are provider-specific.
- **Context MRI's Influence Proxy is not model attention** (see context_mri.md).

## Untrusted inputs

`ContextPolicy.from_yaml` uses `yaml.safe_load`. Pack specs and policies are config —
review them like any other code. Loaders you register run with your process's
privileges; treat them like dependencies.

## Reporting

Found a way to make the scanner leak a secret, or another security issue? Please open
a private report (see `SECURITY.md` at the repo root).

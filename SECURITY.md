# Security policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.2.x   | ✅        |
| 0.1.x   | ❌ (please upgrade) |

## Reporting a vulnerability

If you discover a security issue in ctxbudgeter, please **do not** open a public GitHub issue.

Instead, email **karandey3@outlook.com** with:

- A description of the issue
- Steps to reproduce
- Affected versions
- Suggested mitigation (if any)

You should receive a response within 5 business days. We aim to release a patch within 30 days for confirmed high-severity issues.

## Threat model

ctxbudgeter is a **local-first** library. The core never calls an external API. Threats we care about:

- **Untrusted YAML pack specs.** `load_pack()` uses `yaml.safe_load`. Don't load specs from untrusted sources without review.
- **Untrusted loaders.** Custom loaders run with the same privileges as your process. Treat loader code like any other dependency.
- **Sensitivity leakage.** Items tagged `sensitivity="secret"` are flagged in reports and can be enforced via `secret_policy`. In production, use `refuse` or `redact` — never `allow`. The default is `warn`, which surfaces the leak but does not prevent it.
- **Prompt injection** is **not** addressed by ctxbudgeter. We pack and ship the content you give us; we don't sanitize for injection. Pair this library with a separate input-validation layer for user-supplied content.

## What we are NOT responsible for

- Vulnerabilities in optional dependencies (`anthropic`, `openai`, `langchain-core`, `requests`, `tiktoken`, `PyYAML`). Report those upstream.
- Misconfigured `secret_policy: allow` in production.
- Loader code you wrote.

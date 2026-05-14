# ctxbudgeter

[![PyPI](https://img.shields.io/pypi/v/ctxbudgeter.svg)](https://pypi.org/project/ctxbudgeter/)
[![Python](https://img.shields.io/pypi/pyversions/ctxbudgeter.svg)](https://pypi.org/project/ctxbudgeter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Kayariyan28/ctxbudgeter/actions/workflows/test.yml/badge.svg)](https://github.com/Kayariyan28/ctxbudgeter/actions/workflows/test.yml)

> Compile clean, cheap, auditable context for AI agents.

`ctxbudgeter` is a framework-agnostic **context engineering toolkit** for agentic AI. Hand it the raw materials — system rules, docs, code, memory notes, tool results, the latest user request — and it decides what enters the model, what gets dropped, what gets compressed, what should be cached, and **why**. Every decision is auditable. Every compilation is deterministic. Tests can gate on context the same way they gate on code.

**Use any agent framework.** ctxbudgeter sits in front of LangGraph, CrewAI, OpenAI Agents SDK, PydanticAI, Microsoft Agent Framework, or your own loop. It does one thing: make the context you send to the model cleaner, cheaper, and assertable.

```text
Webpack for agent context  •  pytest for prompt/context quality  •  token budget manager
```

## What you get

- **Token budget compiler** — deterministic, explainable selection with full inclusion/exclusion reasons
- **Just-in-time `Reference`s** — lazy pointers (file paths, URLs, queries) that only load if they fit
- **Eval / assert layer + pytest plugin** — `assert_includes`, `assert_health_at_least`, golden snapshots
- **Cache-aware adapters** — Anthropic `cache_control` placement, OpenAI `prompt_cache_key`, LangChain & PydanticAI
- **Multi-modal attachments** — images and structured tool schemas flow through to OpenAI/Anthropic payloads
- **Sensitivity enforcement** — `allow` | `warn` | `refuse` | `redact` for items tagged `secret`
- **Memory store** (Write strategy) — persist agent notes between turns, query them back into context
- **Isolation** (Isolate strategy) — `pack.fork()` builds a subagent-scoped pack with its own budget
- **Async compile** — concurrent resolution of async References, async-aware compressor hook
- **Declarative YAML/JSON specs** — check pack configuration into git, CI-friendly
- **CLI** — `scan`, `compile`, `pack`, `validate`, `report` for Claude Code and CI workflows
- **Zero LLM calls in the core** — local-first, deterministic, fast

## Install

```bash
pip install ctxbudgeter

# Optional extras
pip install "ctxbudgeter[tiktoken]"        # accurate OpenAI/Anthropic-proxy tokenization
pip install "ctxbudgeter[yaml]"            # YAML pack specs
pip install "ctxbudgeter[http]"            # http_get loader for References
pip install "ctxbudgeter[anthropic,openai,langchain]"
pip install "ctxbudgeter[all]"             # everything
```

Python 3.10+. Adapters are lazy-imported — you only pay for the SDKs you actually use.

## Quick start

```python
from ctxbudgeter import ContextPack

pack = ContextPack(
    model="claude-sonnet-4.6",
    token_budget=24_000,
    reserved_output_tokens=4_000,
)

pack.add(
    name="system_rules",
    content="You are a careful coding agent...",
    kind="system",
    priority=100,
    cache_policy="stable",
    required=True,
)
pack.add_file("README.md", kind="project_doc", priority=80)
pack.add(
    name="task",
    content="Build the referral packet UI.",
    kind="task",
    priority=95,
    required=True,
)

compiled = pack.compile()
print(compiled.report())
```

```text
Included:
  - system_rules: 312 tokens, required, stable cache prefix, system
  - README.md: 1,420 tokens, stable cache prefix, project_doc
  - task: 19 tokens, required, task

Excluded:
  - old_notes.md: token-heavy and low priority — 8,400 tokens, score 41
  - debug.log: token-heavy and low priority — 14,200 tokens, score 12

Estimated input tokens: 1,751
Reserved output tokens: 4,000
Cacheable prefix: 1,732 tokens
Token budget: 24,000 (utilization 8.8%)
Context health score: 87/100
  breakdown: cacheable_prefix_bonus: +5, under_utilized: -5
Tokenizer: tiktoken
```

## Just-in-time References

Don't load context you'll never use. References are lightweight pointers that load only when they could plausibly fit the budget — Anthropic's "JIT" pattern, built in.

```python
from ctxbudgeter import ContextPack
from ctxbudgeter.loaders import file_loader, http_get_loader, register_loader

pack = ContextPack(token_budget=24_000)
pack.add(name="task", content="Refactor auth", kind="task", required=True)

# File reference — only opened if it would fit
pack.add_reference(
    name="auth_module",
    location="src/auth.py",
    loader=file_loader,
    estimated_tokens=1200,
    kind="code",
    priority=70,
)

# HTTP reference — never fetched unless budget allows
pack.add_reference(
    name="api_docs",
    location="https://example.com/docs/api.json",
    loader=http_get_loader,
    estimated_tokens=2000,
    kind="retrieval",
    priority=60,
)

# Or register your own loader
@register_loader("vector_search")
def vector_search(ref):
    return my_vector_store.search(ref.location, k=3)

pack.add_reference(name="docs_hit", location="referral packet UI", loader=vector_search, estimated_tokens=500)

compiled = pack.compile()
```

Async loaders work too — use `await pack.acompile()`:

```python
async def fetch_user_profile(ref):
    async with httpx.AsyncClient() as c:
        r = await c.get(ref.location)
        return r.text

pack.add_reference(name="profile", location="https://api.example.com/me", loader=fetch_user_profile, estimated_tokens=300)
compiled = await pack.acompile()   # async references resolved concurrently
```

## Eval / assert layer — "pytest for prompts"

```python
from ctxbudgeter.testing import (
    assert_includes, assert_excludes,
    assert_health_at_least, assert_cacheable_prefix_at_least,
    assert_no_secret_items, assert_used_tokens_at_most,
    GoldenPack,
)

def test_prod_pack():
    compiled = build_prod_pack().compile()
    assert_includes(compiled, "system_rules", "task")
    assert_excludes(compiled, "debug.log")
    assert_health_at_least(compiled, 80)
    assert_cacheable_prefix_at_least(compiled, 1024)
    assert_no_secret_items(compiled)
    assert_used_tokens_at_most(compiled, 20_000)

def test_pack_golden(ctxbudgeter_golden):
    # Provided by the installed pytest plugin.
    # Stores a golden snapshot the first time, diffs against it after.
    ctxbudgeter_golden().check(build_prod_pack().compile())
```

Refresh goldens after intentional changes:

```bash
pytest --ctxbudgeter-update-golden
```

## Cache-aware adapters

```python
from ctxbudgeter.adapters import (
    to_anthropic_request,  # cache_control on last stable system block
    to_openai_request,     # prompt_cache_key derived from stable prefix hash
    to_langchain_messages,
    to_pydantic_ai_deps,
)

# Anthropic
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(**to_anthropic_request(compiled, user_message="next step?"))

# OpenAI — explicit cache key for prompt-prefix caching
from openai import OpenAI
oa = OpenAI()
resp = oa.chat.completions.create(**to_openai_request(compiled, user_message="what now?"))

# LangChain
from langchain_anthropic import ChatAnthropic
msgs = to_langchain_messages(compiled, user_message="continue")
ChatAnthropic(model=compiled.model).invoke(msgs)

# PydanticAI
deps = to_pydantic_ai_deps(compiled)
agent.run(deps["system_prompt"], message_history=deps["message_history"])
```

## Multi-modal attachments

```python
from ctxbudgeter import ContextPack, ImageBlock, StructuredBlock

pack = ContextPack(token_budget=24_000)
pack.add(
    name="screenshot",
    content="Describe what's wrong in this screenshot.",
    kind="user_message",
    attachments=[
        ImageBlock(url="https://example.com/bug.png", estimated_tokens=400),
    ],
)
pack.add(
    name="tools",
    content="",
    kind="tool_def",
    cache_policy="stable",
    priority=85,
    attachments=[
        StructuredBlock(schema_name="search_db", data={"args": ["query"], "returns": "list[Doc]"}),
    ],
)
```

Image and structured blocks flow through to OpenAI's `image_url` / Anthropic's `image` / `tool_result` formats automatically.

## Sensitivity enforcement

```python
pack.add(name="api_key", content="sk-DEADBEEF...", sensitivity="secret")

pack.set_secret_policy("warn")     # include but flag in report + health penalty (default)
pack.set_secret_policy("refuse")   # raise SecretContentError at compile time
pack.set_secret_policy("redact")   # replace content with [REDACTED — sensitivity=secret]
pack.set_secret_policy("allow")    # silently allow (escape hatch)
```

In CI you almost always want `refuse` or `redact`. The text/markdown reports flag `[!secret]` items so reviewers can catch leaks during PR review.

## Memory (Write strategy) + Isolation (Isolate strategy)

```python
from ctxbudgeter import ContextPack, InMemoryStore, JSONMemoryStore, MemoryNote

# Persist notes across turns
store = JSONMemoryStore(".ctxbudgeter/memory.json")
store.write(MemoryNote(key="auth_runbook", content="JWT rotation...", tags=["auth"]))

# Pull them back into a future pack
pack = ContextPack(token_budget=24_000)
pack.add(name="task", content="Fix auth bug", required=True)
pack.add_memory(store, tags=["auth"], limit=3, priority=70)

# Isolate a subagent's context — only frontend code, smaller budget
frontend_pack = pack.subset_by_kind("project_doc", "code").fork(
    filter=lambda it: it.metadata.get("area") == "frontend",
    token_budget=8_000,
)
```

## Declarative YAML pack specs

Check your pack into git like any other config:

```yaml
# pack.yaml
model: claude-sonnet-4.6
token_budget: 24000
reserved_output_tokens: 4000
secret_policy: refuse

items:
  - name: system_rules
    from_file: prompts/system.md
    kind: system
    priority: 100
    required: true
    cache_policy: stable

  - name: task
    content: "Fix the auth bug."
    kind: task
    priority: 95
    required: true

references:
  - name: api_docs
    location: "https://example.com/docs.json"
    loader: http_get
    estimated_tokens: 1500
    priority: 60
```

Then compile from the CLI:

```bash
ctxbudgeter validate pack.yaml
ctxbudgeter pack pack.yaml --format markdown -o report.md
ctxbudgeter pack pack.yaml --fail-below 80    # exit non-zero on low health for CI
```

Or from Python:

```python
from ctxbudgeter.spec import load_pack
pack = load_pack("pack.yaml")
compiled = pack.compile()
```

## CLI

```bash
# Scan a directory, suggest priorities + cache policies
ctxbudgeter scan . --max-files 50

# Scan + emit a starter pack.yaml you can commit and iterate on
ctxbudgeter scan . --emit-pack pack.yaml --task "ship feature X"

# Ad-hoc compile from a directory + task
ctxbudgeter compile . --task "fix auth bug" --budget 12000 --secret-policy refuse

# Compile from a declarative spec
ctxbudgeter pack pack.yaml --format markdown -o context-report.md

# Re-render a saved compiled pack
ctxbudgeter compile . --task "..." --save-pack pack.json
ctxbudgeter report pack.json --format markdown
```

## Wire it into CI

### GitHub Actions

```yaml
# .github/workflows/context-check.yml
name: Context budget check
on: [pull_request]

jobs:
  ctxbudget:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install "ctxbudgeter[all]"
      - name: Validate and compile pack
        run: |
          ctxbudgeter validate pack.yaml
          ctxbudgeter pack pack.yaml --format markdown --fail-below 80 -o report.md
      - name: Comment report on PR
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: report.md
```

### pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ctxbudgeter-validate
        name: ctxbudgeter validate pack.yaml
        entry: ctxbudgeter validate pack.yaml
        language: system
        pass_filenames: false
        files: ^pack\.yaml$
```

### pytest

```python
# tests/test_context.py
from ctxbudgeter.testing import (
    assert_health_at_least, assert_no_secret_items, assert_includes,
)
from my_app.context import build_pack

def test_production_pack_quality():
    compiled = build_pack(task="fix auth bug").compile()
    assert_includes(compiled, "system_rules", "task")
    assert_health_at_least(compiled, 80)
    assert_no_secret_items(compiled)
```

## How it works

### Compiler algorithm

1. Resolve `Reference`s — load only those whose estimated cost could fit. Loader failures → excluded with reason.
2. Required items go in first; compress (via your hook) or truncate if they don't fit.
3. Optional items ranked by `score_item`, packed greedily.
4. Sensitivity policy applied (warn / refuse / redact / allow).
5. Final prompt order: stable → dynamic → ephemeral, deterministic tie-breaks.
6. Cacheable prefix = consecutive stable items at the top.

```text
score = priority*0.5 + relevance*100*0.3 + freshness*100*0.1 + cache_value*100*0.1 - token_cost_penalty
```

`token_cost_penalty` grows up to ~30 points as an item approaches the full budget — a 50k-token debug log doesn't beat your README on priority alone.

### Health score breakdown

A 0–100 score with explicit, auditable deductions. Each pack reports its breakdown:

```json
{
  "health_score": 87,
  "health_breakdown": {
    "cacheable_prefix_bonus": 5,
    "under_utilized": -5,
    "high_priority_excluded": -10,
    "secrets_included": -10
  }
}
```

If you'd rather not show "health" — treat it as `BudgetCheckScore`: a determinstic, explainable signal, not a quality oracle.

### Compression hook

`ctxbudgeter` never calls an LLM for you. Provide a function — sync or async, your choice:

```python
async def my_summarizer(item, target_tokens):
    return await anthropic_client.summarize(item.content, max_tokens=target_tokens)

pack.set_compressor(my_summarizer)
compiled = await pack.acompile()   # async path; sync compressors also work with .compile()
```

If your compressor returns content larger than `target_tokens`, the compiler retries once with a tighter target before giving up.

## Round-trip + tooling

```python
import json
from ctxbudgeter import compiled_pack_from_dict

# Compile, save, share
compiled = pack.compile()
Path("compiled.json").write_text(json.dumps(compiled.to_dict()))

# Reload later for reporting / diffing / assertions
restored = compiled_pack_from_dict(json.loads(Path("compiled.json").read_text()))
print(restored.report("markdown"))
```

## Positioning

Most agent frameworks ask: *"which agent runs next?"*
ctxbudgeter asks: *"what exact information should this agent see right now — and why?"*

| Layer            | Existing                                  | What ctxbudgeter adds                              |
| ---------------- | ----------------------------------------- | -------------------------------------------------- |
| Agent frameworks | LangGraph, CrewAI, OpenAI Agents SDK, PydanticAI | Decides the **context shape** before the call |
| RAG              | LlamaIndex, LangChain retrievers          | Retrieval ≠ final context; ctxbudgeter is the gate |
| Observability    | LangSmith, AgentOps                       | They show what happened *after*; we prevent *before* |
| Context tools    | ctxforge, contextkit, contextagent        | We're the **assertable** + **deterministic** option  |

## Design choices

- **Local-first.** No LLM API calls in the core. The compiler is pure Python.
- **Deterministic.** Same inputs → identical compiled pack. Same JSON output. Same health score. Always.
- **Explainable.** Every input item shows up in `decisions` with a status and a human-readable reason.
- **Framework-agnostic.** Core has zero hard dependencies on agent SDKs. Adapters are lazy-imported.
- **Composable.** Bring your own tokenizer, your own compressor, your own scoring weights, your own loaders, your own memory store.
- **Assertable.** Quality gates live in `pytest`, not in your head.

## Author

**Karan Chandra Dey** — `[K28]`
Founder and AI Product Builder @ **K28 Design Lab** · [k28art.space](https://k28art.space)

Helping SMEs ship their first AI MVP — from prompt engineering to context engineering to production-ready agents.

| | |
|---|---|
| Web | [k28art.space](https://k28art.space) |
| GitHub | [@Kayariyan28](https://github.com/Kayariyan28) |
| LinkedIn | [karan-chandra-dey](https://www.linkedin.com/in/karan-chandra-dey-23392b1b9) |
| Email | karandey3@outlook.com |

> "Use any agent framework. ctxbudgeter makes your context cleaner, cheaper, and assertable — before the model sees it."

## License

MIT © Karan Chandra Dey / K28 Design Lab.

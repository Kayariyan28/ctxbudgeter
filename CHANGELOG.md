# Changelog

All notable changes to `ctxbudgeter` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-15

The "context engineering toolkit" release. Closes every gap from the v0.1 spec audit.

### Added

- **Eval / assert layer** (`ctxbudgeter.testing`) — `assert_includes`, `assert_excludes`, `assert_health_at_least`, `assert_cacheable_prefix_at_least`, `assert_no_secret_items`, `assert_no_compression_of`, `assert_utilization_between`, `assert_used_tokens_at_most`, `assert_includes_in_order`, `assert_no_loader_failures`, `assert_warnings_empty`, `GoldenPack`.
- **pytest plugin** auto-loaded via `pytest11` entry point — `ctxbudgeter_golden` fixture + `--ctxbudgeter-update-golden` flag for snapshot refresh.
- **Just-in-time `Reference`s** — lazy pointers with sync/async loaders that only resolve if their estimated cost fits the budget. Async references resolve concurrently in `acompile()`.
- **Built-in loaders** — `file`, `glob`, `env`, `inline`, `http_get` — and a `register_loader()` decorator for plugging in your own.
- **Memory store** (`MemoryStore`, `InMemoryStore`, `JSONMemoryStore`, `MemoryNote`) — the LangChain "write" strategy. `pack.add_memory(store, query=..., tags=..., limit=...)` queries notes and folds them into the pack.
- **Isolation** — `pack.fork(filter=...)`, `pack.subset_by_kind(...)`, `pack.subset_by_namespace(...)` for scoped subagent contexts with independent budgets.
- **Multi-modal attachments** — `TextBlock`, `ImageBlock`, `StructuredBlock`. Token cost included in the budget; OpenAI/Anthropic adapters emit them as `image_url` / `image` blocks.
- **Sensitivity enforcement** — `secret_policy: allow | warn | refuse | redact`. Reports flag `[!secret]` items. `SecretContentError` raised in refuse mode.
- **Health breakdown** — `CompiledPack.health_breakdown: dict[str, int]` itemizes every deduction (`cacheable_prefix_bonus: +5`, `under_utilized: -5`, `secrets_included: -10`, …).
- **Async compile** — `pack.acompile()` resolves async References concurrently and supports async compressors.
- **Compressor retry** — if your compressor overshoots `target_tokens`, the compiler retries once with `target * 0.6`.
- **Declarative pack specs** — YAML / TOML / JSON via `ctxbudgeter.spec.load_pack()`. Validated keys, `from_file` inlining, named loaders for safety.
- **CLI commands** — `ctxbudgeter pack`, `ctxbudgeter validate`, `ctxbudgeter scan --emit-pack`. `pack --fail-below N` for CI gating.
- **OpenAI cache awareness** — `stable_prefix_cache_key()` + `to_openai_request()` set `prompt_cache_key` deterministically from a SHA256 of the stable prefix.
- **`compiled_pack_from_dict()`** — round-trip a `CompiledPack` from its `to_dict()` JSON form.
- **Anthropic `to_anthropic_request()`** — full kwargs dict (model, max_tokens, system, messages) for `client.messages.create(**kwargs)`.

### Changed

- **Exclusion reasons** are now human-readable ("token-heavy and low priority — 8,400 tokens, score 41") instead of mechanical ("token cost X exceeds remaining Y").
- **`to_pydantic_ai`** renamed to **`to_pydantic_ai_deps`** to match the spec. The old name is kept as a deprecated alias.
- **`ContextPack.estimate_tokens()`** docstring clarified — it's the sum of all items, not the compiled subset. Use `preview()` or `compile()` for the post-compile count.
- **`CompiledPack.prompt`** kept as an alias for the new **`as_text()`** method.

### Fixed

- Chat turns (user/assistant/tool_result) now preserve **insertion order** instead of being shuffled by the cache-aware prompt sort. Anthropic's "messages must start with user" requirement is auto-corrected.

## [0.1.0] - 2026-05-14

Initial release. ContextPack + token-budget compiler + Anthropic/OpenAI/LangChain/PydanticAI adapters + CLI + 78 tests.

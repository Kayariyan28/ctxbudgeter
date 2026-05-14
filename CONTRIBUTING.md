# Contributing to ctxbudgeter

Thanks for considering a contribution. ctxbudgeter is small, deterministic, and tested — please keep it that way.

## Ground rules

1. **No LLM API calls in the core.** Compression, retrieval, and resolution are user-supplied hooks. The core must run offline with zero network access.
2. **Deterministic.** Same inputs → identical compiled pack, identical health score, identical decisions. No randomness, no time-based behavior in the core.
3. **Auditable.** Every decision the compiler makes shows up in `decisions` with a status and a human-readable reason. If you add a new code path, add a new reason.
4. **Framework-agnostic.** Adapters import their SDKs lazily inside the function that needs them. Don't make the core import `openai` / `anthropic` / `langchain_core`.

## Dev setup

```bash
git clone https://github.com/Kayariyan28/ctxbudgeter
cd ctxbudgeter
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,all]"
pytest -q
```

You should see 169+ tests pass.

## Run the suite

```bash
pytest                                  # all tests
pytest -k reference                     # filter
pytest --ctxbudgeter-update-golden      # refresh golden snapshots after intentional changes
ruff check src tests                    # lint
mypy src                                # types
```

## Submitting a change

1. Open an issue describing the bug / feature before writing code, especially for anything that touches the compiler, scoring, or public API.
2. Branch off `main`. Keep PRs focused.
3. Add tests. Every new feature gets a test file; every bug fix gets a regression test.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Update `README.md` if you added a public API or changed an existing one.

## Versioning

We follow SemVer:

- **Patch** (`0.2.x`) — bug fixes, doc fixes, internal refactors with no public API change.
- **Minor** (`0.x.0`) — backward-compatible additions (new methods, new fields with defaults, new adapters).
- **Major** (`x.0.0`) — breaking changes. The `0.x.0` series may include breaking changes within minors until 1.0.

## Release process (for maintainers)

1. Bump `version` in `pyproject.toml` and `__version__` in `src/ctxbudgeter/__init__.py`.
2. Move `[Unreleased]` to a dated heading in `CHANGELOG.md`.
3. Commit, tag `vX.Y.Z`, push tags.
4. `python -m build && twine check dist/*` locally to verify.
5. Create a GitHub Release from the tag — the `publish.yml` workflow uploads to PyPI via trusted publishing.

## Code style

- Black-compatible (we use Ruff). Line length 100.
- Type hints on all public functions.
- Docstrings on all public classes/functions. No multi-paragraph essays — explain the contract, edge cases, and a usage hint.
- No comments that restate the code. Comments explain **why**, not **what**.

## What we're unlikely to merge

- New LLM-calling code in the core.
- New randomness or wall-clock dependencies in the compiler.
- New required runtime dependencies (anything that isn't already in `[project.dependencies]`).
- Features that only make sense for a single agent framework — those belong in a new adapter, not in the core.

## Code of conduct

Be kind. We're trying to make a useful tool together.

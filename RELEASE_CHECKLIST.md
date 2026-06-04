# Release checklist

Steps to cut a new ctxbudgeter release. Trusted Publishing (OIDC) is configured —
no PyPI token is stored in the repo.

## 1. Pre-flight

- [ ] Update `version` in `pyproject.toml` and `__version__` in `src/ctxbudgeter/__init__.py` (keep them in sync; SemVer).
- [ ] Move `[Unreleased]` notes to a dated heading in `CHANGELOG.md`.
- [ ] Confirm the public API additions are documented in the README.

## 2. Verify locally

```bash
python -m pip install -e ".[dev,viz,yaml]"
python -m pytest -q                 # all tests must pass
ruff check src tests                # lint clean
ruff format --check src tests       # formatting (optional)
python -m build                     # builds sdist + wheel
twine check dist/*                  # metadata + README render OK
```

## 3. Clean-room install test

```bash
python -m venv /tmp/cb && /tmp/cb/bin/pip install dist/ctxbudgeter-*.whl
/tmp/cb/bin/python -c "import ctxbudgeter; print(ctxbudgeter.__version__)"
/tmp/cb/bin/ctxbudgeter --help
# core install must NOT require plotly/jinja2:
/tmp/cb/bin/python -c "from ctxbudgeter.viz import ContextMRI; print('viz importable')"
```

## 4. Tag + GitHub release

```bash
git add -A && git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
gh release create vX.Y.Z --title "vX.Y.Z" --notes-from-tag
```

Creating the GitHub Release fires `.github/workflows/publish.yml`, which builds and
uploads to PyPI via Trusted Publishing. If a `pypi` environment reviewer rule is
configured, approve the deployment in the Actions tab.

## 5. Trusted Publishing setup (one-time, on PyPI)

PyPI → Account → Publishing → Add a pending/trusted publisher:

- Owner: `Kayariyan28`
- Repository: `ctxbudgeter`
- Workflow filename: `publish.yml`
- Environment: `pypi`

### Fallback (only if Trusted Publishing is unavailable)

```bash
python -m build
twine check dist/*
twine upload dist/*    # needs a PyPI API token in your local keyring — never commit it
```

## 6. Verify the published release

```bash
pip install -U ctxbudgeter
python -c "import ctxbudgeter; print(ctxbudgeter.__version__)"
ctxbudgeter --help
```

## 7. Post-release

- [ ] Confirm the PyPI project page renders the README and shows the new version.
- [ ] Announce (LinkedIn/X), update any badges.

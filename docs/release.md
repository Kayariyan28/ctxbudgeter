# Releasing ctxbudgeter

The authoritative, step-by-step checklist lives at the repo root:
[`RELEASE_CHECKLIST.md`](../RELEASE_CHECKLIST.md).

Quick reference:

```bash
# 1. bump version in pyproject.toml + src/ctxbudgeter/__init__.py (keep in sync)
# 2. update CHANGELOG.md
python -m pip install -e ".[dev,viz,yaml]"
python -m pytest -q
ruff check src tests
python -m build
twine check dist/*

# clean-room install
python -m venv /tmp/cb && /tmp/cb/bin/pip install dist/ctxbudgeter-*.whl
/tmp/cb/bin/ctxbudgeter --help

# tag + release (Trusted Publishing handles the upload via publish.yml)
git tag vX.Y.Z && git push origin main --tags
gh release create vX.Y.Z --notes-from-tag
```

Trusted Publishing is configured on PyPI (owner `Kayariyan28`, repo `ctxbudgeter`,
workflow `publish.yml`, environment `pypi`) — **no API token is stored in the repo**.
Fallback `twine upload` is documented in `RELEASE_CHECKLIST.md` but never commits a
token.

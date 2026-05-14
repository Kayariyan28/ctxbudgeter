"""Eval / assertion API — "pytest for prompt/context quality".

Drop these into your test suite to gate PRs on context quality the same way you
gate on code quality:

    from ctxbudgeter.testing import (
        assert_includes, assert_excludes,
        assert_health_at_least, assert_cacheable_prefix_at_least,
        assert_no_secret_items, assert_no_compression_of,
        assert_utilization_between, assert_used_tokens_at_most,
        GoldenPack,
    )

    def test_prod_pack():
        compiled = build_prod_pack().compile()
        assert_includes(compiled, "system_rules", "task")
        assert_excludes(compiled, "debug.log", "old_notes.md")
        assert_health_at_least(compiled, 80)
        assert_cacheable_prefix_at_least(compiled, 1024)
        assert_no_secret_items(compiled)
        assert_used_tokens_at_most(compiled, 20_000)

Golden snapshot testing for context regressions:

    def test_pack_golden():
        golden = GoldenPack("tests/golden/prod_pack.json")
        golden.check(build_prod_pack().compile())
"""

from __future__ import annotations

import json
from pathlib import Path

from .compiler import CompiledPack


class ContextAssertionError(AssertionError):
    """Raised by ctxbudgeter assertions. Subclasses AssertionError so pytest handles it."""


def _statuses_of(pack: CompiledPack, name: str) -> str:
    for d in pack.decisions:
        if d.name == name:
            return d.status
    return "<missing>"


def assert_includes(pack: CompiledPack, *names: str) -> None:
    """Fail unless all `names` were included (or compressed/truncated/redacted)."""
    included = {
        d.name for d in pack.decisions
        if d.status in ("included", "compressed", "truncated", "redacted")
    }
    missing = [n for n in names if n not in included]
    if missing:
        statuses = ", ".join(f"{n}={_statuses_of(pack, n)}" for n in missing)
        raise ContextAssertionError(
            f"Expected items to be included but found: {statuses}"
        )


def assert_excludes(pack: CompiledPack, *names: str) -> None:
    """Fail if any of `names` was included."""
    included = {
        d.name for d in pack.decisions
        if d.status in ("included", "compressed", "truncated", "redacted")
    }
    leaked = [n for n in names if n in included]
    if leaked:
        raise ContextAssertionError(
            f"Expected items to be excluded but they were included: {leaked}"
        )


def assert_health_at_least(pack: CompiledPack, threshold: int) -> None:
    """Fail if health_score is below `threshold`."""
    if pack.health_score < threshold:
        raise ContextAssertionError(
            f"health_score {pack.health_score} < {threshold}. "
            f"Breakdown: {pack.health_breakdown}"
        )


def assert_cacheable_prefix_at_least(pack: CompiledPack, tokens: int) -> None:
    """Fail if the cacheable prefix is smaller than `tokens`."""
    if pack.cacheable_prefix_tokens < tokens:
        raise ContextAssertionError(
            f"cacheable_prefix_tokens {pack.cacheable_prefix_tokens} < {tokens}. "
            "Are your stable items at the top of the pack? Check cache_policy."
        )


def assert_no_secret_items(pack: CompiledPack) -> None:
    """Fail if any included item has sensitivity='secret' (and was not redacted)."""
    leaked = [
        d.name for d in pack.decisions
        if d.sensitivity == "secret" and d.status in ("included", "compressed", "truncated")
    ]
    if leaked:
        raise ContextAssertionError(
            f"Items with sensitivity='secret' were included unredacted: {leaked}. "
            "Set CompilerConfig.secret_policy='refuse' or 'redact'."
        )


def assert_no_compression_of(pack: CompiledPack, *names: str) -> None:
    """Fail if any of `names` was compressed or truncated (i.e. lossily reduced)."""
    bad = [
        d.name for d in pack.decisions
        if d.name in names and d.status in ("compressed", "truncated")
    ]
    if bad:
        raise ContextAssertionError(
            f"Expected lossless inclusion but these were compressed/truncated: {bad}"
        )


def assert_utilization_between(pack: CompiledPack, lo: float, hi: float) -> None:
    """Fail if pack.utilization is outside [lo, hi]."""
    u = pack.utilization
    if not (lo <= u <= hi):
        raise ContextAssertionError(
            f"utilization {u:.3f} outside expected range [{lo}, {hi}]. "
            "Pack is either underfilled (waste) or near overflow (risky)."
        )


def assert_used_tokens_at_most(pack: CompiledPack, max_tokens: int) -> None:
    """Fail if used_tokens exceeds `max_tokens` (a hard input-cost ceiling)."""
    if pack.used_tokens > max_tokens:
        raise ContextAssertionError(
            f"used_tokens {pack.used_tokens} > {max_tokens} (input cost ceiling)."
        )


def assert_includes_in_order(pack: CompiledPack, *names: str) -> None:
    """Fail unless `names` appear in the included_items list in that order
    (other items may be between them)."""
    order = [it.name for it in pack.included_items]
    pos = -1
    for n in names:
        try:
            new_pos = order.index(n, pos + 1)
        except ValueError as e:
            raise ContextAssertionError(
                f"Expected '{n}' to appear after position {pos} in prompt order; "
                f"order was {order}"
            ) from e
        pos = new_pos


def assert_no_loader_failures(pack: CompiledPack) -> None:
    """Fail if any reference loader failed during resolution."""
    failures = [w for w in pack.warnings if "failed" in w.lower() or "raised" in w.lower()]
    if failures:
        raise ContextAssertionError("Reference loader failures:\n  - " + "\n  - ".join(failures))


def assert_warnings_empty(pack: CompiledPack) -> None:
    """Fail if there are any warnings (strict mode for CI)."""
    if pack.warnings:
        raise ContextAssertionError("Compilation produced warnings:\n  - " + "\n  - ".join(pack.warnings))


# ----- Golden snapshot testing -------------------------------------------------------


class GoldenPack:
    """Golden-file regression testing for compiled packs.

    On first run (or with `update=True`), saves the pack's decisions + counts to
    `path`. On subsequent runs, compares the new pack to the stored golden and
    raises on diff.

    Stored fields are the auditable ones: decisions (name + status + reason),
    used_tokens, cacheable_prefix_tokens, health_score, health_breakdown,
    included_order. Token counts and content hashes are NOT stored — those
    change with tokenizer/version. Edit `ignore_fields` to customize.
    """

    DEFAULT_KEYS = (
        "decisions_summary",
        "included_order",
        "used_tokens",
        "cacheable_prefix_tokens",
        "health_score",
        "health_breakdown",
    )

    def __init__(self, path: str | Path, *, update: bool = False, keys: tuple[str, ...] | None = None) -> None:
        self.path = Path(path)
        self.update = update
        self.keys = keys or self.DEFAULT_KEYS

    def _summarize(self, pack: CompiledPack) -> dict:
        summary = {
            "decisions_summary": [
                {"name": d.name, "status": d.status, "kind": d.kind, "required": d.required}
                for d in sorted(pack.decisions, key=lambda d: d.name)
            ],
            "included_order": [it.name for it in pack.included_items],
            "used_tokens": pack.used_tokens,
            "cacheable_prefix_tokens": pack.cacheable_prefix_tokens,
            "health_score": pack.health_score,
            "health_breakdown": dict(pack.health_breakdown),
        }
        return {k: summary[k] for k in self.keys if k in summary}

    def check(self, pack: CompiledPack) -> None:
        """Compare to stored golden file. On first run, save and pass."""
        current = self._summarize(pack)
        if not self.path.exists() or self.update:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(current, indent=2, sort_keys=True))
            return
        stored = json.loads(self.path.read_text())
        if stored != current:
            diff = _shallow_diff(stored, current)
            raise ContextAssertionError(
                f"Golden mismatch in {self.path}. Re-run with --ctxbudgeter-update-golden "
                f"to refresh.\nDiff: {json.dumps(diff, indent=2)}"
            )


def _shallow_diff(a: dict, b: dict) -> dict:
    """Return only the keys that differ between two dicts (shallow)."""
    diff: dict = {}
    keys = set(a) | set(b)
    for k in sorted(keys):
        if a.get(k) != b.get(k):
            diff[k] = {"expected": a.get(k), "actual": b.get(k)}
    return diff

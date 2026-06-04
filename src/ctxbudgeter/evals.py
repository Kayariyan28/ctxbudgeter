"""CI-friendly context evals.

Declare expectations about a compiled context (or its BOM) and check them in CI:
required sources present, forbidden sources absent, token ceiling respected,
no PII/secrets, minimum health score, provenance present, no stale content.

Deterministic and offline. Load suites from YAML/JSON and run them against a BOM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bom import ContextBOM

if TYPE_CHECKING:
    from .compiler import CompiledPack


@dataclass
class EvalCheck:
    """One pass/fail check within an eval result."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalResult:
    """Outcome of running a single ContextEval."""

    name: str
    passed: bool
    checks: list[EvalCheck] = field(default_factory=list)

    @property
    def failures(self) -> list[EvalCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class ContextEval:
    """A declarative context expectation.

    Run against a ``ContextBOM`` or a ``CompiledPack``. Any unmet expectation makes
    ``result.passed`` False with a per-check breakdown.
    """

    name: str
    task: str | None = None
    required_sources: list[str] = field(default_factory=list)
    forbidden_sources: list[str] = field(default_factory=list)
    max_tokens: int | None = None
    must_not_contain: list[str] = field(default_factory=list)
    min_context_health_score: int | None = None
    max_risk_score: int | None = None
    block_pii: bool = False
    block_secrets: bool = False
    require_provenance: bool = False

    def run(self, target: ContextBOM | CompiledPack) -> EvalResult:
        bom = _as_bom(target)
        checks: list[EvalCheck] = []

        included_sources = [i.source or "" for i in bom.included_items]
        included_names = [i.name for i in bom.included_items]

        def _source_present(needle: str) -> bool:
            n = needle.lower()
            return any(n in (s or "").lower() for s in included_sources) or any(
                n in name.lower() for name in included_names
            )

        for src in self.required_sources:
            present = _source_present(src)
            checks.append(EvalCheck(
                name=f"required_source:{src}",
                passed=present,
                detail="present" if present else f"required source '{src}' not in included context",
            ))

        for src in self.forbidden_sources:
            present = _source_present(src)
            checks.append(EvalCheck(
                name=f"forbidden_source:{src}",
                passed=not present,
                detail="absent" if not present else f"forbidden source '{src}' was included",
            ))

        if self.max_tokens is not None:
            ok = bom.total_tokens <= self.max_tokens
            checks.append(EvalCheck(
                name="max_tokens",
                passed=ok,
                detail=f"{bom.total_tokens} <= {self.max_tokens}" if ok
                else f"token budget exceeded: {bom.total_tokens} > {self.max_tokens}",
            ))

        if self.min_context_health_score is not None:
            ok = bom.context_health_score >= self.min_context_health_score
            checks.append(EvalCheck(
                name="min_context_health_score",
                passed=ok,
                detail=f"health {bom.context_health_score} >= {self.min_context_health_score}" if ok
                else f"health too low: {bom.context_health_score} < {self.min_context_health_score}",
            ))

        if self.max_risk_score is not None:
            ok = bom.risk_score <= self.max_risk_score
            checks.append(EvalCheck(
                name="max_risk_score",
                passed=ok,
                detail=f"risk {bom.risk_score} <= {self.max_risk_score}" if ok
                else f"risk too high: {bom.risk_score} > {self.max_risk_score}",
            ))

        redacted = set(bom.redacted_items)
        if self.block_secrets:
            # A finding only counts if its item was NOT redacted away.
            live = [
                f for f in bom.scanner_findings
                if _finding_is_secret(f) and str(f.get("item_name", "")) not in redacted
            ]
            checks.append(EvalCheck(
                name="block_secrets",
                passed=not live,
                detail="no unredacted secrets" if not live
                else f"unredacted secrets in: {sorted({f.get('item_name') for f in live})}",
            ))

        if self.block_pii:
            live = [
                f for f in bom.scanner_findings
                if _finding_is_pii(f) and str(f.get("item_name", "")) not in redacted
            ]
            checks.append(EvalCheck(
                name="block_pii",
                passed=not live,
                detail="no unredacted PII" if not live
                else f"unredacted PII in: {sorted({f.get('item_name') for f in live})}",
            ))

        if self.must_not_contain:
            # Check against item names, sources, and scanner categories — never raw content.
            haystack = " ".join(
                included_names + included_sources
                + [str(f.get("category", "")) for f in bom.scanner_findings]
            ).lower()
            for needle in self.must_not_contain:
                present = needle.lower() in haystack
                checks.append(EvalCheck(
                    name=f"must_not_contain:{needle}",
                    passed=not present,
                    detail="absent" if not present else f"'{needle}' surfaced in context metadata",
                ))

        if self.require_provenance:
            missing = [i.name for i in bom.included_items if not (i.source or i.trust_level)]
            ok = not missing
            checks.append(EvalCheck(
                name="require_provenance",
                passed=ok,
                detail="all items have provenance" if ok else f"missing provenance: {missing}",
            ))

        passed = all(c.passed for c in checks)
        return EvalResult(name=self.name, passed=passed, checks=checks)


@dataclass
class EvalSuite:
    """A collection of evals loaded from YAML/JSON."""

    evals: list[ContextEval] = field(default_factory=list)

    def run(self, target: ContextBOM | CompiledPack) -> list[EvalResult]:
        return [e.run(target) for e in self.evals]

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.evals]

    @classmethod
    def from_file(cls, path: str) -> EvalSuite:
        import json

        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "YAML eval files require PyYAML. Install with `pip install ctxbudgeter[yaml]`."
                ) from e
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        raw_evals = data.get("evals", []) if isinstance(data, dict) else data
        evals = [_eval_from_dict(e) for e in raw_evals]
        return cls(evals=evals)


def _eval_from_dict(data: dict[str, Any]) -> ContextEval:
    known = set(ContextEval.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown eval keys for '{data.get('name', '?')}': {sorted(unknown)}")
    return ContextEval(**data)


def _as_bom(target: ContextBOM | CompiledPack) -> ContextBOM:
    if isinstance(target, ContextBOM):
        return target
    # Duck-type a CompiledPack.
    if hasattr(target, "included_items") and hasattr(target, "decisions"):
        return ContextBOM.from_compiled(target)  # type: ignore[arg-type]
    raise TypeError("ContextEval.run expects a ContextBOM or CompiledPack")


def _finding_is_secret(f: dict) -> bool:
    from .scanner import _SECRET_CATEGORIES

    return str(f.get("category")) in _SECRET_CATEGORIES


def _finding_is_pii(f: dict) -> bool:
    from .scanner import _PII_CATEGORIES

    return str(f.get("category")) in _PII_CATEGORIES

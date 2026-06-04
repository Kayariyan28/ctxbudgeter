"""ContextPolicy — declarative governance for what may enter the model.

A ``ContextPolicy`` is a deterministic rule set evaluated at compile time. It
governs token budgets, PII/secret blocking, provenance requirements, source
allow/deny lists, content age, cache layout preference, and per-item token caps.

The policy itself performs no I/O and calls no LLM. It produces
``PolicyViolation`` records that the compiler and Bill of Materials surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ViolationSeverity = Literal["info", "warning", "error", "critical"]


@dataclass(frozen=True)
class PolicyViolation:
    """A single policy rule breach, attributable to an item where applicable."""

    rule: str
    severity: ViolationSeverity
    message: str
    item_name: str | None = None
    action: str = "flag"  # "flag" | "exclude" | "redact" | "block"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContextPolicy:
    """Governance policy applied during compilation.

    See module docstring. All fields have safe defaults; construct with only the
    knobs you care about. Validation runs in ``__post_init__``.
    """

    max_tokens: int = 24_000
    reserved_output_tokens: int = 4_000
    block_pii: bool = False
    block_secrets: bool = True
    require_provenance: bool = False
    allowed_sources: list[str] | None = None
    forbidden_sources: list[str] | None = None
    max_age_days: int | None = None
    cache_stable_prefix: bool = True
    fail_on_policy_violation: bool = False
    max_item_tokens: int | None = None
    allow_untrusted_sources: bool = True
    redact_sensitive: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must be >= 0")
        if self.reserved_output_tokens >= self.max_tokens:
            raise ValueError("reserved_output_tokens must be smaller than max_tokens")
        if self.max_item_tokens is not None and self.max_item_tokens <= 0:
            raise ValueError("max_item_tokens must be positive if supplied")
        if self.max_age_days is not None and self.max_age_days < 0:
            raise ValueError("max_age_days must be >= 0 if supplied")

    @property
    def available_context_tokens(self) -> int:
        return self.max_tokens - self.reserved_output_tokens

    # ----- source matching --------------------------------------------------

    def is_source_forbidden(self, source: str | None) -> bool:
        if not source or not self.forbidden_sources:
            return False
        s = source.lower()
        return any(pat.lower() in s for pat in self.forbidden_sources)

    def is_source_allowed(self, source: str | None) -> bool:
        """True if the source passes the allow-list. If no allow-list is set,
        everything is allowed (subject to the deny-list elsewhere)."""
        if not self.allowed_sources:
            return True
        if not source:
            return False
        s = source.lower()
        return any(pat.lower() in s for pat in self.allowed_sources)

    def summary(self) -> dict[str, Any]:
        """Compact, human-readable policy summary for BOM/report embedding."""
        return {
            "max_tokens": self.max_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "available_context_tokens": self.available_context_tokens,
            "block_pii": self.block_pii,
            "block_secrets": self.block_secrets,
            "require_provenance": self.require_provenance,
            "allowed_sources": list(self.allowed_sources) if self.allowed_sources else None,
            "forbidden_sources": list(self.forbidden_sources) if self.forbidden_sources else None,
            "max_age_days": self.max_age_days,
            "max_item_tokens": self.max_item_tokens,
            "cache_stable_prefix": self.cache_stable_prefix,
            "allow_untrusted_sources": self.allow_untrusted_sources,
            "redact_sensitive": self.redact_sensitive,
            "fail_on_policy_violation": self.fail_on_policy_violation,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ----- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextPolicy:
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown policy keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> ContextPolicy:
        """Load a policy from a YAML (preferred) or JSON file.

        YAML requires the optional ``pyyaml`` dependency
        (``pip install ctxbudgeter[yaml]``). JSON works with no extra deps.
        """
        import json
        from pathlib import Path

        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "YAML policy files require PyYAML. Install with "
                    "`pip install ctxbudgeter[yaml]`, or use a .json policy file."
                ) from e
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Policy file must contain a mapping/object at the top level.")
        # Allow a nested {"policy": {...}} layout (ctxbudgeter.yaml convention).
        if "policy" in data and isinstance(data["policy"], dict):
            data = data["policy"]
        return cls.from_dict(data)


@dataclass
class PolicyDecision:
    """The outcome of evaluating one item against a policy."""

    item_name: str
    action: Literal["allow", "exclude", "redact", "flag"]
    violations: list[PolicyViolation] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.action == "exclude"

    @property
    def needs_redaction(self) -> bool:
        return self.action == "redact"

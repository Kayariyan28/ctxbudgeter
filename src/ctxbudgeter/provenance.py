"""Context provenance — where context came from and how much to trust it.

Provenance is a lightweight, deterministic metadata record attached to a
``ContextItem``. It records the origin, ownership, trust level, age, retrieval
rank, and transformation history of a piece of context so that downstream tools
(policy enforcement, the Bill of Materials, Context MRI) can reason about it.

Local-first: nothing here touches the network. Timestamps are accepted as
ISO-8601 strings (so output stays deterministic — the caller controls the clock).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TrustLevel = Literal["unknown", "low", "internal", "verified"]
"""How much to trust a source.

- unknown: origin not declared
- low: untrusted / user-supplied / third-party
- internal: internal system or repo
- verified: reviewed, owned, authoritative
"""

_TRUST_ORDER: dict[str, int] = {"unknown": 0, "low": 1, "internal": 2, "verified": 3}


@dataclass
class ContextProvenance:
    """Provenance metadata for a single context item.

    All fields are optional so provenance can be added incrementally. Use
    :meth:`trust_score` for a normalized 0-1 trust value and :meth:`age_days`
    to compute staleness against a reference time.
    """

    source: str | None = None
    source_type: str | None = None  # "file" | "url" | "db" | "api" | "memory" | "tool" | ...
    source_uri: str | None = None
    owner: str | None = None
    created_at: str | None = None  # ISO-8601
    modified_at: str | None = None  # ISO-8601
    trust_level: TrustLevel = "unknown"
    retrieval_rank: int | None = None
    transformation_history: list[str] = field(default_factory=list)
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trust_level not in _TRUST_ORDER:
            raise ValueError(
                f"trust_level must be one of {sorted(_TRUST_ORDER)}, got {self.trust_level!r}"
            )
        if self.retrieval_rank is not None and self.retrieval_rank < 0:
            raise ValueError("retrieval_rank must be >= 0")

    # ----- derived signals --------------------------------------------------

    def trust_score(self) -> float:
        """Normalized trust in [0, 1]."""
        return _TRUST_ORDER[self.trust_level] / 3.0

    def has_provenance(self) -> bool:
        """True if any meaningful origin field is set."""
        return any(
            v for v in (self.source, self.source_uri, self.owner, self.source_type)
        )

    def age_days(self, *, now: str | datetime | None = None) -> float | None:
        """Days between ``modified_at`` (or ``created_at``) and ``now``.

        Returns None if no timestamp is available or parseable. ``now`` may be an
        ISO-8601 string or a datetime; if omitted the current UTC time is used.
        """
        stamp = self.modified_at or self.created_at
        if not stamp:
            return None
        parsed = _parse_iso(stamp)
        if parsed is None:
            return None
        if now is None:
            ref = datetime.now(timezone.utc)
        elif isinstance(now, datetime):
            ref = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        else:
            ref_parsed = _parse_iso(now)
            if ref_parsed is None:
                return None
            ref = ref_parsed
        delta = ref - parsed
        return max(0.0, delta.total_seconds() / 86400.0)

    def with_transformation(self, label: str) -> ContextProvenance:
        """Return a copy with ``label`` appended to transformation_history."""
        return ContextProvenance(
            source=self.source,
            source_type=self.source_type,
            source_uri=self.source_uri,
            owner=self.owner,
            created_at=self.created_at,
            modified_at=self.modified_at,
            trust_level=self.trust_level,
            retrieval_rank=self.retrieval_rank,
            transformation_history=[*self.transformation_history, label],
            checksum=self.checksum,
            metadata=dict(self.metadata),
        )

    # ----- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextProvenance:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _parse_iso(value: str) -> datetime | None:
    """Best-effort ISO-8601 parse. Returns a tz-aware UTC datetime or None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    # Accept trailing Z
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    # Date-only
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(v, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def content_checksum(text: str) -> str:
    """Deterministic short checksum of content (sha256, first 16 hex chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

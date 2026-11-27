"""Context events — the unit of a *live* agent context.

Where a :class:`~ctxbudgeter.item.ContextItem` is a candidate for a single compiled
prompt, a :class:`ContextEvent` is one entry in an agent's evolving history: a
system rule, the user's requirement, a tool call, a tool result, an assistant turn,
or a summary that replaced a range of earlier events.

Events carry a stable *tag* (``"7"``), and summaries carry a range tag
(``"3..17"``) naming the first and last event they cover — the same convention as
NVIDIA NOOA's ``events.collapse`` API, so an agent can retrieve any archived
original by the tag it saw before compaction.

Every event is assigned a :data:`ContextCategory` that drives governance:

========================  =====================================================
``protected``             system instructions, permissions, user requirements —
                          never compacted or archived by an agent
``task_critical``         current objectives, evidence, dependencies — kept
                          verbatim unless a governed summary retains its facts
``working``               current reasoning, plans, recent tool results
``compressible``          completed steps, verbose logs, redundant content
``archivable``            inactive information, recoverable on demand
========================  =====================================================
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Literal

ContextCategory = Literal["protected", "task_critical", "working", "compressible", "archivable"]

CATEGORIES: tuple[ContextCategory, ...] = (
    "protected",
    "task_critical",
    "working",
    "compressible",
    "archivable",
)

#: Retention rank: lower = more important to keep verbatim.
CATEGORY_RANK: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

EventKind = Literal[
    "system",
    "task",
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "document",
    "note",
    "summary",
    "other",
]

#: Default category per event kind. Callers can always override explicitly.
DEFAULT_CATEGORY: dict[str, ContextCategory] = {
    "system": "protected",
    "task": "protected",
    "user_message": "task_critical",
    "note": "task_critical",
    "assistant_message": "working",
    "tool_call": "working",
    "tool_result": "working",
    "document": "working",
    "summary": "working",
    "other": "working",
}


def checksum(text: str) -> str:
    """Short, stable content fingerprint (sha256, 16 hex chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def range_tag(first: str, last: str) -> str:
    """Build a summary tag covering ``first..last`` (NOOA convention)."""
    return first if first == last else f"{first}..{last}"


@dataclass
class ContextEvent:
    """One entry of a live agent context."""

    tag: str
    kind: str
    content: str
    tokens: int
    category: ContextCategory = "working"
    step: int = 0
    source: str | None = None
    pinned: bool = False
    pin_reason: str | None = None
    #: For summary events: the tags of the original events this summary replaced.
    covers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checksum(self) -> str:
        return checksum(self.content)

    @property
    def is_summary(self) -> bool:
        return self.kind == "summary"

    @property
    def is_protected(self) -> bool:
        return self.category == "protected"

    def preview(self, n: int = 120) -> str:
        flat = " ".join(self.content.split())
        return flat if len(flat) <= n else flat[: n - 1] + "…"

    def copy(self, **changes: Any) -> ContextEvent:
        return replace(
            self,
            covers=list(changes.pop("covers", self.covers)),
            metadata=dict(changes.pop("metadata", self.metadata)),
            **changes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "kind": self.kind,
            "content": self.content,
            "tokens": self.tokens,
            "category": self.category,
            "step": self.step,
            "source": self.source,
            "pinned": self.pinned,
            "pin_reason": self.pin_reason,
            "covers": list(self.covers),
            "metadata": dict(self.metadata),
        }

    def to_view(self, preview_chars: int = 120) -> dict[str, Any]:
        """Content-light view for ``inspect()`` output and decision traces."""
        return {
            "tag": self.tag,
            "kind": self.kind,
            "category": self.category,
            "tokens": self.tokens,
            "pinned": self.pinned,
            "source": self.source,
            "step": self.step,
            "covers": list(self.covers),
            "preview": self.preview(preview_chars),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContextEvent:
        return cls(
            tag=str(d["tag"]),
            kind=str(d.get("kind", "other")),
            content=str(d.get("content", "")),
            tokens=int(d.get("tokens", 0)),
            category=d.get("category", "working"),
            step=int(d.get("step", 0)),
            source=d.get("source"),
            pinned=bool(d.get("pinned", False)),
            pin_reason=d.get("pin_reason"),
            covers=[str(t) for t in d.get("covers", [])],
            metadata=dict(d.get("metadata", {})),
        )

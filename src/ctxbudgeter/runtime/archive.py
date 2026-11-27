"""Archive stores — where compacted and archived events go.

Compaction in ctxbudgeter is *non-destructive*: originals move out of the active
context into an archive, keyed by their original tag, so an agent can retrieve them
later (the NOOA ``self.events["9"]`` pattern). The archive is append-only — rollback
restores the active context but never deletes archived originals.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .analysis import match_score, query_terms
from .events import ContextEvent


class ArchiveStore(ABC):
    """Abstract archive. Implement ``put``/``get``/``all`` to plug in your backend."""

    @abstractmethod
    def put(self, event: ContextEvent) -> None: ...

    @abstractmethod
    def get(self, tag: str) -> ContextEvent | None: ...

    @abstractmethod
    def all(self) -> list[ContextEvent]: ...

    def __contains__(self, tag: object) -> bool:
        return isinstance(tag, str) and self.get(tag) is not None

    def __len__(self) -> int:
        return len(self.all())

    def search(self, query: str, *, limit: int = 5, min_score: float = 0.2) -> list[tuple[ContextEvent, float]]:
        """Keyword search over archived originals (summaries excluded)."""
        terms = query_terms(query)
        scored = [
            (ev, match_score(terms, ev.content))
            for ev in self.all()
            if not ev.is_summary
        ]
        scored = [(ev, s) for ev, s in scored if s >= min_score]
        scored.sort(key=lambda p: (-p[1], _tag_key(p[0].tag)))
        return scored[:limit]

    def expand(self, tag: str) -> list[ContextEvent]:
        """Return the original (non-summary) events behind ``tag``, recursively."""
        ev = self.get(tag)
        if ev is None:
            return []
        if not ev.is_summary:
            return [ev]
        out: list[ContextEvent] = []
        for t in ev.covers:
            out.extend(self.expand(t))
        return out


class InMemoryArchive(ArchiveStore):
    def __init__(self) -> None:
        self._data: dict[str, ContextEvent] = {}

    def put(self, event: ContextEvent) -> None:
        # Append-only: the first archived copy of a tag wins (originals are immutable).
        self._data.setdefault(event.tag, event.copy())

    def get(self, tag: str) -> ContextEvent | None:
        ev = self._data.get(tag)
        return ev.copy() if ev is not None else None

    def all(self) -> list[ContextEvent]:
        return [ev.copy() for _, ev in sorted(self._data.items(), key=lambda kv: _tag_key(kv[0]))]


class JSONArchive(InMemoryArchive):
    """File-backed archive that survives process restarts (single writer)."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for d in raw.get("events", []):
                ev = ContextEvent.from_dict(d)
                self._data[ev.tag] = ev

    def put(self, event: ContextEvent) -> None:
        if event.tag in self._data:
            return
        super().put(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"events": [e.to_dict() for e in self.all()]}, indent=2),
            encoding="utf-8",
        )


def _tag_key(tag: str) -> tuple[int, int, str]:
    """Sort key for tags: numeric order, summaries after their first covered event."""
    head, _, tail = tag.partition("..")
    try:
        return (int(head), int(tail) if tail else int(head), tag)
    except ValueError:
        return (10**9, 10**9, tag)

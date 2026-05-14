"""Memory persistence — the LangChain "write" strategy.

A MemoryStore lets an agent write notes between turns and pull them back into
context on later turns. The compiler integrates via `pack.add_memory(store, ...)`,
which queries the store and adds matching items to the pack.

The store is deliberately simple — keyword queries on tags + content substring.
Plug in a real vector store via the abstract base class if you need semantic recall.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MemoryNote:
    """A single memory entry. Tags drive simple keyword matching; content is text."""

    key: str
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "content": self.content,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNote":
        return cls(
            key=d["key"],
            content=d["content"],
            tags=list(d.get("tags", [])),
            created_at=float(d.get("created_at", time.time())),
            metadata=dict(d.get("metadata", {})),
        )


class MemoryStore(ABC):
    """Abstract memory store. Implement read/write/query/delete to plug in your backend."""

    @abstractmethod
    def write(self, note: MemoryNote) -> None: ...

    @abstractmethod
    def read(self, key: str) -> Optional[MemoryNote]: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def query(
        self,
        *,
        text: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        since: Optional[float] = None,
    ) -> list[MemoryNote]: ...

    @abstractmethod
    def all(self) -> list[MemoryNote]: ...


class InMemoryStore(MemoryStore):
    """Process-local store. Good for tests and ephemeral agents."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryNote] = {}

    def write(self, note: MemoryNote) -> None:
        self._data[note.key] = note

    def read(self, key: str) -> Optional[MemoryNote]:
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def query(
        self,
        *,
        text: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        since: Optional[float] = None,
    ) -> list[MemoryNote]:
        results = list(self._data.values())
        if since is not None:
            results = [n for n in results if n.created_at >= since]
        if tags:
            tagset = set(tags)
            results = [n for n in results if tagset & set(n.tags)]
        if text:
            tl = text.lower()
            results = [n for n in results if tl in n.content.lower() or tl in n.key.lower()]
        results.sort(key=lambda n: n.created_at, reverse=True)
        return results[:limit]

    def all(self) -> list[MemoryNote]:
        return sorted(self._data.values(), key=lambda n: n.created_at, reverse=True)


class JSONMemoryStore(MemoryStore):
    """File-backed JSON store. Persists across runs. Not concurrency-safe — fine
    for single-agent or single-developer workflows; use a real DB for multi-agent."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, MemoryNote] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for k, v in raw.items():
            try:
                self._data[k] = MemoryNote.from_dict(v)
            except (KeyError, TypeError):
                continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: n.to_dict() for k, n in self._data.items()}, indent=2),
            encoding="utf-8",
        )

    def write(self, note: MemoryNote) -> None:
        self._data[note.key] = note
        self._save()

    def read(self, key: str) -> Optional[MemoryNote]:
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        existed = self._data.pop(key, None) is not None
        if existed:
            self._save()
        return existed

    def query(
        self,
        *,
        text: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10,
        since: Optional[float] = None,
    ) -> list[MemoryNote]:
        # Delegate to InMemoryStore logic with our data
        proxy = InMemoryStore()
        proxy._data = self._data
        return proxy.query(text=text, tags=tags, limit=limit, since=since)

    def all(self) -> list[MemoryNote]:
        return sorted(self._data.values(), key=lambda n: n.created_at, reverse=True)

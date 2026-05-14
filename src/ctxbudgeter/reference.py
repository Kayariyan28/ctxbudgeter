"""Lazy context references — Anthropic's "just-in-time" pattern.

A Reference is a lightweight pointer (file path, URL, query, key) that knows how to
materialize itself into a ContextItem at compile time. References that are unlikely
to fit the budget are never loaded — saving disk reads, HTTP calls, and DB queries.

Loader contract:
    loader(ref: Reference) -> str | Awaitable[str]

Loaders raise on failure; the compiler catches and excludes the reference with a
clear reason. Async loaders are supported via `acompile()`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Union

from .item import CachePolicy, ContextItem, ContextKind, Sensitivity

SyncLoader = Callable[["Reference"], str]
AsyncLoader = Callable[["Reference"], Awaitable[str]]
Loader = Union[SyncLoader, AsyncLoader]


@dataclass
class Reference:
    """A lazy pointer to content that gets loaded only if it could plausibly fit.

    Use `estimated_tokens` to give the budget planner a hint — references whose
    estimate clearly exceeds the remaining budget are skipped without ever calling
    the loader.

    Set `required=True` if the loader MUST run (the compiler will load and try
    compression/truncation before excluding).

    The `loader` callable can be sync or async; async needs `pack.acompile()`.
    """

    name: str
    location: str
    loader: Loader | None = None
    estimated_tokens: int = 0  # 0 = unknown; compiler will load if budget permits
    kind: ContextKind = "retrieval"
    priority: int = 50
    required: bool = False
    relevance: float = 0.5
    freshness: float = 1.0
    cache_policy: CachePolicy = "dynamic"
    sensitivity: Sensitivity = "internal"
    compressible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Reference.name cannot be blank")
        self.name = self.name.strip()
        if not 0 <= self.priority <= 100:
            raise ValueError("Reference.priority must be in [0, 100]")
        if self.estimated_tokens < 0:
            raise ValueError("Reference.estimated_tokens must be >= 0")

    def to_item(self, content: str) -> ContextItem:
        """Wrap resolved content as a regular ContextItem."""
        return ContextItem(
            name=self.name,
            content=content,
            kind=self.kind,
            priority=self.priority,
            required=self.required,
            relevance=self.relevance,
            freshness=self.freshness,
            source=self.location,
            cache_policy=self.cache_policy,
            sensitivity=self.sensitivity,
            compressible=self.compressible,
            metadata=dict(self.metadata),
        )

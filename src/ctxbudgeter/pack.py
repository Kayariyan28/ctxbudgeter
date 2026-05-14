"""ContextPack — the developer-facing builder.

Typical flow:
    pack = ContextPack(model="claude-sonnet-4.6", token_budget=24_000)
    pack.add(name="system_rules", content="...", priority=100, kind="system", cache_policy="stable")
    pack.add_file("README.md", priority=80, kind="project_doc")
    pack.add_reference("api_docs", "https://example.com/docs", loader=fetch_docs, estimated_tokens=1500)
    pack.add_memory(store, query="auth", limit=3)
    pack.add(name="task", content="...", priority=95, kind="task", required=True)
    compiled = pack.compile()                # sync
    compiled = await pack.acompile()         # async — for async loaders/compressors
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .compiler import (
    CompiledPack,
    CompilerConfig,
    Compressor,
    acompile_items,
    compile_items,
)
from .content import Attachment
from .item import CachePolicy, ContextItem, ContextKind, Sensitivity
from .memory import MemoryStore
from .reference import Loader, Reference
from .tokenizer import TokenCounter


class ContextPack:
    """Mutable builder for a context bundle."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4.6",
        token_budget: int = 200_000,
        reserved_output_tokens: int = 4_000,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        if reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must be >= 0")
        if reserved_output_tokens >= token_budget:
            raise ValueError("reserved_output_tokens must be less than token_budget")
        self.model = model
        self.token_budget = token_budget
        self.reserved_output_tokens = reserved_output_tokens
        self.items: list[ContextItem] = []
        self.references: list[Reference] = []
        self._config: CompilerConfig = CompilerConfig()

    # ----- adding items ---------------------------------------------------------

    def add(
        self,
        *,
        name: str,
        content: str = "",
        kind: ContextKind = "other",
        priority: int = 50,
        required: bool = False,
        freshness: float = 1.0,
        relevance: float = 0.5,
        source: str | None = None,
        cache_policy: CachePolicy = "dynamic",
        sensitivity: Sensitivity = "internal",
        compressible: bool = False,
        compressed_content: str | None = None,
        attachments: list[Attachment] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextItem:
        item = ContextItem(
            name=name,
            content=content,
            kind=kind,
            priority=priority,
            required=required,
            freshness=freshness,
            relevance=relevance,
            source=source,
            cache_policy=cache_policy,
            sensitivity=sensitivity,
            compressible=compressible,
            compressed_content=compressed_content,
            attachments=attachments or [],
            metadata=metadata or {},
        )
        return self.add_item(item)

    def add_item(self, item: ContextItem) -> ContextItem:
        self._reject_duplicate(item.name)
        self.items.append(item)
        return item

    def add_file(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        priority: int = 50,
        kind: ContextKind = "project_doc",
        required: bool = False,
        cache_policy: CachePolicy = "stable",
        sensitivity: Sensitivity = "internal",
        compressible: bool = True,
        encoding: str = "utf-8",
    ) -> ContextItem:
        p = Path(path)
        content = p.read_text(encoding=encoding)
        return self.add(
            name=name or p.name,
            content=content,
            kind=kind,
            priority=priority,
            required=required,
            cache_policy=cache_policy,
            sensitivity=sensitivity,
            compressible=compressible,
            source=str(p),
        )

    def add_reference(
        self,
        name: str,
        location: str,
        *,
        loader: Loader | None = None,
        estimated_tokens: int = 0,
        priority: int = 50,
        kind: ContextKind = "retrieval",
        required: bool = False,
        cache_policy: CachePolicy = "dynamic",
        sensitivity: Sensitivity = "internal",
        compressible: bool = True,
        relevance: float = 0.5,
        freshness: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Reference:
        """Add a lazy Reference. The loader runs at compile time only if the
        reference could plausibly fit the budget."""
        self._reject_duplicate(name)
        ref = Reference(
            name=name,
            location=location,
            loader=loader,
            estimated_tokens=estimated_tokens,
            kind=kind,
            priority=priority,
            required=required,
            cache_policy=cache_policy,
            sensitivity=sensitivity,
            compressible=compressible,
            relevance=relevance,
            freshness=freshness,
            metadata=metadata or {},
        )
        self.references.append(ref)
        return ref

    def add_reference_obj(self, ref: Reference) -> Reference:
        self._reject_duplicate(ref.name)
        self.references.append(ref)
        return ref

    def add_memory(
        self,
        store: MemoryStore,
        *,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
        kind: ContextKind = "memory",
        priority: int = 50,
        cache_policy: CachePolicy = "dynamic",
        name_prefix: str = "memory_",
    ) -> list[ContextItem]:
        """Query a MemoryStore and add the matching notes as ContextItems.

        Notes are added with names `{name_prefix}{key}`. Duplicate keys are
        skipped silently.
        """
        notes = store.query(text=query, tags=tags, limit=limit)
        added: list[ContextItem] = []
        for note in notes:
            item_name = f"{name_prefix}{note.key}"
            if item_name in self:
                continue
            item = self.add(
                name=item_name,
                content=note.content,
                kind=kind,
                priority=priority,
                cache_policy=cache_policy,
                source=f"memory:{note.key}",
                metadata={"tags": list(note.tags), "created_at": note.created_at, **dict(note.metadata)},
            )
            added.append(item)
        return added

    # ----- mutation -------------------------------------------------------------

    def remove(self, name: str) -> bool:
        before = len(self.items) + len(self.references)
        self.items = [it for it in self.items if it.name != name]
        self.references = [r for r in self.references if r.name != name]
        return (len(self.items) + len(self.references)) < before

    def get(self, name: str) -> ContextItem | None:
        for it in self.items:
            if it.name == name:
                return it
        return None

    def get_reference(self, name: str) -> Reference | None:
        for r in self.references:
            if r.name == name:
                return r
        return None

    # ----- isolation (Isolate strategy) -----------------------------------------

    def fork(
        self,
        *,
        filter: Callable[[ContextItem], bool] | None = None,
        ref_filter: Callable[[Reference], bool] | None = None,
        token_budget: int | None = None,
        reserved_output_tokens: int | None = None,
        model: str | None = None,
    ) -> ContextPack:
        """Create a new pack with a filtered subset of items + references.

        Use this to isolate a subagent's context — give it only the items it needs,
        with its own budget. The filter is applied to ContextItems; ref_filter to
        References. None means keep all.
        """
        child = ContextPack(
            model=model or self.model,
            token_budget=token_budget if token_budget is not None else self.token_budget,
            reserved_output_tokens=(
                reserved_output_tokens
                if reserved_output_tokens is not None
                else self.reserved_output_tokens
            ),
        )
        for it in self.items:
            if filter is None or filter(it):
                child.items.append(it.model_copy())
        for r in self.references:
            if ref_filter is None or ref_filter(r):
                child.references.append(r)
        child._config = CompilerConfig(
            weights=dict(self._config.weights),
            allow_truncation=self._config.allow_truncation,
            truncation_marker=self._config.truncation_marker,
            compressor=self._config.compressor,
            compression_retry=self._config.compression_retry,
            secret_policy=self._config.secret_policy,
        )
        return child

    def subset_by_kind(self, *kinds: ContextKind) -> ContextPack:
        """Convenience fork: keep only items of the given kinds."""
        kindset = set(kinds)
        return self.fork(filter=lambda it: it.kind in kindset)

    def subset_by_namespace(self, namespace: str) -> ContextPack:
        """Convenience fork: keep only items whose metadata['namespace'] matches."""
        return self.fork(
            filter=lambda it: it.metadata.get("namespace") == namespace,
            ref_filter=lambda r: r.metadata.get("namespace") == namespace,
        )

    # ----- configuration --------------------------------------------------------

    def set_compressor(self, fn: Compressor) -> None:
        """Register a compression hook. Can be sync or async (async needs acompile())."""
        self._config.compressor = fn

    def set_config(self, config: CompilerConfig) -> None:
        self._config = config

    def set_secret_policy(self, policy: str) -> None:
        """Set sensitivity enforcement: 'allow' | 'warn' | 'refuse' | 'redact'."""
        self._config.secret_policy = policy  # type: ignore[assignment]

    @property
    def config(self) -> CompilerConfig:
        return self._config

    # ----- compilation ----------------------------------------------------------

    def compile(self) -> CompiledPack:
        return compile_items(
            list(self.items),
            model=self.model,
            token_budget=self.token_budget,
            reserved_output_tokens=self.reserved_output_tokens,
            config=self._config,
            counter=TokenCounter(self.model),
            references=list(self.references),
        )

    async def acompile(self) -> CompiledPack:
        return await acompile_items(
            list(self.items),
            model=self.model,
            token_budget=self.token_budget,
            reserved_output_tokens=self.reserved_output_tokens,
            config=self._config,
            counter=TokenCounter(self.model),
            references=list(self.references),
        )

    def preview(self) -> CompiledPack:
        """Alias for compile() — kept for API symmetry with `estimate_tokens()`.

        Use when you want to see what *would* be compiled without committing to a
        production run (cheap and deterministic since there are no LLM calls).
        """
        return self.compile()

    def estimate_tokens(self) -> int:
        """Sum of text token counts of ALL items (not the compiled subset).

        For the post-compile token usage, call `compile().used_tokens` or `preview()`.
        """
        counter = TokenCounter(self.model)
        return sum(counter.count(it.content) for it in self.items)

    # ----- dunder ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items) + len(self.references)

    def __iter__(self) -> Iterator[ContextItem]:
        return iter(self.items)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return any(it.name == name for it in self.items) or any(
            r.name == name for r in self.references
        )

    # ----- internal -------------------------------------------------------------

    def _reject_duplicate(self, name: str) -> None:
        if any(it.name == name for it in self.items) or any(
            r.name == name for r in self.references
        ):
            raise ValueError(f"Item or reference with name '{name}' already exists in this pack.")

"""ctxbudgeter — compile clean, cheap, auditable context for AI agents.

Framework-agnostic. Local-first. Deterministic. No LLM API calls in the core.

Quick start::

    from ctxbudgeter import ContextPack

    pack = ContextPack(model="claude-sonnet-4.6", token_budget=24_000, reserved_output_tokens=4_000)
    pack.add(name="system_rules", content="You are a careful coding agent...",
             kind="system", priority=100, cache_policy="stable", required=True)
    pack.add_file("README.md", kind="project_doc", priority=80)
    pack.add(name="task", content="Build the referral packet UI.",
             kind="task", priority=95, required=True)

    compiled = pack.compile()
    print(compiled.report())
"""

from __future__ import annotations

from . import adapters, loaders
from .compiler import (
    BudgetExceededError,
    CompiledPack,
    CompilerConfig,
    Compressor,
    ItemDecision,
    SecretContentError,
    SecretPolicy,
    acompile_items,
    compile_items,
    compiled_pack_from_dict,
)
from .content import Attachment, ImageBlock, StructuredBlock, TextBlock
from .item import CachePolicy, ContextItem, ContextKind, Sensitivity
from .memory import InMemoryStore, JSONMemoryStore, MemoryNote, MemoryStore
from .pack import ContextPack
from .reference import AsyncLoader, Loader, Reference, SyncLoader
from .report import to_json, to_markdown, to_text
from .scoring import DEFAULT_WEIGHTS, score_item
from .tokenizer import TokenCounter

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_WEIGHTS",
    "AsyncLoader",
    # multimodal
    "Attachment",
    "BudgetExceededError",
    "CachePolicy",
    "CompiledPack",
    "CompilerConfig",
    "Compressor",
    # core
    "ContextItem",
    "ContextKind",
    "ContextPack",
    "ImageBlock",
    "InMemoryStore",
    "ItemDecision",
    "JSONMemoryStore",
    "Loader",
    "MemoryNote",
    # memory (Write strategy)
    "MemoryStore",
    # references (just-in-time loading)
    "Reference",
    "SecretContentError",
    "SecretPolicy",
    "Sensitivity",
    "StructuredBlock",
    "SyncLoader",
    "TextBlock",
    "TokenCounter",
    # version
    "__version__",
    "acompile_items",
    # adapters
    "adapters",
    "compile_items",
    "compiled_pack_from_dict",
    "loaders",
    "score_item",
    "to_json",
    "to_markdown",
    # reports
    "to_text",
]

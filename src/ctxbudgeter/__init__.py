"""ctxbudgeter — the model-agnostic context intelligence layer for AI agents.

Compile, audit and govern context before every model call (0.1-0.3), and — new in
0.4 — enable, teach, govern and evaluate how agents manage their *own* context:
``runtime`` (governed context API), ``governance`` (edit policy + graduated autonomy),
``skills`` (portable context-management skill), ``adaptive`` (behavioural learning),
``bench`` (CTXBench), ``train`` (datasets) and ``adapters`` (OpenAI / Anthropic / MCP).

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

from typing import Any

from . import adapters, loaders
from .bom import BOMItem, ContextBOM
from .cache import CachePlan, CachePlanner
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
from .diff import ContextDiff
from .evals import ContextEval, EvalResult, EvalSuite
from .governance import PolicyViolationError, enforce_policy
from .item import CachePolicy, ContextItem, ContextKind, Sensitivity
from .mcp import MCPToolBudgeter
from .memory import InMemoryStore, JSONMemoryStore, MemoryNote, MemoryStore
from .pack import ContextPack
from .policy import ContextPolicy, PolicyViolation
from .provenance import ContextProvenance
from .reference import AsyncLoader, Loader, Reference, SyncLoader
from .report import to_json, to_markdown, to_text
from .scanner import ContextScanner, Finding, ScanResult
from .scoring import DEFAULT_WEIGHTS, score_item
from .tokenizer import TokenCounter

__version__ = "0.4.1"

# 0.4 Adaptive Context Engine — resolved lazily so `import ctxbudgeter` stays light.
_LAZY: dict[str, tuple[str, str]] = {
    "ContextRuntime": ("ctxbudgeter.runtime", "ContextRuntime"),
    "ContextEvent": ("ctxbudgeter.runtime", "ContextEvent"),
    "HarnessCompactor": ("ctxbudgeter.runtime", "HarnessCompactor"),
    "CompactionRubric": ("ctxbudgeter.runtime", "CompactionRubric"),
    "ContextEditPolicy": ("ctxbudgeter.governance", "ContextEditPolicy"),
    "EditGovernor": ("ctxbudgeter.governance", "EditGovernor"),
    "AutonomyLevel": ("ctxbudgeter.governance", "AutonomyLevel"),
    "AutonomyRegistry": ("ctxbudgeter.governance", "AutonomyRegistry"),
    "render_guidance": ("ctxbudgeter.skills", "render_guidance"),
    "BehaviorProfiler": ("ctxbudgeter.adaptive", "BehaviorProfiler"),
    "PolicyTuner": ("ctxbudgeter.adaptive", "PolicyTuner"),
    "AdaptivePolicy": ("ctxbudgeter.adaptive", "AdaptivePolicy"),
    "BenchRunner": ("ctxbudgeter.bench", "BenchRunner"),
    "TaskGenerator": ("ctxbudgeter.bench", "TaskGenerator"),
    "ContextHarness": ("ctxbudgeter.adapters.harness", "ContextHarness"),
    "ContextMCPServer": ("ctxbudgeter.adapters.mcp_server", "ContextMCPServer"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


__all__ = [
    "DEFAULT_WEIGHTS",
    # 0.4 Adaptive Context Engine (lazy)
    "AdaptivePolicy",
    "AsyncLoader",
    # multimodal
    "Attachment",
    "AutonomyLevel",
    "AutonomyRegistry",
    "BOMItem",
    "BehaviorProfiler",
    "BenchRunner",
    "BudgetExceededError",
    "CachePlan",
    "CachePlanner",
    "CachePolicy",
    "CompactionRubric",
    "CompiledPack",
    "CompilerConfig",
    "Compressor",
    # ContextOps: audit & analysis
    "ContextBOM",
    "ContextDiff",
    "ContextEditPolicy",
    "ContextEval",
    "ContextEvent",
    "ContextHarness",
    # core
    "ContextItem",
    "ContextKind",
    "ContextMCPServer",
    "ContextPack",
    # ContextOps: governance & scanning
    "ContextPolicy",
    "ContextProvenance",
    "ContextRuntime",
    "ContextScanner",
    "EditGovernor",
    "EvalResult",
    "EvalSuite",
    "Finding",
    "HarnessCompactor",
    "ImageBlock",
    "InMemoryStore",
    "ItemDecision",
    "JSONMemoryStore",
    "Loader",
    "MCPToolBudgeter",
    "MemoryNote",
    # memory (Write strategy)
    "MemoryStore",
    "PolicyTuner",
    "PolicyViolation",
    "PolicyViolationError",
    # references (just-in-time loading)
    "Reference",
    "ScanResult",
    "SecretContentError",
    "SecretPolicy",
    "Sensitivity",
    "StructuredBlock",
    "SyncLoader",
    "TaskGenerator",
    "TextBlock",
    "TokenCounter",
    # version
    "__version__",
    "acompile_items",
    # adapters
    "adapters",
    "compile_items",
    "compiled_pack_from_dict",
    "enforce_policy",
    "loaders",
    "render_guidance",
    "score_item",
    "to_json",
    "to_markdown",
    # reports
    "to_text",
]

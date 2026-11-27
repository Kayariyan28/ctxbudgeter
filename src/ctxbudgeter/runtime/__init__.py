"""ctxbudgeter.runtime — live, governed, model-driven context management.

    from ctxbudgeter.runtime import ContextRuntime

    rt = ContextRuntime(model="gpt-6.1", budget=32_000, reserved_output_tokens=4_000)
    rt.add("You are a research agent...", kind="system")
    rt.add("Answer using FY2019 figures.", kind="task")
    rt.add(tool_output, kind="tool_result", source="docs/treasury_2019.txt")
    rt.step()                                   # one model request
    rt.budget(); rt.diagnose(); rt.plan()
    rt.compact("3", "9", summary="...")         # governed; originals archived
    rt.recall("5")                              # originals stay recoverable

Leaf modules are imported eagerly; the runtime itself is loaded lazily so that
``ctxbudgeter.governance`` (which depends on the leaf modules) can be imported
without a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .analysis import retention, salient_terms
from .archive import ArchiveStore, InMemoryArchive, JSONArchive
from .cache_model import PrefillCostModel, RequestCost
from .events import CATEGORIES, DEFAULT_CATEGORY, ContextCategory, ContextEvent
from .rubric import PHASES, CompactionRubric, RubricVerdict
from .summarizer import ExtractiveSummarizer, Summarizer
from .trace import DecisionTrace, TraceLog, load_traces

if TYPE_CHECKING:
    from .diagnostics import Diagnosis, diagnose
    from .harness import HarnessCompactor
    from .planner import ContextPlan, ProposedOp
    from .runtime import (
        ContextOverflowError,
        ContextRuntime,
        OperationResult,
        PendingEdit,
        StepReport,
    )

_LAZY = {
    "ContextRuntime": ".runtime",
    "OperationResult": ".runtime",
    "PendingEdit": ".runtime",
    "StepReport": ".runtime",
    "ContextOverflowError": ".runtime",
    "HarnessCompactor": ".harness",
    "ContextPlan": ".planner",
    "ProposedOp": ".planner",
    "Diagnosis": ".diagnostics",
    "diagnose": ".diagnostics",
}


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(mod, __name__), name)


__all__ = [
    "CATEGORIES",
    "DEFAULT_CATEGORY",
    "PHASES",
    "ArchiveStore",
    "CompactionRubric",
    "ContextCategory",
    "ContextEvent",
    "ContextOverflowError",
    "ContextPlan",
    "ContextRuntime",
    "DecisionTrace",
    "Diagnosis",
    "ExtractiveSummarizer",
    "HarnessCompactor",
    "InMemoryArchive",
    "JSONArchive",
    "OperationResult",
    "PendingEdit",
    "PrefillCostModel",
    "ProposedOp",
    "RequestCost",
    "RubricVerdict",
    "StepReport",
    "Summarizer",
    "TraceLog",
    "diagnose",
    "load_traces",
    "retention",
    "salient_terms",
]

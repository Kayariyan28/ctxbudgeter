"""ctxbudgeter.core — the deterministic context compilation layer (0.1-0.3 API).

A stable namespace for the compile-time toolkit, alongside the 0.4 packages
(``runtime``, ``governance``, ``skills``, ``adaptive``, ``bench``, ``train``,
``adapters``). Everything here is also importable from ``ctxbudgeter`` directly.
"""

from __future__ import annotations

from ..bom import BOMItem, ContextBOM
from ..cache import CachePlan, CachePlanner
from ..compiler import (
    BudgetExceededError,
    CompiledPack,
    CompilerConfig,
    ItemDecision,
    SecretContentError,
    acompile_items,
    compile_items,
    compiled_pack_from_dict,
)
from ..diff import ContextDiff
from ..evals import ContextEval, EvalSuite
from ..item import ContextItem
from ..pack import ContextPack
from ..policy import ContextPolicy, PolicyViolation
from ..scanner import ContextScanner
from ..tokenizer import TokenCounter

__all__ = [
    "BOMItem",
    "BudgetExceededError",
    "CachePlan",
    "CachePlanner",
    "CompiledPack",
    "CompilerConfig",
    "ContextBOM",
    "ContextDiff",
    "ContextEval",
    "ContextItem",
    "ContextPack",
    "ContextPolicy",
    "ContextScanner",
    "EvalSuite",
    "ItemDecision",
    "PolicyViolation",
    "SecretContentError",
    "TokenCounter",
    "acompile_items",
    "compile_items",
    "compiled_pack_from_dict",
]

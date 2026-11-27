"""Context-API tools for any LLM provider or harness.

One set of tool definitions, rendered three ways:

* :func:`openai_tools`     — Chat Completions / Responses ``tools`` (function calling)
* :func:`anthropic_tools`  — Messages API ``tools``
* :func:`mcp_tools`        — MCP ``tools/list`` entries

and one :class:`ContextToolDispatcher` that executes a tool call against a
:class:`~ctxbudgeter.runtime.ContextRuntime` and returns a JSON-serializable result.
Every mutating call goes through the runtime's governor and autonomy level, so
exposing these tools to a model never grants unrestricted authority.

Tool names use ``context_<op>`` (``[a-zA-Z0-9_-]``) to satisfy OpenAI, Anthropic and
MCP naming rules alike.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.runtime import ContextRuntime

_TAG = {"type": "string", "description": "Event tag as shown in context, e.g. '7' or '3..17'."}

TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "context_inspect", "description": "See what is in your context: budget, composition by category "
     "and kind, and every event (tag, kind, category, tokens, pinned, preview).",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "context_budget", "description": "Tokens used/free, usage fraction, pressure level and the "
     "compaction timing verdict (block | defer | allow | compact_now).",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "context_diagnose", "description": "Find duplicate, superseded, conflicting, stale, oversized, "
     "repetitive or sensitive events, with the tokens each fix recovers and a suggested operation.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "context_plan", "description": "Propose a cache-aware list of compact/archive operations that "
     "brings usage down to a target without touching protected, pinned or recent events.",
     "parameters": {"type": "object", "properties": {
         "target_usage": {"type": "number", "minimum": 0.05, "maximum": 0.95,
                          "description": "Target usage fraction (default: just below the floor)."}},
                    "additionalProperties": False}},
    {"name": "context_compact", "description": "Replace events start..end (inclusive) with a summary. "
     "Originals are archived and recallable by tag. Write the summary yourself when you can: keep every "
     "number, identifier, decision and correction. Omit it to use the default extractive summarizer.",
     "parameters": {"type": "object", "properties": {
         "start": _TAG, "end": _TAG,
         "summary": {"type": "string", "description": "Your summary of the range."},
         "target_tokens": {"type": "integer", "minimum": 16,
                           "description": "Length target for the default summarizer."}},
                    "required": ["start", "end"], "additionalProperties": False}},
    {"name": "context_pin", "description": "Pin an event so it cannot be compacted or archived (pin budget "
     "applies). Pin evidence your final answer depends on.",
     "parameters": {"type": "object", "properties": {"tag": _TAG, "reason": {"type": "string"}},
                    "required": ["tag"], "additionalProperties": False}},
    {"name": "context_unpin", "description": "Remove a pin.",
     "parameters": {"type": "object", "properties": {"tag": _TAG}, "required": ["tag"],
                    "additionalProperties": False}},
    {"name": "context_archive", "description": "Move events out of active context. They stay recoverable "
     "with context_recall.",
     "parameters": {"type": "object", "properties": {"tags": {"type": "array", "items": _TAG, "minItems": 1}},
                    "required": ["tags"], "additionalProperties": False}},
    {"name": "context_recall", "description": "Read an archived event by its original tag. restore=true puts "
     "it back into active context (governed).",
     "parameters": {"type": "object", "properties": {"tag": _TAG, "restore": {"type": "boolean"}},
                    "required": ["tag"], "additionalProperties": False}},
    {"name": "context_search_archive", "description": "Keyword search over archived originals. Use before "
     "answering when the question names something you no longer see.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                     "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["query"], "additionalProperties": False}},
    {"name": "context_checkpoint", "description": "Snapshot the active context before a risky edit.",
     "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "additionalProperties": False}},
    {"name": "context_rollback", "description": "Restore a checkpoint (default: the most recent, including "
     "automatic pre-edit checkpoints).",
     "parameters": {"type": "object", "properties": {"checkpoint_id": {"type": "string"}},
                    "additionalProperties": False}},
    {"name": "context_evaluate", "description": "Evaluate one decision (by trace_id) or your context management "
     "so far: compression, fact retention, timing, reopen rate, violations, prefill cost.",
     "parameters": {"type": "object", "properties": {"trace_id": {"type": "string"}},
                    "additionalProperties": False}},
    {"name": "context_signal", "description": "Declare your phase so compaction timing advice fits: exploring, "
     "mid_derivation, stuck, subtask_complete, converging, answering.",
     "parameters": {"type": "object", "properties": {"phase": {"type": "string", "enum": [
         "exploring", "mid_derivation", "stuck", "subtask_complete", "converging", "answering"]}},
                    "required": ["phase"], "additionalProperties": False}},
]

#: Harness-side tool, exposed only when explicitly enabled (e.g. an MCP server that
#: acts as an external context store for a client that cannot hand over its own context).
ADD_TOOL: dict[str, Any] = {
    "name": "context_add", "description": "Append an event to the managed context (harness-side).",
    "parameters": {"type": "object", "properties": {
        "content": {"type": "string"},
        "kind": {"type": "string", "enum": ["system", "task", "user_message", "assistant_message", "tool_call",
                                            "tool_result", "document", "note", "other"]},
        "category": {"type": "string", "enum": ["protected", "task_critical", "working", "compressible",
                                                "archivable"]},
        "source": {"type": "string"}},
                   "required": ["content"], "additionalProperties": False},
}

TOOL_NAMES = tuple(t["name"] for t in TOOL_SPECS)


def _specs(include_add: bool) -> list[dict[str, Any]]:
    return TOOL_SPECS + ([ADD_TOOL] if include_add else [])


def openai_tools(*, include_add: bool = False) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["parameters"]}} for t in _specs(include_add)]


def anthropic_tools(*, include_add: bool = False) -> list[dict[str, Any]]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in _specs(include_add)]


def mcp_tools(*, include_add: bool = False) -> list[dict[str, Any]]:
    return [{"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]}
            for t in _specs(include_add)]


class ContextToolError(ValueError):
    pass


class ContextToolDispatcher:
    """Execute context tool calls against a runtime. Results are plain JSON data."""

    def __init__(self, runtime: ContextRuntime, *, allow_add: bool = False) -> None:
        self.rt = runtime
        self.allow_add = allow_add
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "context_inspect": lambda a: self.rt.inspect(),
            "context_budget": lambda a: self.rt.budget(),
            "context_diagnose": lambda a: [d.to_dict() for d in self.rt.diagnose()],
            "context_plan": lambda a: self.rt.plan(target_usage=a.get("target_usage")).to_dict(),
            "context_compact": lambda a: self.rt.compact(a["start"], a.get("end", a["start"]), a.get("summary"),
                                                         target_tokens=a.get("target_tokens")).to_dict(),
            "context_pin": lambda a: self.rt.pin(a["tag"], a.get("reason", "")).to_dict(),
            "context_unpin": lambda a: self.rt.unpin(a["tag"]).to_dict(),
            "context_archive": lambda a: self.rt.archive(*_as_list(a["tags"])).to_dict(),
            "context_recall": lambda a: self.rt.recall(a["tag"], restore=bool(a.get("restore", False))).to_dict(),
            "context_search_archive": lambda a: self.rt.search_archive(a["query"], limit=int(a.get("limit", 5))),
            "context_checkpoint": lambda a: self.rt.checkpoint(a.get("label")).to_dict(),
            "context_rollback": lambda a: self.rt.rollback(a.get("checkpoint_id")).to_dict(),
            "context_evaluate": lambda a: self.rt.evaluate(a.get("trace_id")),
            "context_signal": self._signal,
            "context_add": self._add,
        }

    def names(self) -> tuple[str, ...]:
        return TOOL_NAMES + (("context_add",) if self.allow_add else ())

    def handles(self, name: str) -> bool:
        return name in self.names()

    def call(self, name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
        """Run one tool call. Never raises for model mistakes: returns ``{"error": ...}``."""
        if not self.handles(name):
            return {"error": f"unknown tool {name!r}", "available": list(self.names())}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        except json.JSONDecodeError as e:
            return {"error": f"arguments are not valid JSON: {e}"}
        try:
            result = self._handlers[name](args)
        except KeyError as e:
            return {"error": f"missing or unknown argument/tag: {e.args[0]}"}
        except (ValueError, TypeError) as e:
            return {"error": str(e)}
        return {"result": result}

    def _signal(self, a: dict[str, Any]) -> dict[str, Any]:
        self.rt.signal(a["phase"])
        return {"phase": self.rt.phase, "timing": self.rt.budget()["timing"]}

    def _add(self, a: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_add:
            raise ValueError("context_add is disabled")
        ev = self.rt.add(a["content"], kind=a.get("kind", "tool_result"), category=a.get("category"),
                         source=a.get("source"))
        return ev.to_view()


def _as_list(v: Any) -> list[str]:
    return [str(x) for x in v] if isinstance(v, list | tuple) else [str(v)]

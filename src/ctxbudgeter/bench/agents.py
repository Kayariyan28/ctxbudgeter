"""Agents for CTXBench.

Two kinds:

* :class:`ScriptedAgent` — deterministic reference behaviours. The presets reproduce
  the context-management *archetypes* NVIDIA measured (a competent manager, an
  anxious early-compactor, an avoidant never-compactor, a reckless over-compressor).
  They exist so the benchmark, the governor and the adaptive loop can be validated
  reproducibly and in CI. They are behaviour models, not models of any vendor's LLM.

* :class:`LLMContextAgent` — a real model managing its context through the context
  tools. Provider-agnostic: you pass a ``complete(messages, tools) -> tool_calls``
  function; :func:`openai_complete` and :func:`anthropic_complete` build one from an
  OpenAI or Anthropic client.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..adapters.context_tools import ContextToolDispatcher, anthropic_tools, openai_tools
from ..runtime.rubric import RubricVerdict
from ..runtime.runtime import ContextRuntime


@dataclass
class Observation:
    step: int
    usage: float
    warning: str | None = None
    timing: RubricVerdict | None = None
    guidance: str | None = None
    phase: str | None = None


class ContextAgent(Protocol):
    name: str

    def act(self, rt: ContextRuntime, obs: Observation) -> None: ...

    def prepare_answer(self, rt: ContextRuntime, question: str, obs: Observation) -> list[tuple[str, str]]:
        """Return recalled ``(tag, content)`` pairs to read alongside active context."""
        ...


_RETRY_RULES = {"over_compression", "summary_detail_loss", "critical_fact_loss"}


@dataclass
class ScriptedAgent:
    name: str = "competent"
    trigger: float = 0.85
    target_usage: float = 0.5
    compression: float = 4.0
    respects_guidance: bool = True
    never_compact: bool = False
    archive_redundant: bool = True
    retry_on_denial: bool = True
    recalls: bool = True
    reckless: bool = False
    stop_on_warning: bool = False
    stopped: bool = field(default=False, init=False)

    def act(self, rt: ContextRuntime, obs: Observation) -> None:
        if self.stop_on_warning and obs.warning:
            self.stopped = True
            return
        if self.never_compact:
            return
        u = obs.usage
        if self.respects_guidance and obs.timing is not None:
            d = obs.timing.decision
            go = d == "compact_now" or (d == "allow" and u >= self.trigger)
        else:
            go = u >= self.trigger
        if not go:
            return
        if self.archive_redundant:
            for diag in rt.diagnose():
                if diag.kind in ("duplicate", "conflict", "superseded") and diag.suggestion.get("op") == "archive":
                    tags = [t for t in diag.suggestion["tags"] if t in rt]
                    if tags:
                        rt.archive(*tags)
        if self.reckless:
            tags = [e.tag for e in rt.events]
            if len(tags) > 3:
                span = sum(e.tokens for e in rt.events[:-1])
                rt.compact(tags[0], tags[-2], target_tokens=max(24, math.ceil(span / self.compression)))
            return
        if rt.usage < self.target_usage:
            return
        plan = rt.plan(target_usage=self.target_usage, compression=self.compression)
        for op in plan.ops:
            if op.op == "archive":
                rt.archive(*[t for t in op.args["tags"] if t in rt])
                continue
            start, end = op.args["start"], op.args["end"]
            if start not in rt or end not in rt:
                continue
            toks = _range_tokens(rt, start, end)
            target = max(24, math.ceil(toks / self.compression))
            r = rt.compact(start, end, target_tokens=target)
            tries = 0
            while (r.status == "denied" and self.retry_on_denial and tries < 2 and
                   {v["rule"] for v in r.decision.get("violations", [])} & _RETRY_RULES):
                target *= 2
                tries += 1
                r = rt.compact(start, end, target_tokens=target)

    def prepare_answer(self, rt: ContextRuntime, question: str, obs: Observation) -> list[tuple[str, str]]:
        if not self.recalls:
            return []
        out: list[tuple[str, str]] = []
        for h in rt.recall_hints(question, limit=8):
            r = rt.recall(h["tag"])
            if r.ok and "content" in r.details:
                out.append((h["tag"], r.details["content"]))
        return out


def _range_tokens(rt: ContextRuntime, start: str, end: str) -> int:
    evs = rt.events
    idx = {e.tag: i for i, e in enumerate(evs)}
    i, j = sorted((idx[start], idx[end]))
    return sum(e.tokens for e in evs[i:j + 1])


PRESETS: dict[str, Callable[[], ScriptedAgent]] = {
    "competent": lambda: ScriptedAgent("competent"),
    "anxious": lambda: ScriptedAgent("anxious", trigger=0.3, target_usage=0.2, compression=6.0,
                                     archive_redundant=False, retry_on_denial=False),
    "avoidant": lambda: ScriptedAgent("avoidant", never_compact=True, respects_guidance=False, recalls=False),
    "reckless": lambda: ScriptedAgent("reckless", trigger=0.7, target_usage=0.15, compression=15.0,
                                      respects_guidance=False, archive_redundant=False, retry_on_denial=False,
                                      reckless=True),
}


def make_agent(name: str) -> ScriptedAgent:
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(f"unknown agent preset {name!r}; expected one of {sorted(PRESETS)}") from None


# ===================================================================== LLM agent

ToolCall = dict[str, Any]  # {"id": str, "name": str, "arguments": dict}
CompleteFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[ToolCall]]


class LLMContextAgent:
    """A real model managing its own context through the context tools.

    ``complete(messages, tools)`` receives OpenAI-format ``messages`` (system / user /
    assistant-with-``tool_calls`` / ``tool``) and OpenAI-format ``tools`` and returns
    the tool calls the model made (empty list = done). Each turn is bounded by
    ``max_calls_per_turn``; the model only acts when usage reaches ``act_from_usage``
    or a pressure warning arrives, which keeps cost proportional to need.
    """

    def __init__(self, complete: CompleteFn, *, name: str = "llm", act_from_usage: float = 0.5,
                 max_calls_per_turn: int = 6, guidance: str | None = None) -> None:
        self.complete = complete
        self.name = name
        self.act_from_usage = act_from_usage
        self.max_calls = max_calls_per_turn
        self.guidance = guidance
        self.calls_made = 0
        self.errors: list[str] = []

    def _system(self, obs: Observation) -> str:
        if obs.guidance or self.guidance:
            return obs.guidance or self.guidance or ""
        from ..skills import render_guidance

        return render_guidance(brief=True)

    def _loop(self, rt: ContextRuntime, prompt: str, obs: Observation) -> list[dict[str, Any]]:
        disp = ContextToolDispatcher(rt)
        tools = openai_tools()
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self._system(obs)},
                                      {"role": "user", "content": prompt}]
        results: list[dict[str, Any]] = []
        for _ in range(self.max_calls):
            calls = self.complete(msgs, tools)
            if not calls:
                break
            ids = [c.get("id") or f"call_{self.calls_made + k}" for k, c in enumerate(calls)]
            msgs.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}
                for cid, c in zip(ids, calls, strict=True)]})
            for cid, c in zip(ids, calls, strict=True):
                out = disp.call(c["name"], c.get("arguments") or {})
                self.calls_made += 1
                if "error" in out:
                    self.errors.append(out["error"])
                results.append({"name": c["name"], "arguments": c.get("arguments") or {}, "output": out})
                msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(out, default=str)[:12000]})
        return results

    def act(self, rt: ContextRuntime, obs: Observation) -> None:
        if obs.usage < self.act_from_usage and not obs.warning:
            return
        state = {"budget": rt.budget(), "events": [e.to_view(100) for e in rt.events]}
        prompt = ("Context maintenance turn. Decide whether to manage your context now, using the context "
                  "tools; reply with no tool calls if nothing should change.\n"
                  + (f"{obs.warning}\n" if obs.warning else "") + json.dumps(state, default=str))
        self._loop(rt, prompt, obs)

    def prepare_answer(self, rt: ContextRuntime, question: str, obs: Observation) -> list[tuple[str, str]]:
        prompt = ("You are about to answer the question below. Make sure every fact it needs is visible; use "
                  "context_search_archive and context_recall for anything you no longer see. Reply with no tool "
                  f"calls when ready.\n{question}\n"
                  + json.dumps({"events": [e.to_view(100) for e in rt.events]}, default=str))
        out: list[tuple[str, str]] = []
        for r in self._loop(rt, prompt, obs):
            res = r["output"].get("result")
            if r["name"] == "context_recall" and isinstance(res, dict) and "content" in res.get("details", {}):
                out.append((res["details"]["tag"], res["details"]["content"]))
        return out


def openai_complete(client: Any, model: str, **kwargs: Any) -> CompleteFn:
    """Build a ``complete`` function from an ``openai.OpenAI`` client (Chat Completions)."""

    def complete(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[ToolCall]:
        resp = client.chat.completions.create(model=model, messages=messages, tools=tools, **kwargs)
        msg = resp.choices[0].message
        return [{"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments or "{}")}
                for tc in (msg.tool_calls or [])]

    return complete


def anthropic_complete(client: Any, model: str, *, max_tokens: int = 2048, **kwargs: Any) -> CompleteFn:
    """Build a ``complete`` function from an ``anthropic.Anthropic`` client (Messages API)."""
    a_tools = anthropic_tools()

    def complete(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[ToolCall]:
        system, conv = to_anthropic_conversation(messages)
        resp = client.messages.create(model=model, system=system, messages=conv, tools=a_tools,
                                      max_tokens=max_tokens, **kwargs)
        return [{"id": b.id, "name": b.name, "arguments": dict(b.input or {})}
                for b in resp.content if getattr(b, "type", None) == "tool_use"]

    return complete


def to_anthropic_conversation(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-format messages (with tool calls) to Anthropic ``system`` + ``messages``."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system" and m.get("content"))
    conv: list[dict[str, Any]] = []

    def push(role: str, block: dict[str, Any]) -> None:
        if conv and conv[-1]["role"] == role:
            conv[-1]["content"].append(block)
        else:
            conv.append({"role": role, "content": [block]})

    for m in messages:
        if m["role"] == "user":
            push("user", {"type": "text", "text": m["content"]})
        elif m["role"] == "assistant":
            if m.get("content"):
                push("assistant", {"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                push("assistant", {"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                   "input": json.loads(tc["function"]["arguments"] or "{}")})
        elif m["role"] == "tool":
            push("user", {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]})
    return system, conv

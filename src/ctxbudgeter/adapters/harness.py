"""ContextHarness — drop the governed runtime into your own agent loop.

    harness = ContextHarness(ContextRuntime(budget=64_000, model="gpt-6.1"), provider="openai")
    harness.runtime.add(SYSTEM, kind="system"); harness.runtime.add(task, kind="task")
    while True:
        req = harness.request()                       # messages + tools + guidance + pressure warning
        resp = client.chat.completions.create(model=..., **req)
        calls = resp.choices[0].message.tool_calls or []
        ctx_results, other_calls = harness.handle_tool_calls(calls)   # context_* handled & governed
        ... run other_calls yourself, then harness.runtime.add(result, kind="tool_result") ...

``request()`` books one model request (``runtime.step()``), so budget enforcement,
fallback compaction and prefill-cost accounting happen exactly once per call. In
``guidance_only`` mode (for harnesses that cannot hand over context editing) no
context tools are exposed and the agent's autonomy is forced to L1: the pressure
warning and plan are surfaced as advice for the harness or user to act on.
"""

from __future__ import annotations

import json
from typing import Any

from ..governance.autonomy import AutonomyLevel
from ..runtime.runtime import ContextRuntime, StepReport
from ..skills import render_guidance
from .context_tools import ContextToolDispatcher, anthropic_tools, openai_tools


class ContextHarness:
    def __init__(self, runtime: ContextRuntime, *, provider: str = "openai", guidance: str | None = None,
                 brief_guidance: bool = True, guidance_only: bool = False, adaptations: tuple[str, ...] = ()) -> None:
        if provider not in ("openai", "anthropic"):
            raise ValueError("provider must be 'openai' or 'anthropic'")
        self.runtime = runtime
        self.provider = provider
        self.guidance_only = guidance_only
        if guidance_only:
            runtime.autonomy = AutonomyLevel.RECOMMEND
        self.guidance = guidance or render_guidance(policy=runtime.policy, autonomy=runtime.autonomy,
                                                    adaptations=adaptations, brief=brief_guidance)
        self.dispatcher = ContextToolDispatcher(runtime)
        self.last_step: StepReport | None = None

    def tools(self) -> list[dict[str, Any]]:
        if self.guidance_only:
            return []
        return openai_tools() if self.provider == "openai" else anthropic_tools()

    def request(self, *, extra_tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Build the next model request from the managed context."""
        self.last_step = self.runtime.step()
        rendered = self.runtime.to_messages(self.provider)
        notice = self.last_step.warning
        if self.guidance_only and notice:
            plan = self.runtime.plan()
            notice += "\n[context advice for the harness] " + json.dumps([o.to_dict() for o in plan.ops])
        tools = self.tools() + list(extra_tools or [])
        if self.provider == "openai":
            msgs = list(rendered)
            sys_text = self.guidance + ("\n\n" + msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else "")
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = {"role": "system", "content": sys_text}
            else:
                msgs.insert(0, {"role": "system", "content": sys_text})
            if notice:
                msgs.append({"role": "user", "content": notice})
            req: dict[str, Any] = {"messages": msgs}
        else:
            system = self.guidance + ("\n\n" + rendered["system"] if rendered["system"] else "")
            msgs = list(rendered["messages"])
            if notice:
                if msgs and msgs[-1]["role"] == "user":
                    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n\n" + notice}
                else:
                    msgs.append({"role": "user", "content": notice})
            req = {"system": system, "messages": msgs}
        if tools:
            req["tools"] = tools
        return req

    def handle_tool_calls(self, calls: list[Any]) -> tuple[list[dict[str, Any]], list[Any]]:
        """Execute ``context_*`` calls (OpenAI tool_call objects/dicts or Anthropic tool_use
        blocks/dicts). Returns (results for context calls, the untouched other calls)."""
        results: list[dict[str, Any]] = []
        others: list[Any] = []
        for c in calls:
            cid, name, args = _parse_call(c)
            if name is not None and self.dispatcher.handles(name):
                out = self.dispatcher.call(name, args)
                results.append({"id": cid, "name": name, "output": out})
            else:
                others.append(c)
        return results, others

    def tool_result_messages(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-format tool-result messages for ``results`` from :meth:`handle_tool_calls`."""
        if self.provider == "openai":
            return [{"role": "tool", "tool_call_id": r["id"], "content": json.dumps(r["output"], default=str)}
                    for r in results]
        return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": r["id"],
                                              "content": json.dumps(r["output"], default=str)} for r in results]}]


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _parse_call(c: Any) -> tuple[str | None, str | None, Any]:
    fn = _get(c, "function")
    if fn is not None:  # OpenAI
        return _get(c, "id"), _get(fn, "name"), _get(fn, "arguments") or "{}"
    if _get(c, "type") == "tool_use" or _get(c, "input") is not None:  # Anthropic
        return _get(c, "id"), _get(c, "name"), _get(c, "input") or {}
    return _get(c, "id"), _get(c, "name"), _get(c, "arguments") or {}

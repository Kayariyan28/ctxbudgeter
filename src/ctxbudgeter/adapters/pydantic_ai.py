"""Adapter: CompiledPack → PydanticAI-friendly dict.

Returns a plain dict — no pydantic-ai SDK dependency. The dict has:
  - "system_prompt": str — concatenated system + non-chat content
  - "message_history": list[dict] — chat turns
  - "deps_context": dict[name -> {content, kind, priority, source}] — every included
    item, indexed by name; useful as `deps` for PydanticAI Agent.run() so tools can
    pull specific context by name without re-parsing the system prompt.
"""

from __future__ import annotations

from typing import Any

from ..compiler import CompiledPack


def to_pydantic_ai_deps(pack: CompiledPack) -> dict[str, Any]:
    """Convert a CompiledPack to a structured dict for PydanticAI agents.

    Returns a dict you can pass into Agent.run() as `deps`, with:
      - `system_prompt`: concatenated system + non-chat content
      - `message_history`: list of chat turns in chronological order
      - `deps_context`: name → {content, kind, priority, source} for tool lookup
    """
    chat_kinds = {"user_message", "assistant_message", "tool_result"}
    system_parts: list[str] = []
    history: list[dict] = []
    deps_context: dict[str, dict] = {}

    # Build deps_context in prompt order; system_parts in prompt order
    for it in pack.included_items:
        content = it.effective_content()
        deps_context[it.name] = {
            "content": content,
            "kind": it.kind,
            "priority": it.priority,
            "cache_policy": it.cache_policy,
            "source": it.source,
        }
        if it.kind in chat_kinds:
            continue
        if it.kind == "system":
            system_parts.append(content)
        else:
            system_parts.append(f"## {it.name}\n{content}")

    # Chat history in insertion order
    for it in pack.included_in_input_order():
        if it.kind not in chat_kinds:
            continue
        content = it.effective_content()
        if it.kind == "user_message":
            history.append({"role": "user", "content": content})
        elif it.kind == "assistant_message":
            history.append({"role": "model", "content": content})
        elif it.kind == "tool_result":
            history.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": it.metadata.get("tool_call_id", it.name),
                }
            )

    return {
        "system_prompt": "\n\n".join(system_parts),
        "message_history": history,
        "deps_context": deps_context,
    }


def to_pydantic_ai(pack: CompiledPack) -> dict[str, Any]:
    """Deprecated alias for :func:`to_pydantic_ai_deps`. Kept for backward compatibility."""
    return to_pydantic_ai_deps(pack)

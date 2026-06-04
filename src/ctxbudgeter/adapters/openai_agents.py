"""Adapter: CompiledPack → OpenAI Agents SDK input.

Returns a plain dict shaped for the OpenAI Agents SDK: ``instructions`` (the
system/stable text) and ``input`` (the conversation turns as messages). No SDK
dependency — pure dicts so it works whether or not ``openai-agents`` is installed.
"""

from __future__ import annotations

from ..compiler import CompiledPack
from .openai import CHAT_KINDS, to_openai_messages


def to_openai_agents_input(pack: CompiledPack, *, user_message: str | None = None) -> dict:
    """Return ``{"instructions": str, "input": list[dict]}`` for Agents SDK runs."""
    system_parts: list[str] = []
    for it in pack.included_items:
        if it.kind in CHAT_KINDS:
            continue
        content = it.effective_content()
        system_parts.append(content if it.kind == "system" else f"## {it.name}\n{content}")

    messages = to_openai_messages(pack, user_message=user_message)
    # Strip system messages out of `input`; they belong in `instructions`.
    input_messages = [m for m in messages if m.get("role") != "system"]

    return {
        "instructions": "\n\n".join(system_parts),
        "input": input_messages,
    }

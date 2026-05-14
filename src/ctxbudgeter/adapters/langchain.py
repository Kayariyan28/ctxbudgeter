"""Adapter: CompiledPack → langchain_core BaseMessage list.

Requires `langchain-core` (install via `pip install ctxbudgeter[langchain]`).
"""

from __future__ import annotations

from typing import Any

from ..compiler import CompiledPack


def to_langchain_messages(
    pack: CompiledPack,
    *,
    user_message: str | None = None,
) -> list[Any]:
    """Convert a CompiledPack into a list of langchain_core message objects.

    Returns a list of SystemMessage / HumanMessage / AIMessage / ToolMessage.

    Raises ImportError if langchain-core is not installed.
    """
    try:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "langchain-core is required for the langchain adapter. "
            "Install via `pip install ctxbudgeter[langchain]`."
        ) from e

    chat_kinds = {"user_message", "assistant_message", "tool_result"}
    system_parts: list[str] = []
    chat: list[Any] = []

    for it in pack.included_items:
        if it.kind in chat_kinds:
            continue
        content = it.effective_content()
        if it.kind == "system":
            system_parts.append(content)
        else:
            system_parts.append(f"## {it.name}\n{content}")

    for it in pack.included_in_input_order():
        if it.kind not in chat_kinds:
            continue
        content = it.effective_content()
        if it.kind == "user_message":
            chat.append(HumanMessage(content=content))
        elif it.kind == "assistant_message":
            chat.append(AIMessage(content=content))
        elif it.kind == "tool_result":
            tool_call_id = str(it.metadata.get("tool_call_id", it.name))
            chat.append(ToolMessage(content=content, tool_call_id=tool_call_id))

    out: list[Any] = []
    if system_parts:
        out.append(SystemMessage(content="\n\n".join(system_parts)))
    out.extend(chat)
    if user_message is not None:
        out.append(HumanMessage(content=user_message))
    return out

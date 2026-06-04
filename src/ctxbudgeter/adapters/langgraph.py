"""Adapter: CompiledPack → LangGraph state dict.

LangGraph state is typically a dict with a ``messages`` key (plus whatever extra
channels your graph declares). This returns a plain dict you can merge into your
initial state: ``messages`` (chat turns), ``system`` (concatenated stable context),
and ``context_items`` (a name→metadata map for tools/nodes to introspect).

No langgraph dependency — pure dicts.
"""

from __future__ import annotations

from ..compiler import CompiledPack
from .openai import CHAT_KINDS, to_openai_messages


def to_langgraph_state(pack: CompiledPack, *, user_message: str | None = None) -> dict:
    """Return a dict suitable for seeding LangGraph state."""
    system_parts: list[str] = []
    context_items: dict[str, dict] = {}
    for it in pack.included_items:
        content = it.effective_content()
        context_items[it.name] = {
            "kind": it.kind,
            "tokens": pack.included_tokens.get(it.name, 0),
            "cache_policy": it.cache_policy,
            "source": it.source,
            "priority": it.priority,
        }
        if it.kind not in CHAT_KINDS:
            system_parts.append(content if it.kind == "system" else f"## {it.name}\n{content}")

    messages = to_openai_messages(pack, user_message=user_message)
    return {
        "messages": messages,
        "system": "\n\n".join(system_parts),
        "context_items": context_items,
    }

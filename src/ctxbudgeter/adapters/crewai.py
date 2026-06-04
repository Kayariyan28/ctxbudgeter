"""Adapter: CompiledPack → CrewAI-friendly context.

CrewAI tasks accept a free-form ``context`` string and agents have a ``backstory``
/ system framing. This returns a plain dict: ``context`` (the compiled stable text,
ready to pass to a Task), ``messages`` (chat turns), and ``items`` (name→metadata).

No crewai dependency — pure dicts/strings.
"""

from __future__ import annotations

from ..compiler import CompiledPack
from .openai import CHAT_KINDS


def to_crewai_context(pack: CompiledPack) -> dict:
    """Return ``{"context": str, "messages": list, "items": dict}`` for CrewAI."""
    parts: list[str] = []
    messages: list[dict] = []
    items: dict[str, dict] = {}

    for it in pack.included_items:
        content = it.effective_content()
        items[it.name] = {
            "kind": it.kind,
            "tokens": pack.included_tokens.get(it.name, 0),
            "source": it.source,
            "priority": it.priority,
        }
        if it.kind in CHAT_KINDS:
            role = "user" if it.kind == "user_message" else (
                "assistant" if it.kind == "assistant_message" else "tool"
            )
            messages.append({"role": role, "content": content})
        else:
            header = "System" if it.kind == "system" else it.name
            parts.append(f"## {header}\n{content}")

    return {
        "context": "\n\n".join(parts),
        "messages": messages,
        "items": items,
    }

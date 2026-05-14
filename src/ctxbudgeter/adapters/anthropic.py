"""Adapter: CompiledPack → Anthropic Messages API format with prompt-caching hints.

Returns plain dicts — no anthropic SDK dependency. Cache breakpoints are placed on
the LAST stable system block, since Anthropic's prompt cache key covers everything
up to and including the marked block. That gives you the longest possible cacheable
prefix from a single breakpoint.

`to_anthropic_request(pack, ...)` wraps the messages payload with `model` and
`max_tokens` so it slots straight into `client.messages.create(**kwargs)`.
"""

from __future__ import annotations

import json
from typing import Optional

from ..compiler import CompiledPack
from ..content import ImageBlock, StructuredBlock, TextBlock

CHAT_KINDS = {"user_message", "assistant_message", "tool_result"}
MAX_CACHE_BREAKPOINTS = 4


def _attachments_to_blocks(content: str, attachments: list) -> list[dict]:
    """Return a list of Anthropic content blocks for a chat message."""
    blocks: list[dict] = []
    if content:
        blocks.append({"type": "text", "text": content})
    for a in attachments:
        if isinstance(a, TextBlock):
            blocks.append({"type": "text", "text": a.text})
        elif isinstance(a, ImageBlock):
            if a.base64:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": a.media_type or "image/png",
                            "data": a.base64,
                        },
                    }
                )
            elif a.url:
                blocks.append({"type": "image", "source": {"type": "url", "url": a.url}})
        elif isinstance(a, StructuredBlock):
            blocks.append({"type": "text", "text": json.dumps(a.data, sort_keys=True, default=str)})
    return blocks


def to_anthropic_messages(
    pack: CompiledPack,
    *,
    user_message: Optional[str] = None,
) -> dict:
    """Convert a CompiledPack to Anthropic Messages API input."""
    system_blocks: list[dict] = []
    chat: list[dict] = []

    system_items = [it for it in pack.included_items if it.kind not in CHAT_KINDS]
    last_stable_idx = -1
    for i, it in enumerate(system_items):
        if it.cache_policy == "stable":
            last_stable_idx = i
    for i, it in enumerate(system_items):
        content = it.effective_content()
        text = content if it.kind == "system" else f"## {it.name}\n{content}"
        block: dict = {"type": "text", "text": text}
        if i == last_stable_idx and last_stable_idx >= 0:
            block["cache_control"] = {"type": "ephemeral"}
        system_blocks.append(block)
        # Attachments on system-attached items become extra text/image blocks
        for extra in _attachments_to_blocks("", it.attachments):
            system_blocks.append(extra)

    for it in pack.included_in_input_order():
        if it.kind not in CHAT_KINDS:
            continue
        content = it.effective_content()
        if it.kind == "user_message":
            blocks = _attachments_to_blocks(content, it.attachments)
            chat.append({"role": "user", "content": blocks if it.attachments else content})
        elif it.kind == "assistant_message":
            blocks = _attachments_to_blocks(content, it.attachments)
            chat.append({"role": "assistant", "content": blocks if it.attachments else content})
        elif it.kind == "tool_result":
            tool_use_id = str(it.metadata.get("tool_use_id", it.name))
            chat.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": content,
                        }
                    ],
                }
            )

    if user_message is not None:
        chat.append({"role": "user", "content": user_message})

    if not chat:
        chat.append({"role": "user", "content": ""})
    elif chat[0]["role"] != "user":
        chat.insert(0, {"role": "user", "content": ""})

    return {"system": system_blocks, "messages": chat}


def to_anthropic_request(
    pack: CompiledPack,
    *,
    user_message: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """Return a kwargs dict for ``client.messages.create(**kwargs)``."""
    payload = to_anthropic_messages(pack, user_message=user_message)
    return {
        "model": pack.model,
        "max_tokens": max_tokens or pack.reserved_output_tokens,
        "system": payload["system"],
        "messages": payload["messages"],
    }

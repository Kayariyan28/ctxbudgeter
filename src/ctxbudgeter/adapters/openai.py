"""Adapter: CompiledPack → OpenAI Chat Completions / Responses message format.

No openai SDK dependency — returns plain dicts. Compatible with both the legacy
Chat Completions schema and the newer Responses API input format.

`to_openai_request(pack, user_message=...)` returns the full kwargs dict including
a deterministic `prompt_cache_key` derived from the stable prefix — wire it
straight into `client.chat.completions.create(**kwargs)` for explicit cache routing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from ..compiler import CompiledPack
from ..content import ImageBlock, StructuredBlock, TextBlock

CHAT_KINDS = {"user_message", "assistant_message", "tool_result"}


def stable_prefix_cache_key(pack: CompiledPack) -> Optional[str]:
    """Hash the stable-cache prefix into a deterministic cache key.

    Returns None if there's no stable prefix (no point setting a cache key for an
    empty prefix). The key is stable across runs as long as the stable items'
    content doesn't change — which is exactly what prompt caching wants.
    """
    parts: list[str] = [pack.model]
    for it in pack.included_items:
        if it.cache_policy != "stable":
            break
        parts.append(it.name)
        parts.append(it.effective_content())
    if len(parts) == 1:
        return None
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"ctxbudgeter-{digest[:24]}"


def _content_for_message(content: str, attachments: list) -> object:
    """If there are attachments, return a list-of-parts OpenAI multimodal payload.
    Otherwise return the plain text string."""
    if not attachments:
        return content
    parts: list[dict] = []
    if content:
        parts.append({"type": "text", "text": content})
    for a in attachments:
        if isinstance(a, TextBlock):
            parts.append({"type": "text", "text": a.text})
        elif isinstance(a, ImageBlock):
            url = a.url or (f"data:{a.media_type or 'image/png'};base64,{a.base64}" if a.base64 else None)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url, "detail": a.detail}})
        elif isinstance(a, StructuredBlock):
            parts.append({"type": "text", "text": json.dumps(a.data, sort_keys=True, default=str)})
    return parts


def to_openai_messages(
    pack: CompiledPack,
    *,
    user_message: Optional[str] = None,
    merge_system: bool = True,
) -> list[dict]:
    """Convert a CompiledPack to a list of OpenAI chat messages."""
    system_parts: list[str] = []
    chat: list[dict] = []

    for it in pack.included_items:
        if it.kind in CHAT_KINDS:
            continue
        content = it.effective_content()
        if it.kind == "system":
            system_parts.append(content)
        else:
            system_parts.append(f"## {it.name}\n{content}")

    for it in pack.included_in_input_order():
        if it.kind not in CHAT_KINDS:
            continue
        content = it.effective_content()
        payload = _content_for_message(content, it.attachments)
        if it.kind == "user_message":
            chat.append({"role": "user", "content": payload})
        elif it.kind == "assistant_message":
            chat.append({"role": "assistant", "content": payload})
        elif it.kind == "tool_result":
            tool_call_id = str(it.metadata.get("tool_call_id", it.name))
            chat.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    out: list[dict] = []
    if system_parts:
        if merge_system:
            out.append({"role": "system", "content": "\n\n".join(system_parts)})
        else:
            out.extend({"role": "system", "content": p} for p in system_parts)
    out.extend(chat)
    if user_message is not None:
        out.append({"role": "user", "content": user_message})
    return out


def to_openai_request(
    pack: CompiledPack,
    *,
    user_message: Optional[str] = None,
    merge_system: bool = True,
    max_completion_tokens: Optional[int] = None,
    set_cache_key: bool = True,
) -> dict:
    """Return a kwargs dict suitable for ``client.chat.completions.create(**kwargs)``.

    Includes `model`, `messages`, `max_completion_tokens` (default: pack.reserved_output_tokens),
    and `prompt_cache_key` (default: hash of stable prefix). Drop directly into the
    OpenAI client for explicit, cache-aware calls.
    """
    kwargs: dict = {
        "model": pack.model,
        "messages": to_openai_messages(
            pack, user_message=user_message, merge_system=merge_system
        ),
        "max_completion_tokens": max_completion_tokens or pack.reserved_output_tokens,
    }
    if set_cache_key:
        key = stable_prefix_cache_key(pack)
        if key:
            kwargs["prompt_cache_key"] = key
    return kwargs

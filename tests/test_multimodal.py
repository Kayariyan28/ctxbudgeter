"""Tests for multimodal attachments."""

from __future__ import annotations

from ctxbudgeter import ContextPack, ImageBlock, StructuredBlock, TextBlock
from ctxbudgeter.adapters import to_anthropic_messages, to_openai_messages


def test_image_attachment_token_cost_included() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="describe image", required=True,
             attachments=[ImageBlock(url="https://example.com/x.png", estimated_tokens=200)])
    compiled = pack.compile()
    decision = next(d for d in compiled.decisions if d.name == "task")
    # original_tokens should reflect attachment cost
    assert decision.original_tokens >= 200


def test_structured_attachment_serialized_to_json() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="tool_schemas", content="", required=True,
             attachments=[StructuredBlock(schema_name="search", data={"name": "search_db", "args": ["query"]})])
    compiled = pack.compile()
    assert compiled.used_tokens > 0


def test_openai_image_attachment_emitted_as_image_url() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="screenshot", content="What's in this image?", kind="user_message",
             attachments=[ImageBlock(url="https://example.com/x.png")])
    msgs = to_openai_messages(pack.compile())
    user_msg = next(m for m in msgs if m["role"] == "user")
    parts = user_msg["content"]
    assert isinstance(parts, list)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "https://example.com/x.png"


def test_anthropic_image_attachment_emitted_as_image_block() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="screenshot", content="What's in this image?", kind="user_message",
             attachments=[ImageBlock(base64="ZmFrZQ==", media_type="image/png")])
    payload = to_anthropic_messages(pack.compile())
    user_msg = next(m for m in payload["messages"] if m["role"] == "user")
    assert isinstance(user_msg["content"], list)
    image_blocks = [b for b in user_msg["content"] if b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["data"] == "ZmFrZQ=="


def test_text_attachment_works() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(
        name="extra",
        content="primary",
        kind="user_message",
        attachments=[TextBlock(text="extra block content")],
    )
    msgs = to_openai_messages(pack.compile())
    user_msg = next(m for m in msgs if m["role"] == "user")
    assert isinstance(user_msg["content"], list)
    text_parts = [p for p in user_msg["content"] if p["type"] == "text"]
    assert any("extra block" in p["text"] for p in text_parts)


def test_no_attachments_keeps_string_content() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="msg", content="just text", kind="user_message")
    msgs = to_openai_messages(pack.compile())
    user_msg = next(m for m in msgs if m["role"] == "user")
    # No attachments → content stays as string for backward compat
    assert user_msg["content"] == "just text"

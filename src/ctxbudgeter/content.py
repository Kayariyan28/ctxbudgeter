"""Multi-modal content blocks.

A ContextItem's primary `content` is text, but it can carry optional `attachments`
(images, structured tool schemas, audio refs). Each block declares its own token
cost since the tokenizer can't count them. Adapters serialize them per-vendor.

Tokenization rule: text content is tokenized by the model's tokenizer; attachments
contribute `estimated_tokens` flat. The compiler treats attachment tokens as
non-compressible cost (you don't shrink an image by changing the prompt).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TextBlock(BaseModel):
    """A plain text block. Useful when you want to keep the primary content empty
    and put everything into attachments — rare, but supported."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content of this block.")
    estimated_tokens: int = Field(0, ge=0, description="Override tokenizer estimate (0 = let the tokenizer count).")


class ImageBlock(BaseModel):
    """An image reference (URL or base64). Estimated tokens default to 85 — a
    rough Anthropic/OpenAI baseline for small images. Supply your own when known."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["image"] = "image"
    url: Optional[str] = Field(None, description="Public URL (http/https) or data URL.")
    base64: Optional[str] = Field(None, description="Base64-encoded image bytes (without data: prefix).")
    media_type: Optional[str] = Field(None, description='MIME type, e.g. "image/png".')
    detail: Literal["auto", "low", "high"] = Field("auto", description="Vendor-specific detail hint.")
    estimated_tokens: int = Field(85, ge=0, description="Estimated token cost.")


class StructuredBlock(BaseModel):
    """Structured data (tool schema, JSON config, function call args). Serialized
    to JSON for tokenization and adapter output."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["structured"] = "structured"
    schema_name: Optional[str] = Field(None, description='Optional name, e.g. tool name "search_db".')
    data: dict[str, Any] = Field(default_factory=dict)
    estimated_tokens: int = Field(0, ge=0, description="If 0, tokenizer counts the JSON-serialized form.")


Attachment = Annotated[
    Union[TextBlock, ImageBlock, StructuredBlock],
    Field(discriminator="type"),
]
"""Discriminated union of attachment block types. Use the `type` field to dispatch."""


def attachment_estimated_tokens(att: Attachment, *, tokenizer_count: Optional[callable] = None) -> int:
    """Return the budgeted token cost for a single attachment.

    For TextBlock: prefers the user override; otherwise asks tokenizer_count.
    For ImageBlock: uses estimated_tokens (tokenizer can't count images).
    For StructuredBlock: prefers override; otherwise tokenizer_count on JSON form.
    """
    if isinstance(att, ImageBlock):
        return att.estimated_tokens
    if att.estimated_tokens:
        return att.estimated_tokens
    if tokenizer_count is None:
        return 0
    if isinstance(att, TextBlock):
        return tokenizer_count(att.text)
    if isinstance(att, StructuredBlock):
        import json

        return tokenizer_count(json.dumps(att.data, sort_keys=True, default=str))
    return 0

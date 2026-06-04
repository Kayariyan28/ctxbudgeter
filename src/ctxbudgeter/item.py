"""Typed model for a single piece of context.

A ContextItem is a unit of information you might want to give an LLM: a system rule,
a doc, a code file, a memory note, a tool result, etc. The compiler decides what
makes it into the final prompt based on these fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .content import Attachment
from .provenance import ContextProvenance

ContextKind = Literal[
    "system",
    "task",
    "user_message",
    "assistant_message",
    "tool_def",
    "tool_result",
    "project_doc",
    "code",
    "memory",
    "retrieval",
    "example",
    "data",
    "other",
]
"""Semantic category of a context item.

- system: system rules / persona / non-negotiable instructions
- task: the current task / user request (often required)
- user_message / assistant_message: chat history turns
- tool_def: tool/function schemas the model can call
- tool_result: output of a tool invocation
- project_doc: README, CONTRIBUTING, architecture docs
- code: source code files
- memory: long-term notes about the user/project
- retrieval: chunks pulled from a vector store / search
- example: few-shot examples
- data: structured data (JSON, CSV, configs)
- other: anything else
"""

CachePolicy = Literal["stable", "dynamic", "ephemeral"]
"""How likely this content is to change request-to-request.

- stable: rarely changes (system prompt, docs, tool defs) — eligible for prompt caching
- dynamic: changes per request (user message, recent state)
- ephemeral: short-lived (latest tool result) — should not go in cacheable prefix
"""

Sensitivity = Literal["public", "internal", "secret"]
"""Sensitivity tag. The compiler does not redact; it surfaces this in reports so
callers can audit what's about to be sent to an external model."""


class ContextItem(BaseModel):
    """A single, addressable piece of context."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )

    name: str = Field(..., min_length=1, description="Unique name within a pack.")
    content: str = Field(..., description="Raw text content.")
    kind: ContextKind = Field("other", description="Semantic kind.")
    priority: int = Field(
        50, ge=0, le=100, description="User priority 0-100 (higher = more important)."
    )
    required: bool = Field(
        False, description="If true, the compiler MUST include this item (or raise)."
    )
    freshness: float = Field(
        1.0, ge=0.0, le=1.0, description="0-1 freshness score; 1 = brand new."
    )
    relevance: float = Field(
        0.5, ge=0.0, le=1.0, description="0-1 relevance to current task (default 0.5)."
    )
    source: str | None = Field(
        None, description="Where the content came from (file path, URL, etc.)."
    )
    cache_policy: CachePolicy = Field(
        "dynamic", description="Prompt-cache hint: stable | dynamic | ephemeral."
    )
    sensitivity: Sensitivity = Field(
        "internal", description="Sensitivity classification."
    )
    compressible: bool = Field(
        False, description="May be compressed if it doesn't fit the budget."
    )
    compressed_content: str | None = Field(
        None, description="Optional pre-computed compressed form, used if original won't fit."
    )
    attachments: list[Attachment] = Field(
        default_factory=list,
        description="Optional multi-modal blocks (images, structured tool schemas).",
    )
    provenance: ContextProvenance | None = Field(
        default=None,
        description="Optional provenance metadata (source, trust, age, transformations).",
    )
    trust_level: str | None = Field(
        default=None,
        description="Convenience trust override: unknown | low | internal | verified. "
        "If set and no provenance object exists, a minimal one is derived at access time.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="User-defined metadata; not interpreted by the compiler."
    )

    def effective_provenance(self) -> ContextProvenance | None:
        """Return provenance, synthesizing a minimal record from source/trust_level
        if no explicit provenance object was supplied."""
        if self.provenance is not None:
            return self.provenance
        if self.source is not None or self.trust_level is not None:
            return ContextProvenance(
                source=self.source,
                trust_level=self.trust_level or "unknown",  # type: ignore[arg-type]
            )
        return None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name cannot be blank or whitespace")
        return v

    def effective_content(self) -> str:
        """Return the content the compiler would use if compression were applied."""
        return self.compressed_content if self.compressed_content is not None else self.content

    def has_attachments(self) -> bool:
        return bool(self.attachments)

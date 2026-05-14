"""Framework adapters. Convert a CompiledPack into the message format of a target SDK.

Each adapter is independent — you only pay the import cost for the one you use, and
optional SDKs (langchain-core, anthropic, openai) are only imported lazily inside the
function that needs them.
"""

from __future__ import annotations

from .anthropic import to_anthropic_messages, to_anthropic_request
from .langchain import to_langchain_messages
from .openai import stable_prefix_cache_key, to_openai_messages, to_openai_request
from .pydantic_ai import to_pydantic_ai, to_pydantic_ai_deps

__all__ = [
    "to_openai_messages",
    "to_openai_request",
    "stable_prefix_cache_key",
    "to_anthropic_messages",
    "to_anthropic_request",
    "to_langchain_messages",
    "to_pydantic_ai_deps",
    "to_pydantic_ai",  # deprecated alias
]

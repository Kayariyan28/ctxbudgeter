"""Framework adapters. Convert a CompiledPack into the message format of a target SDK.

Each adapter is independent — you only pay the import cost for the one you use, and
optional SDKs (langchain-core, anthropic, openai) are only imported lazily inside the
function that needs them.
"""

from __future__ import annotations

from .anthropic import to_anthropic_messages, to_anthropic_request
from .crewai import to_crewai_context
from .langchain import to_langchain_messages
from .langgraph import to_langgraph_state
from .openai import stable_prefix_cache_key, to_openai_messages, to_openai_request
from .openai_agents import to_openai_agents_input
from .pydantic_ai import to_pydantic_ai, to_pydantic_ai_deps

__all__ = [
    "stable_prefix_cache_key",
    "to_anthropic_messages",
    "to_anthropic_request",
    "to_crewai_context",
    "to_langchain_messages",
    "to_langgraph_state",
    "to_openai_agents_input",
    "to_openai_messages",
    "to_openai_request",
    "to_pydantic_ai",  # deprecated alias
    "to_pydantic_ai_deps",
]

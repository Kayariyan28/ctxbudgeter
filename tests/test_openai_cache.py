"""Tests for OpenAI cache_key and request adapter."""

from __future__ import annotations

from ctxbudgeter import ContextPack
from ctxbudgeter.adapters.openai import (
    stable_prefix_cache_key,
    to_openai_request,
)


def _pack_with_stable() -> ContextPack:
    pack = ContextPack(model="gpt-4o", token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    pack.add(name="doc", content="docs", kind="project_doc", priority=80, cache_policy="stable")
    pack.add(name="task", content="task", kind="task", required=True)
    return pack


def test_stable_prefix_cache_key_deterministic() -> None:
    k1 = stable_prefix_cache_key(_pack_with_stable().compile())
    k2 = stable_prefix_cache_key(_pack_with_stable().compile())
    assert k1 == k2
    assert k1.startswith("ctxbudgeter-")


def test_stable_prefix_cache_key_changes_with_content() -> None:
    p1 = _pack_with_stable()
    compiled1 = p1.compile()
    p2 = ContextPack(model="gpt-4o", token_budget=10_000, reserved_output_tokens=1_000)
    p2.add(name="sys", content="DIFFERENT rules", kind="system", required=True, cache_policy="stable")
    p2.add(name="task", content="task", kind="task", required=True)
    compiled2 = p2.compile()
    assert stable_prefix_cache_key(compiled1) != stable_prefix_cache_key(compiled2)


def test_stable_prefix_cache_key_none_without_stable_items() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True, cache_policy="dynamic")
    assert stable_prefix_cache_key(pack.compile()) is None


def test_to_openai_request_includes_cache_key() -> None:
    kwargs = to_openai_request(_pack_with_stable().compile())
    assert "prompt_cache_key" in kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["messages"]
    assert kwargs["max_completion_tokens"] == 1_000


def test_to_openai_request_can_disable_cache_key() -> None:
    kwargs = to_openai_request(_pack_with_stable().compile(), set_cache_key=False)
    assert "prompt_cache_key" not in kwargs


def test_to_openai_request_user_message_appended() -> None:
    kwargs = to_openai_request(_pack_with_stable().compile(), user_message="hello")
    last = kwargs["messages"][-1]
    assert last == {"role": "user", "content": "hello"}

"""Tests for framework adapters."""

from __future__ import annotations

from ctxbudgeter import ContextPack
from ctxbudgeter.adapters import (
    to_anthropic_messages,
    to_openai_messages,
    to_pydantic_ai,
)


def _build_pack() -> ContextPack:
    pack = ContextPack(model="claude-sonnet-4.6", token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="You are an agent.", kind="system", priority=100, cache_policy="stable", required=True)
    pack.add(name="doc", content="# README\nHello.", kind="project_doc", priority=80, cache_policy="stable")
    pack.add(name="task", content="Fix the bug.", kind="task", priority=95, required=True)
    pack.add(name="prior_user", content="Previous question", kind="user_message", priority=70)
    pack.add(name="prior_asst", content="Previous answer", kind="assistant_message", priority=70)
    return pack


def test_openai_messages_shape() -> None:
    compiled = _build_pack().compile()
    msgs = to_openai_messages(compiled)
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert "user" in roles
    assert "assistant" in roles
    # System should contain the doc + task folded in
    sys_content = msgs[0]["content"]
    assert "agent" in sys_content.lower()


def test_openai_messages_appends_user_message() -> None:
    compiled = _build_pack().compile()
    msgs = to_openai_messages(compiled, user_message="What now?")
    assert msgs[-1] == {"role": "user", "content": "What now?"}


def test_openai_unmerged_system_emits_multiple_system_messages() -> None:
    compiled = _build_pack().compile()
    msgs = to_openai_messages(compiled, merge_system=False)
    sys_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(sys_msgs) >= 2  # system_rules + folded doc/task


def test_anthropic_returns_system_and_messages() -> None:
    compiled = _build_pack().compile()
    payload = to_anthropic_messages(compiled)
    assert "system" in payload and "messages" in payload
    assert isinstance(payload["system"], list)
    assert all(b.get("type") == "text" for b in payload["system"])
    assert payload["messages"], "messages must be non-empty"


def test_anthropic_cache_control_on_last_stable_block_only() -> None:
    compiled = _build_pack().compile()
    payload = to_anthropic_messages(compiled)
    blocks_with_cache = [b for b in payload["system"] if "cache_control" in b]
    assert len(blocks_with_cache) == 1, "exactly one cache_control breakpoint expected"
    assert blocks_with_cache[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_no_cache_control_when_no_stable_items() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="x", required=True, cache_policy="dynamic")
    payload = to_anthropic_messages(pack.compile())
    assert all("cache_control" not in b for b in payload["system"])


def test_anthropic_user_message_appended() -> None:
    compiled = _build_pack().compile()
    payload = to_anthropic_messages(compiled, user_message="Q?")
    assert payload["messages"][-1] == {"role": "user", "content": "Q?"}


def test_anthropic_adds_dummy_user_when_none() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    payload = to_anthropic_messages(pack.compile())
    assert payload["messages"][-1]["role"] == "user"


def test_pydantic_ai_dict_shape() -> None:
    compiled = _build_pack().compile()
    out = to_pydantic_ai(compiled)
    assert set(out.keys()) == {"system_prompt", "message_history", "deps_context"}
    assert "agent" in out["system_prompt"].lower()
    assert any(m["role"] == "user" for m in out["message_history"])
    assert any(m["role"] == "model" for m in out["message_history"])
    assert "sys" in out["deps_context"]
    assert out["deps_context"]["sys"]["kind"] == "system"


def test_chat_turns_preserve_insertion_order() -> None:
    """Conversation history must follow insertion (chronological) order, not the
    cache-aware prompt sort. Otherwise user/assistant turns get scrambled."""
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    # Insertion order: user first, then assistant — chronological
    pack.add(name="turn_user", content="Q1", kind="user_message", priority=70)
    pack.add(name="turn_asst", content="A1", kind="assistant_message", priority=70)
    pack.add(name="turn_user2", content="Q2", kind="user_message", priority=70)
    pack.add(name="turn_asst2", content="A2", kind="assistant_message", priority=70)
    compiled = pack.compile()

    o = to_openai_messages(compiled)
    chat = [m for m in o if m["role"] in ("user", "assistant")]
    assert [m["content"] for m in chat] == ["Q1", "A1", "Q2", "A2"]

    a = to_anthropic_messages(compiled)
    # First message must be user (Anthropic requirement)
    assert a["messages"][0]["role"] == "user"
    contents = [m["content"] for m in a["messages"]]
    assert contents == ["Q1", "A1", "Q2", "A2"]

    p = to_pydantic_ai(compiled)
    assert [m["content"] for m in p["message_history"]] == ["Q1", "A1", "Q2", "A2"]


def test_anthropic_assistant_first_gets_user_prepended() -> None:
    """If chat history would start with assistant, prepend an empty user message
    so the Anthropic API accepts it."""
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    pack.add(name="asst_only", content="prior answer", kind="assistant_message")
    payload = to_anthropic_messages(pack.compile())
    assert payload["messages"][0]["role"] == "user"


def test_openai_tool_result_role_mapping() -> None:
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    pack.add(
        name="t1",
        content='{"result": "ok"}',
        kind="tool_result",
        priority=70,
        metadata={"tool_call_id": "call_123"},
    )
    msgs = to_openai_messages(pack.compile())
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_123"

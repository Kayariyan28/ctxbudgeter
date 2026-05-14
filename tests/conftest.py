"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import pytest

from ctxbudgeter import ContextPack


@pytest.fixture
def small_pack() -> ContextPack:
    """A small pack with one system rule and one task — good for adapter tests."""
    pack = ContextPack(
        model="claude-sonnet-4.6",
        token_budget=10_000,
        reserved_output_tokens=1_000,
    )
    pack.add(
        name="system_rules",
        content="You are a careful coding agent. Always reason before acting.",
        kind="system",
        priority=100,
        cache_policy="stable",
        required=True,
    )
    pack.add(
        name="task",
        content="Fix the auth bug in routes/login.ts.",
        kind="task",
        priority=95,
        required=True,
    )
    return pack

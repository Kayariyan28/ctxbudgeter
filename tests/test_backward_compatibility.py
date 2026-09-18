"""Backward-compatibility guarantees for the pre-0.3 public API.

These mirror the way existing users call ctxbudgeter. They must keep working
unchanged across the 0.3 enterprise upgrade.
"""

from __future__ import annotations

import re
from pathlib import Path

import ctxbudgeter
from ctxbudgeter import (
    CompiledPack,
    ContextItem,
    ContextPack,
    compiled_pack_from_dict,
)


def test_import_and_version() -> None:
    """`__version__` must be a release version and agree with `pyproject.toml`.

    A hard-coded literal only proved someone edited this line during a release. The
    invariant that actually matters is the one RELEASE_CHECKLIST calls out: keep
    `pyproject.toml` and `src/ctxbudgeter/__init__.py` in sync. Read pyproject from the
    source tree rather than installed metadata, which goes stale after a version bump
    under an editable install.
    """
    assert hasattr(ctxbudgeter, "__version__")
    assert re.fullmatch(
        r"\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|dev|post)\d*)?", ctxbudgeter.__version__
    ), f"not a release version: {ctxbudgeter.__version__!r}"

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():  # pragma: no cover — installed without the source tree
        return
    declared = re.search(
        r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert declared, "no version found in pyproject.toml"
    assert ctxbudgeter.__version__ == declared.group(1), (
        f"__init__.py says {ctxbudgeter.__version__}, pyproject.toml says "
        f"{declared.group(1)} — bump both"
    )


def test_legacy_pack_flow_unchanged() -> None:
    pack = ContextPack(model="claude-sonnet-4.6", token_budget=24_000, reserved_output_tokens=4_000)
    pack.add(name="system_rules", content="You are careful.", kind="system",
             priority=100, cache_policy="stable", required=True)
    pack.add(name="task", content="Do the thing.", kind="task", priority=95, required=True)
    compiled = pack.compile()  # no task arg — must still work
    assert isinstance(compiled, CompiledPack)
    assert compiled.used_tokens > 0
    assert "system_rules" in [it.name for it in compiled.included_items]


def test_compile_with_task_arg() -> None:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="task", content="x", required=True)
    compiled = pack.compile(task="my task")
    assert compiled.task == "my task"


def test_legacy_reports_unchanged() -> None:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="task", content="x", required=True)
    compiled = pack.compile()
    assert "Included:" in compiled.report("text")
    assert "# Context" in compiled.report("markdown")
    import json

    json.loads(compiled.report("json"))


def test_legacy_item_fields_intact() -> None:
    it = ContextItem(name="x", content="y")
    assert it.priority == 50
    assert it.cache_policy == "dynamic"
    assert it.sensitivity == "internal"
    # new optional fields default to None / empty
    assert it.provenance is None
    assert it.trust_level is None


def test_legacy_adapters_still_functions() -> None:
    from ctxbudgeter.adapters import to_anthropic_messages, to_openai_messages

    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="x", required=True)
    compiled = pack.compile()
    assert isinstance(to_openai_messages(compiled), list)
    assert "system" in to_anthropic_messages(compiled)


def test_legacy_round_trip() -> None:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
    pack.add(name="task", content="x", required=True)
    compiled = pack.compile()
    restored = compiled_pack_from_dict(compiled.to_dict())
    assert restored.model == compiled.model
    assert restored.health_score == compiled.health_score


def test_new_compiled_methods_present() -> None:
    pack = ContextPack(token_budget=10_000, reserved_output_tokens=1_000)
    pack.add(name="task", content="x", required=True)
    compiled = pack.compile()
    # New instance adapter methods
    assert hasattr(compiled, "to_openai_messages")
    assert hasattr(compiled, "to_langgraph_state")
    assert hasattr(compiled, "to_crewai_context")
    assert hasattr(compiled, "bom")

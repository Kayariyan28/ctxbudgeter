"""The portable context-management skill."""

from __future__ import annotations

import re

import pytest

from ctxbudgeter.adapters.context_tools import TOOL_NAMES
from ctxbudgeter.governance import ContextEditPolicy
from ctxbudgeter.skills import (
    ADAPTATIONS,
    export_skill,
    load_reference,
    load_skill,
    render_guidance,
    skill_dir,
    skill_metadata,
)


def test_skill_layout_and_frontmatter() -> None:
    d = skill_dir()
    assert (d / "SKILL.md").is_file()
    assert (d / "references" / "api.md").is_file() and (d / "references" / "examples.md").is_file()
    meta = skill_metadata()
    assert meta["name"] == "context-management"
    assert len(meta["description"]) > 200  # describes both what and when
    full = load_skill(include_frontmatter=True)
    assert full.startswith("---\nname: context-management\n")
    body = load_skill()
    assert not body.startswith("---") and len(body.splitlines()) < 500
    assert "references/api.md" in body and "references/examples.md" in body


def test_skill_documents_every_context_tool() -> None:
    api = load_reference("api")
    body = load_skill()
    for name in TOOL_NAMES:
        assert name in api or name in body, name


def test_render_guidance_includes_parameters_and_adaptations() -> None:
    pol = ContextEditPolicy(min_usage_to_compact=0.7, max_compression_ratio=4, min_fact_retention=0.6)
    text = render_guidance(policy=pol, autonomy=2, adaptations=["timing_instruction", "preserve_detail"], brief=True)
    assert "Floor: don't compact below 70%" in text
    assert "at most 4x" in text and "at least 60%" in text
    assert "Autonomy L2" in text
    assert "Timing: consider compacting only once usage reaches 85%" in text
    assert len(text) < len(load_skill())
    with pytest.raises(ValueError):
        render_guidance(adaptations=["made_up"])
    assert set(ADAPTATIONS) >= {"timing_instruction", "compaction_expected", "use_archive_search"}


def test_export_skill(tmp_path) -> None:
    target = export_skill(tmp_path)
    assert (target / "SKILL.md").read_text(encoding="utf-8") == load_skill(include_frontmatter=True)
    with pytest.raises(FileExistsError):
        export_skill(tmp_path)
    assert export_skill(tmp_path, overwrite=True) == target
    assert not list(target.rglob("*.pyc"))


def test_skill_names_valid_for_skill_loaders() -> None:
    assert re.fullmatch(r"[a-z0-9-]{1,64}", skill_metadata()["name"])

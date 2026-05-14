"""Tests for ContextItem validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ctxbudgeter import ContextItem


def test_minimal_item_ok() -> None:
    it = ContextItem(name="foo", content="hello")
    assert it.name == "foo"
    assert it.content == "hello"
    assert it.kind == "other"
    assert it.priority == 50
    assert it.required is False
    assert it.cache_policy == "dynamic"
    assert it.sensitivity == "internal"


def test_name_stripped() -> None:
    it = ContextItem(name="  foo  ", content="x")
    assert it.name == "foo"


def test_blank_name_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextItem(name="   ", content="x")


def test_priority_bounds() -> None:
    with pytest.raises(ValidationError):
        ContextItem(name="a", content="x", priority=-1)
    with pytest.raises(ValidationError):
        ContextItem(name="a", content="x", priority=101)


def test_freshness_relevance_bounds() -> None:
    with pytest.raises(ValidationError):
        ContextItem(name="a", content="x", freshness=1.5)
    with pytest.raises(ValidationError):
        ContextItem(name="a", content="x", relevance=-0.01)


def test_invalid_cache_policy_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextItem(name="a", content="x", cache_policy="permanent")  # type: ignore[arg-type]


def test_invalid_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextItem(name="a", content="x", kind="banana")  # type: ignore[arg-type]


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ContextItem(name="a", content="x", foo="bar")  # type: ignore[call-arg]


def test_effective_content_prefers_compressed() -> None:
    it = ContextItem(name="a", content="big text", compressed_content="small")
    assert it.effective_content() == "small"
    it2 = ContextItem(name="b", content="big text")
    assert it2.effective_content() == "big text"


def test_metadata_default_is_independent_dict() -> None:
    a = ContextItem(name="a", content="x")
    b = ContextItem(name="b", content="x")
    a.metadata["k"] = 1
    assert "k" not in b.metadata

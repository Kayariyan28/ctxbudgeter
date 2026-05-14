"""Tests for the memory store (Write strategy) and pack.add_memory."""

from __future__ import annotations

from pathlib import Path

from ctxbudgeter import ContextPack, InMemoryStore, JSONMemoryStore, MemoryNote


def _note(key: str, content: str, *tags: str) -> MemoryNote:
    return MemoryNote(key=key, content=content, tags=list(tags))


def test_in_memory_write_read_delete() -> None:
    store = InMemoryStore()
    store.write(_note("k1", "hello"))
    assert store.read("k1").content == "hello"
    assert store.delete("k1") is True
    assert store.read("k1") is None


def test_in_memory_query_by_tag_and_text() -> None:
    store = InMemoryStore()
    store.write(_note("a", "alpha note", "auth"))
    store.write(_note("b", "beta note", "ui"))
    store.write(_note("c", "alpha and ui", "ui", "auth"))

    by_tag = store.query(tags=["auth"])
    assert {n.key for n in by_tag} == {"a", "c"}

    by_text = store.query(text="alpha")
    assert {n.key for n in by_text} == {"a", "c"}


def test_json_store_persists(tmp_path: Path) -> None:
    p = tmp_path / "memory.json"
    s1 = JSONMemoryStore(p)
    s1.write(_note("persist_me", "hello world", "saved"))
    s2 = JSONMemoryStore(p)
    assert s2.read("persist_me") is not None
    assert s2.read("persist_me").content == "hello world"


def test_pack_add_memory_pulls_notes() -> None:
    store = InMemoryStore()
    store.write(_note("auth_notes", "JWT secret rotation runbook", "auth"))
    store.write(_note("ui_notes", "design system guidelines", "ui"))

    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="fix auth bug", required=True)
    added = pack.add_memory(store, tags=["auth"], priority=70)
    assert len(added) == 1
    assert added[0].name.startswith("memory_")
    assert "JWT" in added[0].content


def test_add_memory_skips_duplicates() -> None:
    store = InMemoryStore()
    store.write(_note("k", "v", "t"))
    pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
    pack.add(name="task", content="t", required=True)
    pack.add_memory(store, tags=["t"])
    pack.add_memory(store, tags=["t"])  # second call should not duplicate
    names = [it.name for it in pack.items]
    assert names.count("memory_k") == 1


def test_query_since_filter() -> None:
    import time
    store = InMemoryStore()
    old = MemoryNote(key="old", content="x", created_at=time.time() - 86_400)
    new = MemoryNote(key="new", content="y", created_at=time.time())
    store.write(old)
    store.write(new)
    recent = store.query(since=time.time() - 3600)
    assert {n.key for n in recent} == {"new"}


def test_memory_note_round_trip() -> None:
    n = _note("k", "c", "t1", "t2")
    d = n.to_dict()
    n2 = MemoryNote.from_dict(d)
    assert n2.key == n.key
    assert n2.content == n.content
    assert n2.tags == n.tags

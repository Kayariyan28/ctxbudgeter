"""Declarative pack specs — YAML/TOML/JSON → ContextPack.

A pack spec lets teams check context configuration into version control, review it
in PRs, and load it from a CLI without writing Python. Loaders referenced by name
are looked up in ``ctxbudgeter.loaders`` (safer than dotted-path import).

Example pack.yaml::

    model: claude-sonnet-4.6
    token_budget: 24000
    reserved_output_tokens: 4000
    secret_policy: warn

    items:
      - name: system_rules
        content: |
          You are a careful coding agent.
        kind: system
        priority: 100
        required: true
        cache_policy: stable

      - name: readme
        from_file: README.md
        kind: project_doc
        priority: 80
        cache_policy: stable

      - name: task
        content: "Fix the auth bug."
        kind: task
        priority: 95
        required: true

    references:
      - name: api_docs
        location: "https://example.com/api.json"
        loader: http_get
        estimated_tokens: 1500
        priority: 60
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .compiler import CompilerConfig
from .loaders import get_loader
from .pack import ContextPack


class SpecError(ValueError):
    """Raised when a pack spec file is malformed."""


_TOP_LEVEL_KEYS = {
    "model",
    "token_budget",
    "reserved_output_tokens",
    "secret_policy",
    "allow_truncation",
    "weights",
    "items",
    "references",
}

_ITEM_KEYS = {
    "name",
    "content",
    "from_file",
    "kind",
    "priority",
    "required",
    "freshness",
    "relevance",
    "source",
    "cache_policy",
    "sensitivity",
    "compressible",
    "compressed_content",
    "metadata",
}

_REF_KEYS = {
    "name",
    "location",
    "loader",
    "estimated_tokens",
    "kind",
    "priority",
    "required",
    "cache_policy",
    "sensitivity",
    "compressible",
    "relevance",
    "freshness",
    "metadata",
}


def load_pack(path: str | Path, *, base_dir: Optional[Path] = None) -> ContextPack:
    """Load a pack spec from a YAML, TOML, or JSON file."""
    p = Path(path)
    if not p.exists():
        raise SpecError(f"Pack spec file does not exist: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = _parse_yaml(text)
    elif suffix == ".toml":
        data = _parse_toml(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        # Try YAML first, fall back to JSON
        try:
            data = _parse_yaml(text)
        except Exception:
            data = json.loads(text)
    return build_pack(data, base_dir=base_dir or p.parent)


def build_pack(data: dict[str, Any], *, base_dir: Optional[Path] = None) -> ContextPack:
    """Build a ContextPack from a spec dict (already parsed from YAML/TOML/JSON)."""
    if not isinstance(data, dict):
        raise SpecError("Pack spec root must be a mapping/object")
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise SpecError(f"Unknown top-level keys: {sorted(unknown)}")
    base = base_dir or Path.cwd()

    pack = ContextPack(
        model=data.get("model", "claude-sonnet-4.6"),
        token_budget=int(data.get("token_budget", 200_000)),
        reserved_output_tokens=int(data.get("reserved_output_tokens", 4_000)),
    )

    # CompilerConfig overrides
    cfg = CompilerConfig(
        weights=dict(data.get("weights", {})),
        allow_truncation=bool(data.get("allow_truncation", False)),
        secret_policy=str(data.get("secret_policy", "warn")),  # type: ignore[arg-type]
    )
    pack.set_config(cfg)

    for raw in data.get("items", []) or []:
        _add_item(pack, raw, base)
    for raw in data.get("references", []) or []:
        _add_reference(pack, raw)

    return pack


def _add_item(pack: ContextPack, raw: dict[str, Any], base: Path) -> None:
    if not isinstance(raw, dict):
        raise SpecError(f"Each item must be a mapping; got {type(raw).__name__}")
    unknown = set(raw) - _ITEM_KEYS
    if unknown:
        raise SpecError(f"Unknown item keys for '{raw.get('name', '?')}': {sorted(unknown)}")
    name = raw.get("name")
    if not name:
        raise SpecError("item.name is required")
    if "from_file" in raw and "content" in raw:
        raise SpecError(f"item '{name}': set either content OR from_file, not both")
    if "from_file" in raw:
        file_path = (base / raw["from_file"]).resolve()
        if not file_path.exists():
            raise SpecError(f"item '{name}': from_file '{file_path}' does not exist")
        content = file_path.read_text(encoding="utf-8")
        source = str(file_path)
    else:
        content = str(raw.get("content", ""))
        source = raw.get("source")
    pack.add(
        name=str(name),
        content=content,
        kind=raw.get("kind", "other"),
        priority=int(raw.get("priority", 50)),
        required=bool(raw.get("required", False)),
        freshness=float(raw.get("freshness", 1.0)),
        relevance=float(raw.get("relevance", 0.5)),
        source=source,
        cache_policy=raw.get("cache_policy", "dynamic"),
        sensitivity=raw.get("sensitivity", "internal"),
        compressible=bool(raw.get("compressible", False)),
        compressed_content=raw.get("compressed_content"),
        metadata=raw.get("metadata", {}),
    )


def _add_reference(pack: ContextPack, raw: dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise SpecError(f"Each reference must be a mapping; got {type(raw).__name__}")
    unknown = set(raw) - _REF_KEYS
    if unknown:
        raise SpecError(f"Unknown reference keys for '{raw.get('name', '?')}': {sorted(unknown)}")
    name = raw.get("name")
    location = raw.get("location")
    if not name or location is None:
        raise SpecError("reference requires 'name' and 'location'")
    loader_name = raw.get("loader")
    loader = get_loader(loader_name) if loader_name else None
    pack.add_reference(
        name=str(name),
        location=str(location),
        loader=loader,
        estimated_tokens=int(raw.get("estimated_tokens", 0)),
        kind=raw.get("kind", "retrieval"),
        priority=int(raw.get("priority", 50)),
        required=bool(raw.get("required", False)),
        cache_policy=raw.get("cache_policy", "dynamic"),
        sensitivity=raw.get("sensitivity", "internal"),
        compressible=bool(raw.get("compressible", True)),
        relevance=float(raw.get("relevance", 0.5)),
        freshness=float(raw.get("freshness", 1.0)),
        metadata=raw.get("metadata", {}),
    )


def validate(path: str | Path) -> list[str]:
    """Validate a pack spec file. Returns a list of human-readable issues (empty = OK)."""
    issues: list[str] = []
    try:
        load_pack(path)
    except SpecError as e:
        issues.append(str(e))
    except FileNotFoundError as e:
        issues.append(str(e))
    return issues


def dump_pack(pack: ContextPack) -> dict[str, Any]:
    """Serialize a ContextPack back to a spec-format dict (round-trip friendly)."""
    return {
        "model": pack.model,
        "token_budget": pack.token_budget,
        "reserved_output_tokens": pack.reserved_output_tokens,
        "secret_policy": pack.config.secret_policy,
        "allow_truncation": pack.config.allow_truncation,
        "weights": dict(pack.config.weights),
        "items": [
            {
                "name": it.name,
                "content": it.content,
                "kind": it.kind,
                "priority": it.priority,
                "required": it.required,
                "freshness": it.freshness,
                "relevance": it.relevance,
                "source": it.source,
                "cache_policy": it.cache_policy,
                "sensitivity": it.sensitivity,
                "compressible": it.compressible,
                "metadata": dict(it.metadata),
            }
            for it in pack.items
        ],
        "references": [
            {
                "name": r.name,
                "location": r.location,
                "estimated_tokens": r.estimated_tokens,
                "kind": r.kind,
                "priority": r.priority,
                "required": r.required,
                "cache_policy": r.cache_policy,
                "sensitivity": r.sensitivity,
                "compressible": r.compressible,
                "relevance": r.relevance,
                "freshness": r.freshness,
                "metadata": dict(r.metadata),
            }
            for r in pack.references
        ],
    }


# ----- parsers ---------------------------------------------------------------


def _parse_yaml(text: str) -> dict:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:
        raise SpecError(
            "YAML pack specs require PyYAML. Install with `pip install ctxbudgeter[yaml]`."
        ) from e
    parsed = yaml.safe_load(text)
    return parsed if isinstance(parsed, dict) else {}


def _parse_toml(text: str) -> dict:
    try:
        import tomllib  # py3.11+
        return tomllib.loads(text)
    except ImportError:
        pass
    try:
        import tomli  # type: ignore[import-not-found]
        return tomli.loads(text)
    except ImportError as e:
        raise SpecError(
            "TOML pack specs require Python 3.11+ or `tomli`. "
            "Install with `pip install tomli`."
        ) from e

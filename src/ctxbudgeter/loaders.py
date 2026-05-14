"""Built-in loaders for References. Pluggable via `register_loader()`.

Loaders can be referenced from YAML pack specs by their registered name (safer
than importing arbitrary Python via dotted paths). Register custom loaders in
your project's conftest.py or app startup.
"""

from __future__ import annotations

import glob as _glob
import os
from collections.abc import Callable
from pathlib import Path

from .reference import Reference

_REGISTRY: dict[str, Callable] = {}


def register_loader(name: str, fn: Callable | None = None) -> Callable:
    """Register a loader under a short name.

    Two forms:
        register_loader("name", fn)                 # call form
        @register_loader("name")                    # decorator form
        def my_loader(ref: Reference) -> str: ...
    """
    if fn is not None:
        _REGISTRY[name] = fn
        return fn

    def _wrap(actual_fn: Callable) -> Callable:
        _REGISTRY[name] = actual_fn
        return actual_fn

    return _wrap


def get_loader(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown loader '{name}'. Registered: {sorted(_REGISTRY)}. "
            "Register custom loaders via ctxbudgeter.loaders.register_loader()."
        )
    return _REGISTRY[name]


def list_loaders() -> list[str]:
    return sorted(_REGISTRY)


# ----- built-ins --------------------------------------------------------------


@register_loader("file")
def file_loader(ref: Reference) -> str:
    """Read a text file from `ref.location`. Encoding is utf-8."""
    return Path(ref.location).read_text(encoding="utf-8")


@register_loader("glob")
def glob_loader(ref: Reference) -> str:
    """Concatenate text files matching the glob pattern in `ref.location`."""
    parts: list[str] = []
    for path in sorted(_glob.glob(ref.location, recursive=True)):
        try:
            parts.append(f"## {path}\n{Path(path).read_text(encoding='utf-8', errors='replace')}")
        except OSError:
            continue
    return "\n\n".join(parts)


@register_loader("env")
def env_loader(ref: Reference) -> str:
    """Return the value of an environment variable named `ref.location`."""
    val = os.environ.get(ref.location)
    if val is None:
        raise KeyError(f"Environment variable '{ref.location}' is not set.")
    return val


@register_loader("inline")
def inline_loader(ref: Reference) -> str:
    """Return the literal location string. Useful for testing and small fixtures."""
    return ref.location


@register_loader("http_get")
def http_get_loader(ref: Reference) -> str:
    """HTTP GET. Requires the `requests` library at call time (not a hard dep)."""
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "http_get loader requires `requests`. Install with `pip install requests`."
        ) from e
    timeout = float(ref.metadata.get("timeout", 10.0))
    resp = requests.get(ref.location, timeout=timeout)
    resp.raise_for_status()
    return resp.text

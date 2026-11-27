"""MCP server exposing the governed context API over stdio (JSON-RPC 2.0).

    ctxbudgeter serve-mcp --budget 128000 --model claude-opus-5.5 --state .ctx/session.json

Any MCP client (Claude Code, Claude Desktop, Cursor, an Agents SDK) can then call
``context_inspect``, ``context_compact``, ``context_recall``, ... and read the
context-management skill as a resource.

Important limitation (by design of MCP): a server cannot rewrite the *internal*
context window of a proprietary client. Over MCP the runtime acts as a governed,
external working-context store: the client (or its harness) pushes events with
``context_add`` (enabled with ``--allow-add``) and renders the managed context from
``context_inspect`` / the ``ctxbudgeter://context/rendered`` resource. Harnesses that
control model requests should embed :class:`~ctxbudgeter.runtime.ContextRuntime`
directly (see :class:`~ctxbudgeter.adapters.harness.ContextHarness`) for full runtime
integration; restricted harnesses run in guidance-only mode.

No third-party MCP SDK is required: the protocol subset implemented is
``initialize``, ``ping``, ``tools/list``, ``tools/call``, ``resources/list``,
``resources/read`` and ``prompts/list``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import IO, Any

from .. import __version__
from ..runtime.archive import JSONArchive
from ..runtime.runtime import ContextRuntime
from ..runtime.trace import TraceLog
from .context_tools import ContextToolDispatcher, mcp_tools

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
_MUTATING = {"context_compact", "context_pin", "context_unpin", "context_archive", "context_recall",
             "context_rollback", "context_checkpoint", "context_signal", "context_add"}

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602


class ContextMCPServer:
    def __init__(self, runtime: ContextRuntime, *, allow_add: bool = False,
                 state_path: str | Path | None = None) -> None:
        self.rt = runtime
        self.allow_add = allow_add
        self.state_path = Path(state_path) if state_path else None
        self.dispatcher = ContextToolDispatcher(runtime, allow_add=allow_add)
        self.initialized = False

    # ------------------------------------------------------------- persistence

    @classmethod
    def from_state(cls, state_path: str | Path, *, budget: int, model: str, allow_add: bool = False,
                   **runtime_kwargs: Any) -> ContextMCPServer:
        """Load (or create) a persistent session: runtime state, archive and traces live
        next to ``state_path``."""
        sp = Path(state_path)
        archive = JSONArchive(sp.with_suffix(".archive.json"))
        trace = TraceLog(sp.with_suffix(".traces.jsonl"))
        if sp.exists():
            rt = ContextRuntime.from_dict(json.loads(sp.read_text(encoding="utf-8")), archive=archive, trace=trace)
        else:
            rt = ContextRuntime(budget=budget, model=model, archive=archive, trace=trace, **runtime_kwargs)
        return cls(rt, allow_add=allow_add, state_path=sp)

    def save(self) -> None:
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.rt.to_dict(), indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- protocol

    def handle(self, msg: Any) -> dict[str, Any] | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return _error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "invalid request")
        method, mid, params = msg["method"], msg.get("id"), msg.get("params") or {}
        is_notification = "id" not in msg
        try:
            result = self._dispatch(method, params)
        except _RPCError as e:
            return None if is_notification else _error(mid, e.code, str(e))
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
            self.initialized = True
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": "ctxbudgeter-context", "version": __version__},
                "instructions": ("Governed context API. Read the resource ctxbudgeter://skills/context-management "
                                 "for how and when to compact, pin, archive and recall."),
            }
        if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": mcp_tools(include_add=self.allow_add)}
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                raise _RPCError(INVALID_PARAMS, "tools/call requires a tool name")
            out = self.dispatcher.call(name, params.get("arguments") or {})
            if name in _MUTATING and "error" not in out:
                self.save()
            is_error = "error" in out
            return {"content": [{"type": "text", "text": json.dumps(out.get("result", out), default=str)}],
                    "structuredContent": out, "isError": is_error}
        if method == "resources/list":
            return {"resources": [
                {"uri": "ctxbudgeter://skills/context-management", "name": "context-management skill",
                 "mimeType": "text/markdown", "description": "How to manage your context with this API."},
                {"uri": "ctxbudgeter://context/rendered", "name": "managed context",
                 "mimeType": "text/plain", "description": "The active context, rendered with event tags."},
            ]}
        if method == "resources/read":
            uri = params.get("uri")
            if uri == "ctxbudgeter://skills/context-management":
                from ..skills import render_guidance

                text = render_guidance(policy=self.rt.policy, autonomy=self.rt.autonomy)
                return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": text}]}
            if uri == "ctxbudgeter://context/rendered":
                return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": self.rt.render_text()}]}
            raise _RPCError(INVALID_PARAMS, f"unknown resource {uri!r}")
        if method == "prompts/list":
            return {"prompts": []}
        raise _RPCError(METHOD_NOT_FOUND, f"method not found: {method}")

    def serve(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        """Serve newline-delimited JSON-RPC until EOF."""
        inp = stdin or sys.stdin
        out = stdout or sys.stdout
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                resp: dict[str, Any] | None = _error(None, PARSE_ERROR, f"parse error: {e}")
            else:
                if isinstance(msg, list):  # JSON-RPC batch
                    batch = [r for r in (self.handle(m) for m in msg) if r is not None]
                    if batch:
                        out.write(json.dumps(batch) + "\n")
                        out.flush()
                    continue
                resp = self.handle(msg)
            if resp is not None:
                out.write(json.dumps(resp, default=str) + "\n")
                out.flush()


class _RPCError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

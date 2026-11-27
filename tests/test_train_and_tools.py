"""Training datasets, context tool schemas/dispatcher, MCP server, harness integration, CLI."""

from __future__ import annotations

import io
import json
import re
from types import SimpleNamespace

from typer.testing import CliRunner

from ctxbudgeter.adapters import (
    TOOL_NAMES,
    ContextHarness,
    ContextMCPServer,
    ContextToolDispatcher,
    anthropic_tools,
    mcp_tools,
    openai_tools,
)
from ctxbudgeter.bench import BenchRunner, TaskGenerator, make_agent
from ctxbudgeter.cli import app
from ctxbudgeter.runtime import ContextRuntime, TraceLog, load_traces
from ctxbudgeter.train import build_datasets, label_traces

runner = CliRunner()


def busy_runtime(**kw) -> ContextRuntime:
    rt = ContextRuntime(budget=2600, reserved_output_tokens=200, **kw)
    rt.add("You are an agent.", kind="system")
    rt.add("Find ALPHA-7.", kind="task")
    for i in range(12):
        rt.add(f"Doc {i}: entity ALPHA-{i} value {100 + i}. " + "filler words here " * 30, source=f"d{i}.txt")
    return rt


# ------------------------------------------------------------------ train


def _bench_traces(tmp_path):
    log = TraceLog(tmp_path / "tr.jsonl")
    tasks = TaskGenerator(seed=2).generate("research", 1, stream_tokens=5000)
    BenchRunner(tasks, lambda: make_agent("reckless"), modes=["unrestricted", "adaptive"], budgets=[1800],
                trace=log, optimize=False).run()
    return load_traces(tmp_path / "tr.jsonl")


def test_labels_and_datasets(tmp_path) -> None:
    traces = _bench_traces(tmp_path)
    labels = label_traces(traces)
    assert labels and {lab.label for lab in labels.values()} <= {"good", "bad", "neutral"}
    assert any(lab.label == "bad" and any("critical" in r for r in lab.reasons) for lab in labels.values())
    manifest = build_datasets(traces, tmp_path / "ds")
    assert manifest["files"]["rewards"] == len(labels)
    for line in (tmp_path / "ds" / "sft.jsonl").read_text().splitlines():
        row = json.loads(line)
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        call = row["messages"][2]["tool_calls"][0]["function"]
        assert call["name"] in TOOL_NAMES and isinstance(json.loads(call["arguments"]), dict)
    prefs = [json.loads(x) for x in (tmp_path / "ds" / "preferences.jsonl").read_text().splitlines()]
    assert prefs and all({"prompt", "chosen", "rejected"} <= set(p) for p in prefs)
    assert all(-1 <= json.loads(x)["reward"] <= 1 for x in (tmp_path / "ds" / "rewards.jsonl").read_text().splitlines())


# ------------------------------------------------------------------ tools


def test_tool_schemas_are_consistent_and_valid() -> None:
    names = [t["function"]["name"] for t in openai_tools()]
    assert names == [t["name"] for t in anthropic_tools()] == [t["name"] for t in mcp_tools()] == list(TOOL_NAMES)
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in names)
    assert "context_add" in [t["name"] for t in mcp_tools(include_add=True)]
    for t in mcp_tools():
        assert t["inputSchema"]["type"] == "object"


def test_dispatcher_executes_and_reports_errors() -> None:
    rt = busy_runtime()
    d = ContextToolDispatcher(rt)
    assert d.call("context_budget")["result"]["used"] == rt.used_tokens
    plan = d.call("context_plan", {"target_usage": 0.4})["result"]
    assert plan["ops"]
    op = plan["ops"][0]
    res = d.call("context_compact", json.dumps(op["args"]))["result"]
    assert res["status"] == "applied"
    assert d.call("context_recall", {"tag": op["args"]["start"]})["result"]["status"] == "applied"
    assert "error" in d.call("context_nope")
    assert "error" in d.call("context_compact", "{bad json")
    assert "error" in d.call("context_pin", {})
    assert "error" in d.call("context_signal", {"phase": "dancing"})
    assert "error" in d.call("context_add", {"content": "x"})  # disabled by default
    assert d.call("context_signal", {"phase": "subtask_complete"})["result"]["phase"] == "subtask_complete"


# ------------------------------------------------------------------ MCP


def rpc(server: ContextMCPServer, method: str, params: dict | None = None, mid: int | None = 1):
    msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if mid is not None:
        msg["id"] = mid
    return server.handle(msg)


def test_mcp_protocol_surface() -> None:
    server = ContextMCPServer(busy_runtime(), allow_add=True)
    init = rpc(server, "initialize", {"protocolVersion": "2024-11-05"})["result"]
    assert init["protocolVersion"] == "2024-11-05" and init["capabilities"]["tools"] is not None
    assert rpc(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == "2025-06-18"
    assert rpc(server, "notifications/initialized", mid=None) is None
    assert rpc(server, "ping")["result"] == {}
    tools = rpc(server, "tools/list")["result"]["tools"]
    assert "context_add" in [t["name"] for t in tools]
    out = rpc(server, "tools/call", {"name": "context_archive", "arguments": {"tags": ["1"]}})["result"]
    assert out["isError"] is False and json.loads(out["content"][0]["text"])["status"] == "denied"
    err = rpc(server, "tools/call", {"name": "context_nope"})["result"]
    assert err["isError"] is True
    res = rpc(server, "resources/list")["result"]["resources"]
    uris = {r["uri"] for r in res}
    skill = rpc(server, "resources/read", {"uri": "ctxbudgeter://skills/context-management"})["result"]
    assert "Context management" in skill["contents"][0]["text"] and len(uris) == 2
    assert rpc(server, "resources/read", {"uri": "x://y"})["error"]["code"] == -32602
    assert rpc(server, "nope")["error"]["code"] == -32601
    assert server.handle({"id": 3})["error"]["code"] == -32600
    assert rpc(server, "tools/call", {})["error"]["code"] == -32602


def test_mcp_stdio_and_persistence(tmp_path) -> None:
    state = tmp_path / "s.json"
    server = ContextMCPServer.from_state(state, budget=4000, model="m", allow_add=True)
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "context_add", "arguments": {"content": "hello world", "kind": "user_message"}}},
    ]
    out = io.StringIO()
    server.serve(io.StringIO("\n".join(json.dumps(x) for x in lines) + "\nnot json\n"
                             + json.dumps([{"jsonrpc": "2.0", "id": 9, "method": "ping"}]) + "\n"), out)
    responses = [json.loads(x) for x in out.getvalue().splitlines()]
    assert responses[0]["id"] == 1 and responses[2]["error"]["code"] == -32700
    assert responses[3] == [{"jsonrpc": "2.0", "id": 9, "result": {}}]
    assert state.exists()
    again = ContextMCPServer.from_state(state, budget=4000, model="m")
    assert again.rt.events[0].content == "hello world"


# ------------------------------------------------------------------ harness


def test_context_harness_openai_flow() -> None:
    h = ContextHarness(busy_runtime(), provider="openai")
    req = h.request()
    assert req["messages"][0]["role"] == "system" and "Context management" in req["messages"][0]["content"]
    assert len(req["tools"]) == len(TOOL_NAMES)
    call_obj = SimpleNamespace(id="c1", function=SimpleNamespace(name="context_budget", arguments="{}"))
    other = {"id": "c2", "function": {"name": "web_search", "arguments": "{}"}}
    results, others = h.handle_tool_calls([call_obj, other])
    assert results[0]["name"] == "context_budget" and others == [other]
    msgs = h.tool_result_messages(results)
    assert msgs[0]["role"] == "tool" and msgs[0]["tool_call_id"] == "c1"
    assert h.runtime.step_index == 1


def test_context_harness_anthropic_and_guidance_only() -> None:
    h = ContextHarness(busy_runtime(), provider="anthropic")
    req = h.request()
    assert "You are an agent." in req["system"] and req["tools"][0]["input_schema"]
    results, _ = h.handle_tool_calls([{"type": "tool_use", "id": "u1", "name": "context_inspect", "input": {}}])
    assert h.tool_result_messages(results)[0]["content"][0]["tool_use_id"] == "u1"
    g = ContextHarness(busy_runtime(), provider="openai", guidance_only=True)
    req = g.request()
    assert "tools" not in req
    assert g.runtime.autonomy.name == "RECOMMEND"
    if g.last_step and g.last_step.warning:
        assert "[context advice for the harness]" in req["messages"][-1]["content"]


# ------------------------------------------------------------------ CLI


def test_cli_new_commands(tmp_path) -> None:
    r = runner.invoke(app, ["skill", "export", str(tmp_path / "skills")])
    assert r.exit_code == 0 and (tmp_path / "skills" / "context-management" / "SKILL.md").exists()
    r = runner.invoke(app, ["skill", "show", "--brief", "--model", "claude-opus-5.5"])
    assert r.exit_code == 0 and "context-pressure warning" in r.stdout
    r = runner.invoke(app, ["context-tools", "-p", "anthropic"])
    assert r.exit_code == 0 and json.loads(r.stdout)[0]["input_schema"]
    assert runner.invoke(app, ["context-tools", "-p", "x"]).exit_code == 2
    rep, tr = tmp_path / "rep.json", tmp_path / "tr.jsonl"
    r = runner.invoke(app, ["bench", "-a", "reckless", "-n", "1", "--families", "research", "--budgets", "1800",
                            "--stream-tokens", "5000", "--no-optimize", "-o", str(rep), "--traces", str(tr)])
    assert r.exit_code == 0, r.output
    assert "| adaptive |" in r.stdout and json.loads(rep.read_text())["episodes"]
    assert runner.invoke(app, ["bench", "-a", "nobody"]).exit_code == 2
    r = runner.invoke(app, ["profile", str(tr), "--store", str(tmp_path / "store"), "-o", str(tmp_path / "p.json")])
    assert r.exit_code == 0 and "unsafe" in r.stdout
    r = runner.invoke(app, ["train-export", str(tr), "-o", str(tmp_path / "ds")])
    assert r.exit_code == 0 and (tmp_path / "ds" / "manifest.json").exists()
    reg = tmp_path / "aut.json"
    r = runner.invoke(app, ["autonomy", "certify", str(rep), "--registry", str(reg)])
    assert r.exit_code == 0 and "hold" in r.stdout
    r = runner.invoke(app, ["autonomy", "show", "reckless", "bench-model", "--registry", str(reg)])
    assert r.exit_code == 0 and "L1" in r.stdout
    assert runner.invoke(app, ["serve-mcp", "--budget", "100"]).exit_code == 2
    r = runner.invoke(app, ["serve-mcp", "--budget", "8000"],
                      input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n")
    assert r.exit_code == 0 and "context_compact" in r.stdout

"""Compile context and emit a LangGraph state dict (no langgraph dependency needed).

Run:  python examples/langgraph_adapter_demo.py
"""

from __future__ import annotations

import json

from ctxbudgeter import ContextPack


def main() -> None:
    pack = ContextPack(model="claude-sonnet-4.6", token_budget=24_000, reserved_output_tokens=4_000)
    pack.add(name="system", content="You are a routing agent.", kind="system",
             priority=100, required=True, cache_policy="stable")
    pack.add(name="tools", content="search(query), create_issue(title, body)",
             kind="tool_def", priority=85, cache_policy="stable")
    pack.add(name="user", content="File a bug about the login page.", kind="user_message", priority=80)
    pack.add(name="task", content="Decide which tool to call.", kind="task", required=True)

    compiled = pack.compile(task="route")
    state = compiled.to_langgraph_state(user_message="Proceed.")
    print("LangGraph state keys:", sorted(state.keys()))
    print("messages:", json.dumps(state["messages"], indent=2)[:400])
    print("context_items:", list(state["context_items"].keys()))


if __name__ == "__main__":
    main()

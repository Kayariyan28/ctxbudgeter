"""Compile context and emit OpenAI Agents SDK input (no openai dependency needed).

Run:  python examples/openai_agents_adapter_demo.py
"""

from __future__ import annotations

import json

from ctxbudgeter import ContextPack


def main() -> None:
    pack = ContextPack(model="gpt-4o", token_budget=24_000, reserved_output_tokens=4_000)
    pack.add(name="system", content="You are a helpful coding agent.", kind="system",
             priority=100, required=True, cache_policy="stable")
    pack.add(name="readme", content="# Project\nA small library.", kind="project_doc",
             priority=80, cache_policy="stable")
    pack.add(name="prior_user", content="What does this repo do?", kind="user_message", priority=70)
    pack.add(name="task", content="Summarize the repository.", kind="task", required=True)

    compiled = pack.compile(task="summarize")
    payload = compiled.to_openai_agents_input(user_message="Go.")
    print("instructions:\n", payload["instructions"][:300])
    print("\ninput messages:\n", json.dumps(payload["input"], indent=2)[:400])


if __name__ == "__main__":
    main()

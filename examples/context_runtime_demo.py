"""Governed, model-driven context management in ~60 lines.

Run:  python examples/context_runtime_demo.py

Shows the context API an agent (or its harness) uses: budget and timing verdicts,
diagnosis, a cache-aware plan, a governed compaction whose originals stay
recoverable, a denied attempt to remove protected context, recall before answering,
and the session evaluation with prefill cost under prefix caching vs suffix KV reuse.
"""

from __future__ import annotations

from ctxbudgeter.governance import ContextEditPolicy
from ctxbudgeter.runtime import ContextRuntime

rt = ContextRuntime(model="claude-sonnet-4.6", budget=3_000, reserved_output_tokens=300,
                    policy=ContextEditPolicy(min_fact_retention=0.5))
rt.add("You are a research agent. Report exact figures.", kind="system")
rt.add("Task: find FY2019 receipts for ALPHA-7. Use corrected figures.", kind="task")

for i in range(14):
    fact = f"Ledger note: entity ALPHA-{i} reported receipts for FY2019 of {1_000 + 37 * i:,} million USD."
    rt.add(fact + " " + "The consolidated schedule reconciles the bureau accounts. " * 10,
           kind="document", source=f"corpus/treasury_{i:02d}.txt")
    if i % 4 == 3:
        rt.signal("subtask_complete")
    report = rt.step()
    if report.warning:
        print(f"step {report.step}: {report.warning[:110]}…")
        break

print("\nbudget:", {k: rt.budget()[k] for k in ("used", "available", "usage", "pressure")})
print("timing:", rt.budget()["timing"]["decision"], "—", rt.budget()["timing"]["reasons"][0])
print("diagnose:", [(d.kind, d.tags) for d in rt.diagnose()][:4])

plan = rt.plan()
print("plan:", [o.to_dict()["args"] for o in plan.ops], "| notes:", plan.notes)
for result in rt.apply_plan(plan):
    print(f"  {result.op}: {result.status}, freed {result.tokens_freed:,} tokens, "
          f"retention {result.decision['notes'].get('fact_retention')}")

rt.add("Next document after compaction. " * 5, kind="document", source="corpus/treasury_extra.txt")
rt.step()  # the request after an edit: prefix caching recomputes the suffix, suffix reuse does not

denied = rt.archive("1")
print("\ntry to archive the system prompt →", denied.status, denied.decision["violations"][0]["rule"])

question = "What were ALPHA-7 receipts for FY2019?"
rt.add(question, kind="user_message")
for hint in rt.recall_hints(question, limit=1):
    print("recall hint:", hint["tag"], hint["preview"][:70])
    print("recalled:", rt.recall(hint["tag"]).details["content"][:90])

ev = rt.evaluate()
print("\nsession:", {k: ev[k] for k in ("compactions_applied", "median_compression_factor",
                                        "mean_fact_retention", "reopen_rate", "critical_violations")})
print("prefill:", ev["cost"])

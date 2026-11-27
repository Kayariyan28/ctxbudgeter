"""CTXBench: compare harness-driven, unrestricted, guided and adaptive context management.

Run:  python examples/ctxbench_demo.py

Uses the deterministic reference agents. To benchmark a real model, swap the agent
factory for ``LLMContextAgent(openai_complete(OpenAI(), "gpt-..."))`` or
``LLMContextAgent(anthropic_complete(Anthropic(), "claude-..."))`` and pass an
``answer_fn`` to score end-to-end answers.
"""

from __future__ import annotations

from ctxbudgeter.bench import BenchRunner, TaskGenerator, make_agent
from ctxbudgeter.governance import AutonomyRegistry

tasks = TaskGenerator(seed=7).suite(["research", "coding", "workflow"], n=1, stream_tokens=10_000)

for preset in ("competent", "avoidant", "reckless"):
    report = BenchRunner(tasks, lambda preset=preset: make_agent(preset), budgets=[3_000, 6_000]).run()
    print(f"\n=== {preset} ===")
    for mode, agg in report.summary.items():
        print(f"{mode:>12}: CIS {agg['cis']:5.1f}  acc {agg['accuracy']:.2f}  retention {agg['retention']:.2f}  "
              f"critical {agg['critical_violations']}")
    detected = report.calibration_profile["behaviors"] if report.calibration_profile else []
    selected = report.adaptive_policy["evidence"]["optimization"]["selected"]
    print(f"detected {detected} → selected policy '{selected}' ({report.adaptive_policy['status']})")

    # Graduated autonomy: certify from held-out evidence.
    reg = AutonomyRegistry()
    result = reg.certify(preset, "bench-model", report.evidence("adaptive"))
    print("autonomy:", result.action, f"L{int(result.current)} → L{int(result.new)}", result.reasons[0])

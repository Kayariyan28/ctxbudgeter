"""CTXBench — a benchmark for how responsibly agents manage their own context.

    from ctxbudgeter.bench import BenchRunner, TaskGenerator, make_agent

    tasks = TaskGenerator(seed=7).suite(n=2, stream_tokens=12_000)
    report = BenchRunner(tasks, lambda: make_agent("competent"), budgets=[4_000, 8_000]).run()
    print(report.to_markdown())

Bring a real model with :class:`LLMContextAgent` (``openai_complete`` /
``anthropic_complete``) and, for end-to-end accuracy, an ``answer_fn``.
"""

from __future__ import annotations

from .agents import (
    PRESETS,
    ContextAgent,
    LLMContextAgent,
    Observation,
    ScriptedAgent,
    anthropic_complete,
    make_agent,
    openai_complete,
    to_anthropic_conversation,
)
from .runner import MODES, BenchReport, BenchRunner, EpisodeResult, answer_correct, run_episode
from .scoring import DEFAULT_WEIGHTS, context_intelligence_score, grade_for
from .tasks import (
    FAMILIES,
    SPECS,
    BenchTask,
    TaskEvent,
    TaskGenerator,
    read_answer,
    read_facts,
    value_present,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "FAMILIES",
    "MODES",
    "PRESETS",
    "SPECS",
    "BenchReport",
    "BenchRunner",
    "BenchTask",
    "ContextAgent",
    "EpisodeResult",
    "LLMContextAgent",
    "Observation",
    "ScriptedAgent",
    "TaskEvent",
    "TaskGenerator",
    "answer_correct",
    "anthropic_complete",
    "context_intelligence_score",
    "grade_for",
    "make_agent",
    "openai_complete",
    "read_answer",
    "read_facts",
    "run_episode",
    "to_anthropic_conversation",
    "value_present",
]

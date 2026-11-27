"""ctxbudgeter.train — turn context decisions into training data (Level 3: parametric).

Decision traces from live sessions or CTXBench runs become:

* **SFT** examples (OpenAI chat fine-tuning JSONL): state → the good tool call;
* **preference pairs** (TRL-style conversational DPO JSONL): at the same state, the
  governor-approved / better decision is ``chosen`` and the denied or harmful one
  ``rejected`` (a denied premature compaction pairs with "keep working");
* **reward records** (JSONL) for RL: state, action, a shaped per-decision reward plus
  the episode return for credit assignment.

Labels come from outcomes, not opinions: governance verdicts, critical violations,
fact retention of summaries, whether archived content had to be reopened, timing
relative to the rubric floor, and — via ``session_end`` traces — whether the episode
was answered correctly.
"""

from __future__ import annotations

from .export import (
    build_datasets,
    export_preferences,
    export_rewards,
    export_sft,
    state_prompt,
    to_tool_call,
)
from .labeling import Label, label_trace, label_traces

__all__ = [
    "Label",
    "build_datasets",
    "export_preferences",
    "export_rewards",
    "export_sft",
    "label_trace",
    "label_traces",
    "state_prompt",
    "to_tool_call",
]

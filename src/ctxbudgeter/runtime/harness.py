"""Harness-driven compaction — the baseline today's agent harnesses use.

When usage crosses a threshold (80% here, as in NVIDIA's study), the harness
summarizes the earlier history, preserving the initial instructions and the most
recent events (10). The model decides nothing; originals are *discarded*, not
archived. This is Baseline A in :mod:`ctxbudgeter.bench`, and a drop-in policy for
harnesses that want deterministic compaction without agent involvement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .summarizer import Summarizer

if TYPE_CHECKING:
    from .runtime import ContextRuntime


@dataclass
class HarnessCompactor:
    threshold: float = 0.8
    keep_recent: int = 10
    summary_fraction: float = 0.25
    """Summary target as a fraction of the replaced range's tokens."""
    max_summary_fraction_of_budget: float = 0.15
    summarizer: Summarizer | None = None

    def maybe_compact(self, rt: ContextRuntime) -> bool:
        if rt.usage < self.threshold:
            return False
        events = rt.events
        start = 0
        while start < len(events) and events[start].kind in ("system", "task"):
            start += 1  # the initial instructions
        end = len(events) - self.keep_recent
        if end - start < 1:
            return False
        rng = events[start:end]
        toks = sum(e.tokens for e in rng)
        target = max(32, min(math.ceil(toks * self.summary_fraction),
                             int(rt.available_tokens * self.max_summary_fraction_of_budget)))
        summarize = self.summarizer or rt.summarizer
        text = summarize(rng, target)
        rt._replace_range(start, end - 1, text, archive_originals=False, actor="harness")
        return True

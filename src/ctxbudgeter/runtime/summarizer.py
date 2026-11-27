"""Summarizers turn a range of events into the text of a summary event.

The runtime accepts any callable ``(events, target_tokens) -> str``, so an agent can
write its own summary (the model-driven path), or a harness can plug in an LLM.
:class:`ExtractiveSummarizer` is the deterministic, offline default: it keeps the
sentences that carry the most salient terms (numbers, identifiers, paths, decisions,
corrections) up to a token target, in original order, labelled by source event tag.
It is what the harness baseline and the governed fallback use, and what the planner
uses to price a compaction.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

from ..tokenizer import TokenCounter
from .analysis import salient_terms, split_sentences
from .events import ContextEvent

Summarizer = Callable[[Sequence[ContextEvent], int], str]

_SIGNAL = re.compile(
    r"\b(decid\w*|decision|must|never|always|error|fail\w*|fix\w*|result\w*|answer|"
    r"correction|supersed\w*|revised|todo|blocked|because|conclu\w*|found|confirm\w*)\b",
    re.IGNORECASE,
)


_DIGITS = re.compile(r"\d[\d,.]*")
_WORDISH = re.compile(r"\b[A-Za-z]+\b")


def _shape(sentence: str) -> str:
    """Template signature: numbers masked, first five words kept."""
    masked = _DIGITS.sub("#", sentence)
    return " ".join(_WORDISH.findall(masked)[:5]) + "|" + str(masked.count("#"))


class ExtractiveSummarizer:
    """Deterministic salient-sentence extraction under a token target."""

    def __init__(self, model: str = "claude-sonnet-4.6", *, counter: TokenCounter | None = None) -> None:
        self.counter = counter or TokenCounter(model)

    def __call__(self, events: Sequence[ContextEvent], target_tokens: int) -> str:
        candidates: list[tuple[float, int, int, str]] = []  # (score, ev_idx, s_idx, sentence)
        seen: set[str] = set()
        raw: list[tuple[int, int, str]] = []
        shape_count: Counter[str] = Counter()
        for ei, ev in enumerate(events):
            for si, sent in enumerate(split_sentences(ev.content)):
                norm = " ".join(sent.split())
                if norm in seen or len(norm) < 3:
                    continue
                seen.add(norm)
                raw.append((ei, si, norm))
                shape_count[_shape(norm)] += 1
        for ei, si, norm in raw:
            score = len(salient_terms(norm)) + (1.5 if _SIGNAL.search(norm) else 0.0)
            score += 0.25 if si == 0 else 0.0  # topic sentence bonus
            # Redundancy discount: rows stamped from one template (table rows, log lines)
            # are individually less informative than a one-off statement.
            score /= 1.0 + math.log(shape_count[_shape(norm)])
            candidates.append((score, ei, si, norm))

        header = f"Summary of events {events[0].tag}..{events[-1].tag}:" if events else "Summary:"
        budget = max(0, target_tokens - self.counter.count(header))
        chosen: list[tuple[int, int, str]] = []
        used = 0
        for score, ei, si, sent in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
            if score <= 0 and chosen:
                break
            cost = self.counter.count(sent) + 2
            if used + cost > budget:
                continue
            chosen.append((ei, si, sent))
            used += cost
        chosen.sort()
        lines = [header]
        for ei, _si, sent in chosen:
            lines.append(f"- [{events[ei].tag}] {sent}")
        return "\n".join(lines)

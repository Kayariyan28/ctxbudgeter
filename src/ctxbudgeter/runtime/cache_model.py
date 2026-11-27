"""Prefill cost model for edited contexts (NVIDIA "Letting Models Manage Their Own
Context", Section 7).

Inference engines reuse the KV cache only for the longest unchanged *prefix* of a
prompt. Replacing a range of history with a summary keeps the cache valid up to the
edit, but forces recomputation of the summary *and everything after it* — even text
that did not change. Engines that support *suffix KV reuse* (SideQuest, Context
Language Models' Suffix Cache Reuse, Cache Engineering for Agents) also reuse the
cached states of unchanged events after the edit.

:class:`PrefillCostModel` replays a session's requests at event granularity and
reports, per request and cumulatively, how many input tokens must be prefilled
under each regime. Event identity is (tag, checksum), so any content change counts
as new. This is an accounting model — it does not run an inference engine — but it
reproduces the mechanism the paper measures and lets a planner price an edit before
making it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .events import ContextEvent


@dataclass(frozen=True)
class RequestCost:
    step: int
    input_tokens: int
    prefix_reused: int
    prefill_prefix_cache: int
    prefill_suffix_reuse: int

    def to_dict(self) -> dict[str, int]:
        return {
            "step": self.step,
            "input_tokens": self.input_tokens,
            "prefix_reused": self.prefix_reused,
            "prefill_prefix_cache": self.prefill_prefix_cache,
            "prefill_suffix_reuse": self.prefill_suffix_reuse,
        }


@dataclass
class PrefillCostModel:
    """Tracks prefill tokens under prefix caching vs. suffix KV reuse."""

    requests: list[RequestCost] = field(default_factory=list)
    _prev: list[tuple[str, str, int]] = field(default_factory=list)

    @staticmethod
    def _key(events: Sequence[ContextEvent]) -> list[tuple[str, str, int]]:
        return [(e.tag, e.checksum, e.tokens) for e in events]

    def record(self, events: Sequence[ContextEvent], step: int) -> RequestCost:
        cur = self._key(events)
        total = sum(t for _, _, t in cur)
        prefix = 0
        i = 0
        while i < min(len(cur), len(self._prev)) and cur[i][:2] == self._prev[i][:2]:
            prefix += cur[i][2]
            i += 1
        seen = {k[:2] for k in self._prev}
        suffix_reused = sum(t for (tag, cs, t) in cur[i:] if (tag, cs) in seen)
        rc = RequestCost(
            step=step,
            input_tokens=total,
            prefix_reused=prefix,
            prefill_prefix_cache=total - prefix,
            prefill_suffix_reuse=total - prefix - suffix_reused,
        )
        self.requests.append(rc)
        self._prev = cur
        return rc

    @staticmethod
    def edit_recompute_cost(events: Sequence[ContextEvent], first_index: int) -> int:
        """Tokens a prefix-caching engine must recompute after editing at ``first_index``
        (excluding the replacement text itself)."""
        return sum(e.tokens for e in events[first_index:])

    def totals(self) -> dict[str, int | float]:
        inp = sum(r.input_tokens for r in self.requests)
        pc = sum(r.prefill_prefix_cache for r in self.requests)
        sr = sum(r.prefill_suffix_reuse for r in self.requests)
        return {
            "requests": len(self.requests),
            "cumulative_input_tokens": inp,
            "prefill_prefix_cache": pc,
            "prefill_suffix_reuse": sr,
            "suffix_reuse_saving": round(1 - sr / pc, 4) if pc else 0.0,
        }

    def reset(self) -> None:
        self.requests.clear()
        self._prev = []

"""Decision traces — the raw material for evaluation, adaptation and training.

Every context operation an agent proposes is recorded as a :class:`DecisionTrace`:
what was proposed, the state it was proposed in (content-light), what the governor
decided, and — filled in later — its outcome (was the compacted content reopened?
did facts survive?). Traces are written as JSON Lines so they can be appended to
from many sessions and streamed into the profiler and dataset exporters.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DecisionTrace:
    trace_id: str
    session_id: str
    agent_id: str
    model: str
    step: int
    op: str
    args: dict[str, Any]
    autonomy: int
    verdict: str  # applied | denied | recommended | pending | modified | observed
    usage_before: float
    tokens_before: int
    tokens_after: int
    phase: str | None = None
    violations: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def compression_factor(self) -> float | None:
        if self.op != "compact" or self.verdict not in ("applied", "modified") or self.tokens_after <= 0:
            return None
        return self.tokens_before / self.tokens_after

    @property
    def critical(self) -> bool:
        return any(v.get("severity") == "critical" for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecisionTrace:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class TraceLog:
    """In-memory trace list with optional JSONL persistence (append-only)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.traces: list[DecisionTrace] = []

    def append(self, trace: DecisionTrace) -> None:
        self.traces.append(trace)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(trace.to_dict(), sort_keys=True, default=str) + "\n")

    def update_outcome(self, trace_id: str, **outcome: Any) -> None:
        """Record an outcome in memory. Persist annotated traces with :meth:`write`."""
        for t in self.traces:
            if t.trace_id == trace_id:
                t.outcome.update(outcome)
                return

    def write(self, path: str | Path) -> Path:
        """Write the full (outcome-annotated) log to ``path`` as JSONL."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for t in self.traces:
                fh.write(json.dumps(t.to_dict(), sort_keys=True, default=str) + "\n")
        return p

    def __iter__(self) -> Iterator[DecisionTrace]:
        return iter(self.traces)

    def __len__(self) -> int:
        return len(self.traces)

    def extend(self, traces: Iterable[DecisionTrace]) -> None:
        for t in traces:
            self.append(t)


def load_traces(*paths: str | Path) -> list[DecisionTrace]:
    """Load traces from one or more JSONL files. Later records with the same
    ``trace_id`` replace earlier ones, so an annotated rewrite supersedes the log."""
    by_id: dict[str, DecisionTrace] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            t = DecisionTrace.from_dict(json.loads(line))
            by_id[t.trace_id] = t
    return list(by_id.values())

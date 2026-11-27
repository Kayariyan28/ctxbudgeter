"""CTXBench tasks — procedurally generated long-horizon episodes under context pressure.

Each task is a deterministic (seeded) stream of events — system instructions, the
user's requirement, documents and tool outputs, verbose logs, duplicate re-reads,
cross-agent handoffs — in which a handful of *facts* are planted. Some facts are
later *corrected*. At the end the agent is asked a question whose answer needs
specific facts, and only the latest corrected values are right.

The stream is several times larger than the context budget, so an agent cannot keep
everything: it must decide what to compact, what to keep verbatim, and what to
recall. Distractor content is deliberately dense in numbers and identifiers so that
naive extractive compression competes with the facts and loses some of them.

Answers are read deterministically from the final context (see
:func:`read_answer`), which isolates *context management* from reasoning ability:
an episode is correct exactly when the needed latest facts are available to the
model at answer time. Plug a real model in through ``answer_fn`` in the runner to
also measure end-to-end accuracy.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

FAMILIES = ("research", "coding", "analytics", "workflow", "multi_agent")


@dataclass(frozen=True)
class FamilySpec:
    name: str
    role: str
    fact_template: str
    pattern: str  # regex with groups: entity, metric, scope, value
    entity_prefixes: tuple[str, ...]
    metrics: tuple[str, ...]
    scopes: tuple[str, ...]
    value_range: tuple[int, int]
    nouns: tuple[str, ...]
    verbs: tuple[str, ...]
    adjectives: tuple[str, ...]
    log_line: str
    table_line: str
    doc_source: str


SPECS: dict[str, FamilySpec] = {
    "research": FamilySpec(
        "research", "document research",
        "Ledger note: entity {entity} reported {metric} for {scope} of {value} million USD.",
        r"entity ([A-Z]+-\d+) reported (\w+) for (FY\d{4}) of ([\d,]+) million USD",
        ("ALPHA", "BRAVO", "DELTA", "SIGMA", "OMEGA"), ("receipts", "outlays", "grants", "transfers"),
        ("FY2017", "FY2018", "FY2019", "FY2020"), (400, 9000),
        ("appropriation", "statement", "schedule", "account", "bureau", "ledger", "audit", "summary"),
        ("reconciles", "describes", "aggregates", "references", "restates", "clarifies"),
        ("consolidated", "quarterly", "supplementary", "preliminary", "historical", "detailed"),
        "{ts} INFO fetch page={n} status=200 bytes={b} parser=table-v2",
        "Line {n}: account {acct} adjustments {a} ; offsets {b} ; memo {c}",
        "corpus/treasury/{name}.txt",
    ),
    "coding": FamilySpec(
        "coding", "software engineering",
        "Decision log: module {entity} pins {metric} in {scope} to {value}.",
        r"module ([A-Z]+-\d+) pins (\w+) in (\w+) to ([\d,]+)",
        ("AUTH", "CART", "SEARCH", "BILLING", "QUEUE"), ("timeout_ms", "max_retries", "pool_size", "ttl_s"),
        ("production", "staging", "canary", "batch"), (2, 9000),
        ("handler", "fixture", "migration", "service", "adapter", "module", "test", "config"),
        ("wraps", "imports", "patches", "validates", "serializes", "dispatches"),
        ("legacy", "async", "flaky", "generated", "deprecated", "internal"),
        "{ts} PASSED tests/unit/test_case_{n}.py::test_path_{b} [{c}%]",
        "src/pkg/mod_{n}.py:{a}: warning W{b} unused variable tmp_{c}",
        "repo/src/{name}.py",
    ),
    "analytics": FamilySpec(
        "analytics", "data analytics",
        "Metric snapshot: segment {entity} shows {metric} for {scope} at {value} units.",
        r"segment ([A-Z]+-\d+) shows (\w+) for (Q\d-\d{4}) at ([\d,]+) units",
        ("NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"), ("revenue", "churn", "signups", "returns"),
        ("Q1-2025", "Q2-2025", "Q3-2025", "Q4-2025"), (100, 99000),
        ("cohort", "dashboard", "pivot", "query", "warehouse", "partition", "join", "report"),
        ("filters", "groups", "samples", "joins", "buckets", "normalizes"),
        ("weekly", "rolling", "seasonal", "raw", "deduplicated", "weighted"),
        "{ts} query_id=q{n} rows={b} scanned_mb={c} cache=miss",
        "row {n} | store {acct} | units {a} | price {b} | discount {c}",
        "warehouse/exports/{name}.csv",
    ),
    "workflow": FamilySpec(
        "workflow", "long-running operations",
        "Runbook state: job {entity} recorded {metric} for {scope} as {value}.",
        r"job ([A-Z]+-\d+) recorded (\w+) for (stage\d) as ([\d,]+)",
        ("INGEST", "BACKFILL", "EXPORT", "REINDEX", "ARCHIVE"), ("checkpoint", "batch_size", "error_count", "lag_s"),
        ("stage1", "stage2", "stage3", "stage4"), (1, 50000),
        ("worker", "shard", "queue", "lease", "cursor", "snapshot", "retry", "window"),
        ("advances", "resumes", "throttles", "commits", "drains", "rebalances"),
        ("idle", "stalled", "healthy", "degraded", "paused", "warm"),
        "{ts} worker-{n} heartbeat ok lease={b} cursor={c}",
        "shard {n} offset {a} committed {b} pending {c} owner w{acct}",
        "ops/runs/{name}.log",
    ),
    "multi_agent": FamilySpec(
        "multi_agent", "multi-agent coordination",
        "Handoff: agent {entity} commits {metric} for {scope} = {value}.",
        r"agent ([A-Z]+-\d+) commits (\w+) for (sprint\d+) = ([\d,]+)",
        ("PLANNER", "CODER", "REVIEWER", "TESTER", "OPS"), ("estimate_h", "story_pts", "budget_usd", "eta_days"),
        ("sprint11", "sprint12", "sprint13", "sprint14"), (1, 20000),
        ("ticket", "proposal", "review", "plan", "branch", "handoff", "standup", "backlog"),
        ("proposes", "assigns", "blocks", "approves", "escalates", "summarizes"),
        ("urgent", "parked", "shared", "draft", "accepted", "contested"),
        "{ts} bus msg_id={n} from=agent-{b} ack={c}",
        "ticket T-{n} status {a} points {b} assignee a{acct} age {c}d",
        "agents/{name}.msg",
    ),
}

_FACT_PATTERNS = {name: re.compile(spec.pattern) for name, spec in SPECS.items()}


@dataclass
class TaskEvent:
    kind: str
    content: str
    category: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    phase: str | None = None
    fact_keys: list[str] = field(default_factory=list)


@dataclass
class BenchTask:
    task_id: str
    family: str
    seed: int
    events: list[TaskEvent]
    question: str
    required_keys: list[str]
    expected: dict[str, str]
    superseded: dict[str, str]
    stream_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "family": self.family, "seed": self.seed, "question": self.question,
                "required_keys": list(self.required_keys), "expected": dict(self.expected),
                "superseded": dict(self.superseded), "events": len(self.events),
                "stream_tokens": self.stream_tokens}


def fact_key(entity: str, metric: str, scope: str) -> str:
    return f"{entity}|{metric}|{scope}"


def _is_correction(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 40):start].lower()
    after = text[end:end + 60].lower()
    return "correction:" in before or "supersedes" in after


def read_facts(text: str) -> dict[str, str]:
    """Value per fact key as a careful reader would take it from ``text``.

    Explicit corrections ("Correction: ... supersedes the earlier figure") outrank
    plain statements wherever they appear; among statements of the same kind, the
    later one wins.
    """
    found: list[tuple[bool, int, str, str]] = []
    for pat in _FACT_PATTERNS.values():
        for m in pat.finditer(text):
            found.append((_is_correction(text, m.start(), m.end()), m.start(),
                          fact_key(m.group(1), m.group(2), m.group(3)), m.group(4).replace(",", "")))
    found.sort()
    out: dict[str, str] = {}
    for _, _, k, v in found:
        out[k] = v
    return out


def read_answer(text: str, keys: list[str]) -> dict[str, str | None]:
    facts = read_facts(text)
    return {k: facts.get(k) for k in keys}


def value_present(text: str, key: str, value: str) -> bool:
    """True if ``text`` states ``key = value`` anywhere (order-insensitive)."""
    for pat in _FACT_PATTERNS.values():
        for m in pat.finditer(text):
            if fact_key(m.group(1), m.group(2), m.group(3)) == key and m.group(4).replace(",", "") == value:
                return True
    return False


class TaskGenerator:
    """Deterministic task generator. Same seed + parameters → identical tasks."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(
        self,
        family: str,
        n: int = 3,
        *,
        stream_tokens: int = 24_000,
        n_facts: int = 10,
        n_required: int = 3,
        n_corrections: int = 1,
        doc_tokens: int = 600,
    ) -> list[BenchTask]:
        if family not in SPECS:
            raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
        return [self._one(family, i, stream_tokens, n_facts, n_required, n_corrections, doc_tokens)
                for i in range(n)]

    def suite(self, families: tuple[str, ...] | list[str] = FAMILIES, n: int = 2, **kw: Any) -> list[BenchTask]:
        out: list[BenchTask] = []
        for f in families:
            out.extend(self.generate(f, n, **kw))
        return out

    # ------------------------------------------------------------------ internals

    def _one(self, family: str, idx: int, stream_tokens: int, n_facts: int, n_required: int,
             n_corrections: int, doc_tokens: int) -> BenchTask:
        spec = SPECS[family]
        seed = _stable_seed(self.seed, family, idx)
        rng = random.Random(seed)

        # facts
        facts: dict[str, str] = {}
        meta: dict[str, tuple[str, str, str]] = {}
        while len(facts) < n_facts:
            ent = f"{rng.choice(spec.entity_prefixes)}-{rng.randint(1, 40)}"
            met, scope = rng.choice(spec.metrics), rng.choice(spec.scopes)
            k = fact_key(ent, met, scope)
            if k in facts:
                continue
            facts[k] = str(rng.randint(*spec.value_range))
            meta[k] = (ent, met, scope)
        keys = list(facts)
        required = rng.sample(keys, min(n_required, len(keys)))
        corrected = rng.sample(required, min(n_corrections, len(required)))
        superseded: dict[str, str] = {}
        expected = dict(facts)
        for k in corrected:
            superseded[k] = facts[k]
            new = facts[k]
            while new == facts[k]:
                new = str(rng.randint(*spec.value_range))
            expected[k] = new

        def fact_sentence(k: str, v: str, correction: bool = False) -> str:
            ent, met, scope = meta[k]
            s = spec.fact_template.format(entity=ent, metric=met, scope=scope, value=f"{int(v):,}")
            return f"Correction: {s[0].lower()}{s[1:]} This supersedes the earlier figure." if correction else s

        events: list[TaskEvent] = [
            TaskEvent("system", f"You are a careful {spec.role} agent. Work through the material, keep the "
                      "evidence you need, and answer precisely with exact figures.", "protected"),
            TaskEvent("task", f"Task: review the {spec.role} material that follows. At the end you will be asked "
                      "for specific figures. Requirement: always report the latest corrected value.", "protected"),
        ]
        approx = 0
        n_docs = max(len(keys) + 2, stream_tokens // max(200, doc_tokens))
        fact_slots = rng.sample(range(n_docs), len(keys))
        slot_of = dict(zip(fact_slots, keys, strict=False))
        corr_slots = {k: rng.randint(max(int(n_docs * 0.6), fact_slots[keys.index(k)] + 1), n_docs + 1)
                      for k in corrected}
        prev_doc: TaskEvent | None = None
        for d in range(n_docs):
            phase = "subtask_complete" if d % 4 == 3 else ("mid_derivation" if d % 4 == 1 else "exploring")
            r = 1.0 if d in slot_of else rng.random()  # fact slots are always documents
            if r < 0.18:
                content = self._log(spec, rng, doc_tokens)
                ev = TaskEvent("tool_result", content, "compressible", f"tool:{family}-runner", phase=phase)
            elif r < 0.24 and prev_doc is not None:
                ev = TaskEvent(prev_doc.kind, prev_doc.content, prev_doc.category, prev_doc.source,
                               dict(prev_doc.metadata), phase, list(prev_doc.fact_keys))  # exact re-read
            else:
                body = self._doc(spec, rng, doc_tokens)
                fk: list[str] = []
                if d in slot_of:
                    k = slot_of[d]
                    body = _insert(body, fact_sentence(k, facts[k]), rng)
                    fk.append(k)
                kind = "user_message" if family == "multi_agent" else (
                    "document" if family in ("research", "analytics") else "tool_result")
                cat = "task_critical" if kind == "user_message" else "working"
                src = spec.doc_source.format(name=f"{rng.choice(spec.nouns)}_{d:03d}")
                ev = TaskEvent(kind, body, cat, src, {"key": fk[0]} if fk else {}, phase, fk)
                prev_doc = ev
            events.append(ev)
            approx += len(ev.content) // 4
            for k, slot in corr_slots.items():
                if slot == d + 1:
                    body = _insert(self._doc(spec, rng, doc_tokens // 2), fact_sentence(k, expected[k], True), rng)
                    events.append(TaskEvent("tool_result", body, "working",
                                            spec.doc_source.format(name=f"errata_{d:03d}"), {"key": k},
                                            "exploring", [k]))
        for k, slot in corr_slots.items():
            if slot > n_docs:
                events.append(TaskEvent("tool_result", fact_sentence(k, expected[k], True), "working",
                                        spec.doc_source.format(name="errata_final"), {"key": k}, "converging", [k]))
        ents = [meta[k] for k in required]
        ask = "; ".join(f"{m} for {s} reported by {e}" for e, m, s in ents)
        question = f"Question: what are the latest values of: {ask}? Give exact figures."
        return BenchTask(f"{family}-{self.seed}-{idx}", family, seed, events, question, required,
                         {k: expected[k] for k in required}, {k: v for k, v in superseded.items() if k in required},
                         approx)

    @staticmethod
    def _sentence(spec: FamilySpec, rng: random.Random) -> str:
        return (f"The {rng.choice(spec.adjectives)} {rng.choice(spec.nouns)} {rng.choice(spec.verbs)} the "
                f"{rng.choice(spec.adjectives)} {rng.choice(spec.nouns)} for the {rng.choice(spec.nouns)} review.")

    def _doc(self, spec: FamilySpec, rng: random.Random, tokens: int) -> str:
        parts: list[str] = []
        size = 0
        while size < tokens * 4:
            if rng.random() < 0.35:
                s = spec.table_line.format(n=rng.randint(100, 9999), acct=rng.randint(10, 99),
                                           a=f"{rng.randint(10, 9999):,}", b=rng.randint(10, 9999),
                                           c=rng.randint(1, 999))
            else:
                s = self._sentence(spec, rng)
            parts.append(s)
            size += len(s) + 1
        return "\n".join(parts)

    def _log(self, spec: FamilySpec, rng: random.Random, tokens: int) -> str:
        base = [spec.log_line.format(ts=f"2026-10-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00",
                                     n=rng.randint(1, 40), b=rng.randint(100, 999), c=rng.randint(1, 99))
                for _ in range(6)]
        lines: list[str] = []
        size = 0
        while size < tokens * 4:
            ln = base[rng.randrange(len(base))]
            lines.append(ln)
            size += len(ln) + 1
        return "\n".join(lines)


def _insert(body: str, sentence: str, rng: random.Random) -> str:
    lines = body.split("\n")
    lines.insert(rng.randint(1, max(1, len(lines) - 1)), sentence)
    return "\n".join(lines)


def _stable_seed(seed: int, family: str, idx: int) -> int:
    import hashlib

    return int(hashlib.sha256(f"{seed}:{family}:{idx}".encode()).hexdigest()[:8], 16)

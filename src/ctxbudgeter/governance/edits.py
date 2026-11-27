"""Edit governance — validating context edits an agent proposes.

*The agent proposes. ctxbudgeter governs. The harness executes.*

:class:`EditGovernor` reviews each :class:`EditProposal` against a declarative
:class:`ContextEditPolicy` and returns an :class:`EditDecision`: approve, modify
(e.g. redact a secret that leaked into a summary), or deny — always with
attributable :class:`EditViolation` records. The governor is pure and deterministic;
autonomy levels (observe / recommend / supervise / govern / adapt) are applied by the
runtime on top of its verdict.

A policy with ``enforce=False`` (see :meth:`ContextEditPolicy.unrestricted`) still
*detects* every violation but lets the edit through. That is the "unrestricted
agent" baseline: violations are then *committed* rather than *blocked*, which is
exactly what the benchmark needs to measure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

from ..runtime.analysis import retention, salient_terms
from ..runtime.events import ContextEvent
from ..runtime.rubric import CompactionRubric
from ..scanner import ContextScanner

EditOp = Literal["compact", "archive", "pin", "unpin", "recall", "restore", "checkpoint", "rollback"]
Severity = Literal["info", "warning", "error", "critical"]
Verdict = Literal["approve", "modify", "deny"]

PROVENANCE_PREFIX = "[ctx-provenance]"


@dataclass(frozen=True)
class EditViolation:
    rule: str
    severity: Severity
    message: str
    tags: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        return self.severity in ("error", "critical")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


@dataclass
class EditProposal:
    """Everything the governor needs to judge one edit, assembled by the runtime."""

    op: str
    targets: list[ContextEvent] = field(default_factory=list)
    usage: float = 0.0
    available_tokens: int = 1
    phase: str | None = None
    summary: str | None = None
    summary_tokens: int = 0
    recent_tags: tuple[str, ...] = ()
    pinned_tokens: int = 0
    used_tokens: int = 0
    restore_tokens: int = 0
    mutations_this_step: int = 0

    @property
    def target_tags(self) -> tuple[str, ...]:
        return tuple(e.tag for e in self.targets)

    @property
    def target_tokens(self) -> int:
        return sum(e.tokens for e in self.targets)


@dataclass
class EditDecision:
    verdict: Verdict
    violations: list[EditViolation] = field(default_factory=list)
    summary: str | None = None  # possibly rewritten summary (modify)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.verdict in ("approve", "modify")

    @property
    def critical(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "violations": [v.to_dict() for v in self.violations],
            "notes": dict(self.notes),
        }


@dataclass
class ContextEditPolicy:
    """Declarative rules for agent-initiated context edits.

    Usage thresholds are fractions of the *available* context budget.
    """

    enforce: bool = True
    protected_categories: tuple[str, ...] = ("protected",)
    min_usage_to_compact: float = 0.5
    compaction_trigger: float = 0.85
    hard_limit: float = 0.95
    warn_at: float = 0.85
    keep_recent_events: int = 2
    max_pinned_fraction: float = 0.3
    pinned_compactable: bool = False
    max_compaction_span_tokens: int | None = None
    max_compression_ratio: float | None = None
    min_fact_retention: float = 0.0
    """Minimum share of salient terms from *task-critical / pinned* events a summary keeps."""
    min_summary_retention: float = 0.0
    """Minimum share of salient terms from *all* compacted events a summary keeps."""
    scan_summaries: bool = True
    redact_summary_secrets: bool = True
    require_summary_provenance: bool = True
    max_mutations_per_step: int | None = 8
    allow_restore_over_limit: bool = False
    auto_checkpoint: bool = True
    fallback_compaction: bool = False
    """If True the runtime performs a governed, summarizer-based compaction at
    ``hard_limit`` when the agent has not acted (for avoidant agents)."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_usage_to_compact <= self.compaction_trigger <= self.hard_limit <= 1.0:
            raise ValueError("require 0 <= min_usage_to_compact <= compaction_trigger <= hard_limit <= 1")
        if not 0.0 <= self.max_pinned_fraction <= 1.0:
            raise ValueError("max_pinned_fraction must be within [0, 1]")
        if not 0.0 <= self.min_fact_retention <= 1.0 or not 0.0 <= self.min_summary_retention <= 1.0:
            raise ValueError("retention thresholds must be within [0, 1]")
        if self.keep_recent_events < 0:
            raise ValueError("keep_recent_events must be >= 0")

    @property
    def rubric(self) -> CompactionRubric:
        return CompactionRubric(
            min_usage=self.min_usage_to_compact,
            trigger=self.compaction_trigger,
            hard_limit=self.hard_limit,
        )

    @classmethod
    def unrestricted(cls) -> ContextEditPolicy:
        """Detect-only policy: every violation is recorded, nothing is blocked."""
        return cls(enforce=False, auto_checkpoint=False)

    @classmethod
    def strict(cls) -> ContextEditPolicy:
        return cls(
            min_usage_to_compact=0.7,
            keep_recent_events=4,
            max_compression_ratio=6.0,
            min_fact_retention=0.6,
            max_pinned_fraction=0.25,
            max_mutations_per_step=4,
        )

    def with_overrides(self, **overrides: Any) -> ContextEditPolicy:
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown edit-policy keys: {sorted(unknown)}")
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["protected_categories"] = list(self.protected_categories)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextEditPolicy:
        d = dict(data)
        if "protected_categories" in d:
            d["protected_categories"] = tuple(d["protected_categories"])
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown edit-policy keys: {sorted(unknown)}")
        return cls(**d)


class EditGovernor:
    """Pure, deterministic reviewer of :class:`EditProposal` objects."""

    def __init__(self, policy: ContextEditPolicy | None = None, *, scanner: ContextScanner | None = None) -> None:
        self.policy = policy or ContextEditPolicy()
        self.scanner = scanner or ContextScanner()

    # ------------------------------------------------------------------ public

    def review(self, p: EditProposal) -> EditDecision:
        handler = getattr(self, f"_review_{p.op}", None)
        if handler is None:
            return EditDecision("deny", [EditViolation("unknown_op", "error", f"unknown operation '{p.op}'")])
        violations, summary, notes = handler(p)
        violations = list(violations)
        if self.policy.max_mutations_per_step is not None and p.op in ("compact", "archive", "pin", "restore") \
                and p.mutations_this_step >= self.policy.max_mutations_per_step:
            violations.append(EditViolation(
                "mutation_rate", "error",
                f"{p.mutations_this_step} edits already this step (max {self.policy.max_mutations_per_step})"))
        blocking = [v for v in violations if v.blocking and v.rule != "secret_in_summary_redacted"]
        if blocking and self.policy.enforce:
            return EditDecision("deny", violations, None, notes)
        changed = summary is not None and summary != p.summary
        return EditDecision("modify" if changed else "approve", violations,
                            summary if summary is not None else p.summary, notes)

    # ---------------------------------------------------------------- per-op

    def _common_target_checks(self, p: EditProposal, verb: str) -> list[EditViolation]:
        pol = self.policy
        out: list[EditViolation] = []
        prot = tuple(e.tag for e in p.targets if e.category in pol.protected_categories)
        if prot:
            out.append(EditViolation(
                "protected_context", "critical",
                f"cannot {verb} protected context (system instructions, permissions, user requirements)", prot))
        pinned = tuple(e.tag for e in p.targets if e.pinned)
        if pinned and not (verb == "compact" and pol.pinned_compactable):
            out.append(EditViolation("pinned_context", "error", f"cannot {verb} pinned events; unpin first", pinned))
        recent = tuple(t for t in p.target_tags if t in p.recent_tags)
        if recent:
            out.append(EditViolation(
                "recent_context", "error",
                f"the {pol.keep_recent_events} most recent events stay verbatim; {verb} older events", recent))
        return out

    def _review_compact(self, p: EditProposal) -> tuple[list[EditViolation], str | None, dict[str, Any]]:
        pol = self.policy
        out = self._common_target_checks(p, "compact")
        notes: dict[str, Any] = {}
        if not p.targets:
            out.append(EditViolation("empty_range", "error", "no events in the requested range"))
            return out, p.summary, notes
        verdict = pol.rubric.evaluate(p.usage, p.phase)
        notes["rubric"] = verdict.to_dict()
        if verdict.decision == "block":
            out.append(EditViolation("premature_compaction", "error", "; ".join(verdict.reasons)))
        if pol.max_compaction_span_tokens is not None and p.target_tokens > pol.max_compaction_span_tokens:
            out.append(EditViolation(
                "compaction_span", "error",
                f"range holds {p.target_tokens:,} tokens (max {pol.max_compaction_span_tokens:,}); compact smaller spans"))

        summary = p.summary or ""
        ratio = p.target_tokens / max(1, p.summary_tokens)
        notes["compression_factor"] = round(ratio, 3)
        if pol.max_compression_ratio is not None and ratio > pol.max_compression_ratio:
            out.append(EditViolation(
                "over_compression", "error",
                f"summary is {ratio:.1f}x smaller than the range (max {pol.max_compression_ratio:.1f}x); keep more detail"))

        critical_terms: set[str] = set()
        all_terms: set[str] = set()
        for e in p.targets:
            terms = salient_terms(e.content)
            all_terms |= terms
            if e.category == "task_critical" or e.pinned:
                critical_terms |= terms
        overall = retention(all_terms, summary)
        notes["fact_retention"] = round(overall, 4)
        if all_terms and overall < pol.min_summary_retention:
            out.append(EditViolation(
                "summary_detail_loss", "error",
                f"summary keeps {overall:.0%} of the range's numbers/identifiers (min {pol.min_summary_retention:.0%})"))
        crit_ret = retention(critical_terms, summary)
        notes["critical_fact_retention"] = round(crit_ret, 4)
        if critical_terms and crit_ret < pol.min_fact_retention:
            out.append(EditViolation(
                "critical_fact_loss", "error",
                f"summary keeps {crit_ret:.0%} of task-critical facts (min {pol.min_fact_retention:.0%})",
                tuple(e.tag for e in p.targets if e.category == "task_critical")))

        new_summary: str | None = summary
        if pol.scan_summaries and summary:
            scan = self.scanner.scan(summary)
            if scan.has_secrets or scan.has_pii:
                what = "secret" if scan.has_secrets else "PII"
                if pol.redact_summary_secrets:
                    new_summary = self.scanner.redact(summary)
                    out.append(EditViolation("secret_in_summary_redacted", "warning",
                                             f"{what} found in summary — redacted before applying"))
                else:
                    out.append(EditViolation("secret_in_summary", "critical",
                                             f"{what} found in summary; it would re-enter the context"))
        if pol.require_summary_provenance and new_summary is not None and PROVENANCE_PREFIX not in new_summary:
            new_summary = new_summary.rstrip() + "\n" + provenance_footer(p.targets)
        return out, new_summary, notes

    def _review_archive(self, p: EditProposal) -> tuple[list[EditViolation], str | None, dict[str, Any]]:
        out = self._common_target_checks(p, "archive")
        if not p.targets:
            out.append(EditViolation("empty_range", "error", "no events to archive"))
        crit = tuple(e.tag for e in p.targets if e.category == "task_critical")
        if crit and p.usage < self.policy.min_usage_to_compact:
            out.append(EditViolation(
                "premature_archive", "error",
                f"archiving task-critical events at {p.usage:.0%} usage (< {self.policy.min_usage_to_compact:.0%}); "
                "space remains", crit))
        return out, None, {}

    def _review_pin(self, p: EditProposal) -> tuple[list[EditViolation], str | None, dict[str, Any]]:
        out: list[EditViolation] = []
        new_pinned = p.pinned_tokens + sum(e.tokens for e in p.targets if not e.pinned)
        frac = new_pinned / max(1, p.available_tokens)
        if frac > self.policy.max_pinned_fraction:
            out.append(EditViolation(
                "pin_budget", "error",
                f"pinned context would use {frac:.0%} of the budget (max {self.policy.max_pinned_fraction:.0%})",
                p.target_tags))
        return out, None, {"pinned_fraction_after": round(frac, 4)}

    def _review_unpin(self, p: EditProposal) -> tuple[list[EditViolation], str | None, dict[str, Any]]:
        return [], None, {}

    def _review_recall(self, p: EditProposal) -> tuple[list[EditViolation], str | None, dict[str, Any]]:
        return [], None, {}

    def _review_restore(self, p: EditProposal) -> tuple[list[EditViolation], str | None, dict[str, Any]]:
        out: list[EditViolation] = []
        after = (p.used_tokens + p.restore_tokens) / max(1, p.available_tokens)
        if after > self.policy.hard_limit and not self.policy.allow_restore_over_limit:
            out.append(EditViolation(
                "restore_over_limit", "error",
                f"restoring would raise usage to {after:.0%} (> hard limit {self.policy.hard_limit:.0%}); "
                "recall without restoring, or compact first"))
        return out, None, {"usage_after": round(after, 4)}

    def _review_checkpoint(self, p: EditProposal) -> tuple[list[EditViolation], str | None, dict[str, Any]]:
        return [], None, {}

    def _review_rollback(self, p: EditProposal) -> tuple[list[EditViolation], str | None, dict[str, Any]]:
        return [], None, {}


def provenance_footer(events: Sequence[ContextEvent]) -> str:
    tags = [e.tag for e in events]
    span = tags[0] if len(tags) == 1 else f"{tags[0]}..{tags[-1]}"
    sources = sorted({e.source for e in events if e.source})
    src = f"; sources: {', '.join(sources[:8])}" + (" …" if len(sources) > 8 else "") if sources else ""
    return f"{PROVENANCE_PREFIX} covers events {span}{src}; originals archived — recall by tag"

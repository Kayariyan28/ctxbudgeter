"""Policy tuning: turn an observed behaviour profile into an adaptive policy.

Level 2 ("behavioural") learning changes nothing in the model. It changes the
*environment* the model acts in: governance thresholds, fallback safety nets and the
guidance the agent is given — the levers NVIDIA's study showed move behaviour (a
timing instruction moved Kimi K3's first compaction from 29% back to ~84% of the
budget) and SelfCompact showed matter (the rubric is what makes self-compaction work).

Every tuning rule is explicit, attributable to a behaviour, and recorded in the
policy's ``evidence``. A tuned policy starts as a ``candidate``; it becomes
``validated`` only after it beats its baseline on held-out benchmark tasks (see
:mod:`ctxbudgeter.adaptive.optimizer`). Transfer to another model always produces a
new unvalidated candidate.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..governance.edits import ContextEditPolicy
from .profiler import ModelProfile

STATUSES = ("candidate", "validated", "rejected", "transfer_candidate")


@dataclass
class AdaptivePolicy:
    model: str
    policy: ContextEditPolicy
    guidance: list[str] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    rules_applied: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    parent_id: str | None = None
    status: str = "candidate"
    validation: list[dict[str, Any]] = field(default_factory=list)
    autonomy_cap: int | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def policy_id(self) -> str:
        payload = json.dumps({"model": self.model, "policy": self.policy.to_dict(), "guidance": self.guidance,
                              "version": self.version, "parent": self.parent_id}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def render_guidance(self, *, autonomy: int | None = None, brief: bool = False) -> str:
        from ..skills import render_guidance

        return render_guidance(policy=self.policy, autonomy=autonomy, adaptations=self.guidance, brief=brief)

    def transfer_to(self, model: str) -> AdaptivePolicy:
        """Copy this policy for another model as an *unvalidated* transfer candidate."""
        return AdaptivePolicy(
            model=model, policy=ContextEditPolicy.from_dict(self.policy.to_dict()), guidance=list(self.guidance),
            behaviors=list(self.behaviors), rules_applied=list(self.rules_applied),
            evidence={**self.evidence, "transferred_from": {"model": self.model, "policy_id": self.policy_id}},
            version=0, parent_id=self.policy_id, status="transfer_candidate", autonomy_cap=self.autonomy_cap,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id, "model": self.model, "version": self.version,
            "parent_id": self.parent_id, "status": self.status, "policy": self.policy.to_dict(),
            "guidance": list(self.guidance), "behaviors": list(self.behaviors),
            "rules_applied": list(self.rules_applied), "evidence": self.evidence,
            "validation": list(self.validation), "autonomy_cap": self.autonomy_cap, "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AdaptivePolicy:
        return cls(
            model=d["model"], policy=ContextEditPolicy.from_dict(d["policy"]), guidance=list(d.get("guidance", [])),
            behaviors=list(d.get("behaviors", [])), rules_applied=list(d.get("rules_applied", [])),
            evidence=dict(d.get("evidence", {})), version=int(d.get("version", 0)), parent_id=d.get("parent_id"),
            status=d.get("status", "candidate"), validation=list(d.get("validation", [])),
            autonomy_cap=d.get("autonomy_cap"), created_at=float(d.get("created_at", time.time())),
        )


class PolicyTuner:
    """Deterministic behaviour → policy rules. See module docstring."""

    def __init__(self, base: ContextEditPolicy | None = None) -> None:
        self.base = base or ContextEditPolicy()

    def tune(self, profile: ModelProfile, *, parent: AdaptivePolicy | None = None) -> AdaptivePolicy:
        pol = parent.policy if parent is not None else self.base
        ov: dict[str, Any] = {}
        guidance: list[str] = list(parent.guidance) if parent is not None else []
        rules: list[str] = []
        cap: int | None = None
        b = set(profile.behaviors)

        def add(g: str) -> None:
            if g not in guidance:
                guidance.append(g)

        if "avoidant" in b:
            ov["fallback_compaction"] = True
            ov["warn_at"] = min(pol.warn_at, 0.75)
            add("compaction_expected")
            rules.append("avoidant → governed fallback compaction at the hard limit; earlier pressure warning; "
                         "'compaction is expected' guidance")
        if "anxious" in b:
            floor = max(pol.min_usage_to_compact, 0.7)
            ov["min_usage_to_compact"] = floor
            ov["compaction_trigger"] = max(pol.compaction_trigger, floor, 0.85)
            add("timing_instruction")
            rules.append(f"anxious → usage floor raised to {floor:.0%}; explicit timing instruction")
        if "over_compressing" in b:
            ratio = min(pol.max_compression_ratio or 4.0, 4.0)
            ov["max_compression_ratio"] = ratio
            ov["min_summary_retention"] = max(pol.min_summary_retention, 0.4)
            add("preserve_detail")
            rules.append(f"over_compressing → compression capped at {ratio:g}x; summaries keep >= 40% of detail")
        if "lossy" in b:
            ov["min_fact_retention"] = max(pol.min_fact_retention, 0.6)
            ov["min_summary_retention"] = max(ov.get("min_summary_retention", pol.min_summary_retention), 0.4)
            add("pin_before_compact")
            rules.append("lossy → summaries must keep >= 60% of task-critical facts; pin before compacting")
        if "under_recovering" in b:
            add("use_archive_search")
            rules.append("under_recovering → archive-search-before-answering guidance")
        if "stale_keeping" in b:
            add("resolve_conflicts")
            rules.append("stale_keeping → resolve conflicts (keep latest value) guidance")
        if "unsafe" in b:
            ov["keep_recent_events"] = max(pol.keep_recent_events, 4)
            ov["max_mutations_per_step"] = min(pol.max_mutations_per_step or 4, 4)
            cap = 2
            rules.append("unsafe → stricter edit rate and recency rules; autonomy capped at L2 until re-certified")
        if ov and not pol.fallback_compaction and "fallback_compaction" not in ov:
            # Any tightening can leave an agent unable to compact; governance that denies
            # must also provide a safe path, or the agent overflows and loses its oldest events.
            ov["fallback_compaction"] = True
            rules.append("tightened policy → governed fallback compaction enabled as the safe path")
        if not rules:
            rules.append("no failure behaviours detected → base policy retained")

        tuned = pol.with_overrides(**ov) if ov else ContextEditPolicy.from_dict(pol.to_dict())
        return AdaptivePolicy(
            model=profile.model, policy=tuned, guidance=guidance, behaviors=sorted(b), rules_applied=rules,
            evidence={"profile": profile.to_dict(), "overrides": ov},
            version=(parent.version + 1) if parent is not None else 1,
            parent_id=parent.policy_id if parent is not None else None, autonomy_cap=cap,
        )

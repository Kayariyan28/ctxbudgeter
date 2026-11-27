"""Governance: pack-level policy enforcement and agent context-edit governance.

* :func:`enforce_policy` / :class:`PolicyViolationError` — the 0.3 compile-time
  ContextPolicy pass (unchanged; still importable from ``ctxbudgeter.governance``).
* :class:`EditGovernor` / :class:`ContextEditPolicy` — validates context edits an
  agent proposes at runtime (compact, archive, pin, restore, ...).
* :class:`AutonomyLevel` / :class:`AutonomyRegistry` — graduated context autonomy
  with evidence-based promotion and automatic demotion.
"""

from __future__ import annotations

from ._pack import PolicyViolationError, _mark_excluded, _set_redacted, enforce_policy
from .autonomy import (
    DEFAULT_CRITERIA,
    AutonomyCertifier,
    AutonomyLevel,
    AutonomyRegistry,
    CertificationResult,
    Evidence,
    LevelCriteria,
)
from .edits import (
    PROVENANCE_PREFIX,
    ContextEditPolicy,
    EditDecision,
    EditGovernor,
    EditProposal,
    EditViolation,
    provenance_footer,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "PROVENANCE_PREFIX",
    "AutonomyCertifier",
    "AutonomyLevel",
    "AutonomyRegistry",
    "CertificationResult",
    "ContextEditPolicy",
    "EditDecision",
    "EditGovernor",
    "EditProposal",
    "EditViolation",
    "Evidence",
    "LevelCriteria",
    "PolicyViolationError",
    "_mark_excluded",
    "_set_redacted",
    "enforce_policy",
    "provenance_footer",
]

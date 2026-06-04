"""Policy enforcement — the governance pass applied after compilation.

``enforce_policy`` runs deterministically over a compiled pack and:
  * scans every included item for PII/secrets (local, offline),
  * redacts or excludes items per the policy,
  * records source allow/deny, provenance, age, and item-token-cap violations,
  * recomputes token totals and the cacheable prefix after mutation,
  * stamps ``policy_summary`` / ``policy_violations`` / ``scanner_findings`` /
    ``item_risk`` onto the compiled pack (consumed by the BOM and reports).

It mutates the compiled pack in place and returns it. No secret ever leaves the
process unmasked — scanner findings carry only masked previews, and redaction
replaces the included content itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .policy import ContextPolicy, PolicyViolation
from .scanner import ContextScanner
from .tokenizer import TokenCounter

if TYPE_CHECKING:
    from .compiler import CompiledPack


class PolicyViolationError(ValueError):
    """Raised when fail_on_policy_violation=True and an error/critical breach occurs."""

    def __init__(self, violations: list[PolicyViolation]) -> None:
        self.violations = violations
        names = "; ".join(f"{v.rule} ({v.item_name or 'global'})" for v in violations)
        super().__init__(f"Policy enforcement failed on {len(violations)} violation(s): {names}")


def enforce_policy(
    compiled: CompiledPack,
    policy: ContextPolicy,
    *,
    model: str = "claude-sonnet-4.6",
    scanner: ContextScanner | None = None,
) -> CompiledPack:
    """Apply ``policy`` to ``compiled`` in place. Returns the same object."""
    scanner = scanner or ContextScanner()
    counter = TokenCounter(model)

    violations: list[PolicyViolation] = []
    scanner_findings: list[dict] = []
    item_risk: dict[str, str] = {}

    kept: list = []  # surviving included items, in order
    for item in list(compiled.included_items):
        name = item.name
        excluded = False
        keep_item = item  # may be replaced by a redacted copy below
        content = item.effective_content()

        # --- source allow/deny ---
        if policy.is_source_forbidden(item.source):
            violations.append(PolicyViolation(
                rule="forbidden_source", severity="critical", action="exclude",
                item_name=name,
                message=f"source '{item.source}' matches a forbidden-source pattern",
            ))
            excluded = True
        elif not policy.is_source_allowed(item.source):
            violations.append(PolicyViolation(
                rule="source_not_allowed", severity="error",
                action=("exclude" if not policy.allow_untrusted_sources else "flag"),
                item_name=name,
                message=f"source '{item.source}' is not in the allow-list",
            ))
            if not policy.allow_untrusted_sources:
                excluded = True

        # --- provenance requirement ---
        prov = item.effective_provenance()
        if policy.require_provenance and (prov is None or not prov.has_provenance()):
            violations.append(PolicyViolation(
                rule="missing_provenance", severity="warning", action="flag",
                item_name=name, message="item has no source/provenance metadata",
            ))

        # --- age / staleness ---
        if policy.max_age_days is not None and prov is not None:
            age = prov.age_days()
            if age is not None and age > policy.max_age_days:
                violations.append(PolicyViolation(
                    rule="stale_content", severity="warning", action="flag",
                    item_name=name,
                    message=f"content is {age:.0f} days old (> max_age_days={policy.max_age_days})",
                ))

        # --- per-item token cap ---
        item_tokens = compiled.included_tokens.get(name, 0)
        if policy.max_item_tokens is not None and item_tokens > policy.max_item_tokens:
            violations.append(PolicyViolation(
                rule="item_token_cap", severity="warning", action="flag",
                item_name=name,
                message=f"item uses {item_tokens:,} tokens (> max_item_tokens={policy.max_item_tokens:,})",
            ))

        # --- PII / secret scan ---
        scan = scanner.scan(content)
        if scan.findings:
            item_risk[name] = scan.risk_level
            for f in scan.findings:
                fd = f.to_dict()
                fd["item_name"] = name
                scanner_findings.append(fd)

        blocks_secret = policy.block_secrets and scan.has_secrets
        blocks_pii = policy.block_pii and scan.has_pii
        if blocks_secret or blocks_pii:
            what = "secrets" if blocks_secret else "PII"
            if policy.redact_sensitive:
                # Redact the content in place — never exclude required items silently.
                redacted = scanner.redact(content)
                new_tokens = counter.count(redacted)
                keep_item = _set_redacted(compiled, item, redacted, new_tokens)
                violations.append(PolicyViolation(
                    rule=f"blocked_{what}", severity="error", action="redact",
                    item_name=name, message=f"{what} detected — content redacted in place",
                ))
            else:
                violations.append(PolicyViolation(
                    rule=f"blocked_{what}", severity="critical", action="exclude",
                    item_name=name, message=f"{what} detected — item excluded by policy",
                ))
                if not item.required or not policy.fail_on_policy_violation:
                    excluded = True

        if excluded:
            _mark_excluded(compiled, name, item_risk.get(name, "none"))
        else:
            kept.append(keep_item)

    compiled.included_items = kept

    # --- recompute token totals + cacheable prefix after mutation ---
    used = sum(compiled.included_tokens.get(it.name, 0) for it in kept)
    compiled.used_tokens = used
    cacheable = 0
    for it in kept:
        if it.cache_policy == "stable":
            cacheable += compiled.included_tokens.get(it.name, 0)
        else:
            break
    compiled.cacheable_prefix_tokens = cacheable

    # --- stamp governance metadata ---
    compiled.policy_summary = policy.summary()
    compiled.policy_violations = [v.to_dict() for v in violations]
    compiled.scanner_findings = scanner_findings
    compiled.item_risk = item_risk

    # --- token-budget violation (global) ---
    if used > policy.available_context_tokens:
        compiled.policy_violations.append(PolicyViolation(
            rule="token_budget_exceeded", severity="error", action="flag",
            message=f"used {used:,} tokens > available {policy.available_context_tokens:,}",
        ).to_dict())

    # --- fail-fast if requested ---
    if policy.fail_on_policy_violation:
        blocking = [v for v in violations if v.severity in ("error", "critical")]
        if blocking:
            raise PolicyViolationError(blocking)

    return compiled


def _set_redacted(compiled: CompiledPack, item, redacted_text: str, new_tokens: int):
    """Replace an item's content with redacted text and update token bookkeeping.

    Returns the new (redacted) item so the caller can keep it in included_items.
    """
    new_item = item.model_copy(update={"compressed_content": redacted_text})
    compiled.items_by_name[item.name] = new_item
    compiled.included_tokens[item.name] = new_tokens
    # Reflect redaction in the decision row.
    new_decisions = []
    for d in compiled.decisions:
        if d.name == item.name:
            from dataclasses import replace

            new_decisions.append(replace(
                d, status="redacted", tokens=new_tokens,
                reason="redacted by policy (PII/secret detected)",
            ))
        else:
            new_decisions.append(d)
    compiled.decisions = new_decisions
    return new_item


def _mark_excluded(compiled: CompiledPack, name: str, risk: str) -> None:
    """Flip an item's decision to excluded due to policy."""
    from dataclasses import replace

    new_decisions = []
    for d in compiled.decisions:
        if d.name == name:
            new_decisions.append(replace(
                d, status="excluded", tokens=0,
                reason="excluded by policy (forbidden source / blocked sensitive content)",
            ))
        else:
            new_decisions.append(d)
    compiled.decisions = new_decisions

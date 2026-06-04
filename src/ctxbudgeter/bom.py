"""Context Bill of Materials (BOM).

A ``ContextBOM`` is a structured, auditable artifact describing exactly what was
compiled into a model call: which items were included/excluded/compressed/redacted,
their tokens, provenance, risk, cache policy, the governing policy summary, scanner
findings, and actionable recommendations.

The JSON form is deterministic (sorted keys, no wall-clock unless supplied) so it
can be committed and diffed in CI. The Markdown form is human-readable for PR review.
Secrets are never included — only masked scanner previews flow through.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compiler import CompiledPack

SCHEMA_VERSION = "1.0"


@dataclass
class BOMItem:
    """One included context item in the BOM."""

    name: str
    kind: str
    tokens: int
    priority: int
    cache_policy: str
    included_reason: str
    status: str = "included"
    relevance_score: float | None = None
    freshness: float | None = None
    retrieval_rank: int | None = None
    risk_level: str = "none"
    source: str | None = None
    trust_level: str | None = None
    transformations: list[str] = field(default_factory=list)
    checksum: str | None = None
    original_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BOMExcludedItem:
    name: str
    kind: str
    tokens: int
    priority: int
    excluded_reason: str
    source: str | None = None
    risk_level: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContextBOM:
    """A complete Context Bill of Materials."""

    run_id: str
    package_version: str
    model: str
    task: str | None
    total_tokens: int
    reserved_output_tokens: int
    available_context_tokens: int
    cacheable_tokens: int
    cache_efficiency_score: int
    context_health_score: int
    risk_score: int
    included_items: list[BOMItem] = field(default_factory=list)
    excluded_items: list[BOMExcludedItem] = field(default_factory=list)
    compressed_items: list[str] = field(default_factory=list)
    redacted_items: list[str] = field(default_factory=list)
    policy_summary: dict[str, Any] | None = None
    policy_violations: list[dict[str, Any]] = field(default_factory=list)
    scanner_findings: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    provenance_summary: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    schema_version: str = SCHEMA_VERSION

    # ----- construction -----------------------------------------------------

    @classmethod
    def from_compiled(
        cls,
        compiled: CompiledPack,
        *,
        task: str | None = None,
        created_at: str | None = None,
        run_id: str | None = None,
    ) -> ContextBOM:
        """Build a BOM from a compiled pack.

        Pulls policy summary, violations, scanner findings, and per-item risk from
        the compiled pack when the compiler recorded them (i.e. a policy was set).
        """
        from . import __version__
        from .provenance import content_checksum

        policy_summary = getattr(compiled, "policy_summary", None)
        policy_violations = list(getattr(compiled, "policy_violations", []) or [])
        scanner_findings = list(getattr(compiled, "scanner_findings", []) or [])
        item_risk: dict[str, str] = dict(getattr(compiled, "item_risk", {}) or {})
        resolved_task = task if task is not None else getattr(compiled, "task", None)

        decisions_by_name = {d.name: d for d in compiled.decisions}

        included: list[BOMItem] = []
        compressed: list[str] = []
        redacted: list[str] = []
        for it in compiled.included_items:
            d = decisions_by_name.get(it.name)
            status = d.status if d else "included"
            prov = it.effective_provenance()
            checksum = content_checksum(it.effective_content())
            included.append(BOMItem(
                name=it.name,
                kind=it.kind,
                tokens=compiled.included_tokens.get(it.name, d.tokens if d else 0),
                priority=it.priority,
                cache_policy=it.cache_policy,
                included_reason=d.reason if d else "included",
                status=status,
                relevance_score=round(it.relevance, 3),
                freshness=round(it.freshness, 3),
                retrieval_rank=(prov.retrieval_rank if prov else None),
                risk_level=item_risk.get(it.name, "none"),
                source=it.source,
                trust_level=(prov.trust_level if prov else None),
                transformations=(list(prov.transformation_history) if prov else []),
                checksum=checksum,
                original_tokens=(d.original_tokens if d else None),
            ))
            if status == "compressed":
                compressed.append(it.name)
            elif status == "redacted":
                redacted.append(it.name)

        excluded: list[BOMExcludedItem] = []
        for d in compiled.decisions:
            if d.status == "excluded":
                excluded.append(BOMExcludedItem(
                    name=d.name,
                    kind=d.kind,
                    tokens=d.original_tokens,
                    priority=d.priority,
                    excluded_reason=d.reason,
                    source=d.source,
                    risk_level=item_risk.get(d.name, "none"),
                ))

        cacheable = compiled.cacheable_prefix_tokens
        cache_eff = (
            round(100 * cacheable / compiled.used_tokens) if compiled.used_tokens else 0
        )
        risk_score = _aggregate_risk(scanner_findings, policy_violations, item_risk)
        recommendations = _build_recommendations(
            compiled, included, excluded, policy_violations, cache_eff, item_risk
        )
        provenance_summary = _provenance_summary(included)

        rid = run_id or _deterministic_run_id(compiled, resolved_task)

        return cls(
            run_id=rid,
            package_version=__version__,
            model=compiled.model,
            task=resolved_task,
            total_tokens=compiled.used_tokens,
            reserved_output_tokens=compiled.reserved_output_tokens,
            available_context_tokens=compiled.available_tokens,
            cacheable_tokens=cacheable,
            cache_efficiency_score=cache_eff,
            context_health_score=compiled.health_score,
            risk_score=risk_score,
            included_items=included,
            excluded_items=excluded,
            compressed_items=compressed,
            redacted_items=redacted,
            policy_summary=policy_summary,
            policy_violations=policy_violations,
            scanner_findings=scanner_findings,
            recommendations=recommendations,
            provenance_summary=provenance_summary,
            created_at=created_at,
        )

    # ----- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "package_version": self.package_version,
            "model": self.model,
            "task": self.task,
            "total_tokens": self.total_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "available_context_tokens": self.available_context_tokens,
            "cacheable_tokens": self.cacheable_tokens,
            "cache_efficiency_score": self.cache_efficiency_score,
            "context_health_score": self.context_health_score,
            "risk_score": self.risk_score,
            "policy_summary": self.policy_summary,
            "included_items": [i.to_dict() for i in self.included_items],
            "excluded_items": [i.to_dict() for i in self.excluded_items],
            "compressed_items": list(self.compressed_items),
            "redacted_items": list(self.redacted_items),
            "policy_violations": list(self.policy_violations),
            "scanner_findings": list(self.scanner_findings),
            "recommendations": list(self.recommendations),
            "provenance_summary": dict(self.provenance_summary),
        }

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str:
        """Return (and optionally write) deterministic JSON."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_markdown(self, path: str | None = None) -> str:
        text = _bom_markdown(self)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path: str) -> ContextBOM:
        from pathlib import Path

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextBOM:
        included = [BOMItem(**_filter(BOMItem, i)) for i in data.get("included_items", [])]
        excluded = [BOMExcludedItem(**_filter(BOMExcludedItem, i)) for i in data.get("excluded_items", [])]
        return cls(
            run_id=data.get("run_id", "unknown"),
            package_version=data.get("package_version", "unknown"),
            model=data.get("model", "unknown"),
            task=data.get("task"),
            total_tokens=int(data.get("total_tokens", 0)),
            reserved_output_tokens=int(data.get("reserved_output_tokens", 0)),
            available_context_tokens=int(data.get("available_context_tokens", 0)),
            cacheable_tokens=int(data.get("cacheable_tokens", 0)),
            cache_efficiency_score=int(data.get("cache_efficiency_score", 0)),
            context_health_score=int(data.get("context_health_score", 0)),
            risk_score=int(data.get("risk_score", 0)),
            included_items=included,
            excluded_items=excluded,
            compressed_items=list(data.get("compressed_items", [])),
            redacted_items=list(data.get("redacted_items", [])),
            policy_summary=data.get("policy_summary"),
            policy_violations=list(data.get("policy_violations", [])),
            scanner_findings=list(data.get("scanner_findings", [])),
            recommendations=list(data.get("recommendations", [])),
            provenance_summary=dict(data.get("provenance_summary", {})),
            created_at=data.get("created_at"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ----- helpers -----------------------------------------------------------------


def _filter(cls: type, data: dict) -> dict:
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in known}


def _aggregate_risk(
    scanner_findings: list[dict], policy_violations: list[dict], item_risk: dict[str, str]
) -> int:
    score = 0
    sev_w = {"low": 5, "medium": 15, "high": 30, "critical": 50}
    for f in scanner_findings:
        score += sev_w.get(str(f.get("severity", "low")), 5)
    vio_w = {"info": 2, "warning": 8, "error": 20, "critical": 40}
    for v in policy_violations:
        score += vio_w.get(str(v.get("severity", "warning")), 8)
    lvl_w = {"none": 0, "low": 3, "medium": 8, "high": 16, "critical": 25}
    for lvl in item_risk.values():
        score += lvl_w.get(lvl, 0)
    return min(100, score)


def _provenance_summary(items: list[BOMItem]) -> dict[str, Any]:
    by_trust: dict[str, int] = {}
    with_source = 0
    for i in items:
        t = i.trust_level or "unknown"
        by_trust[t] = by_trust.get(t, 0) + 1
        if i.source:
            with_source += 1
    return {
        "items_total": len(items),
        "items_with_source": with_source,
        "by_trust_level": dict(sorted(by_trust.items())),
    }


def _build_recommendations(
    compiled: CompiledPack,
    included: list[BOMItem],
    excluded: list[BOMExcludedItem],
    policy_violations: list[dict],
    cache_eff: int,
    item_risk: dict[str, str],
) -> list[str]:
    recs: list[str] = []
    if compiled.available_tokens and compiled.utilization > 0.95:
        recs.append("Context is near the budget ceiling (>95%); consider compressing low-relevance items.")
    if cache_eff < 30 and compiled.used_tokens > 1000:
        recs.append(
            "Low cache efficiency: move stable items (system rules, docs, tool defs) to the "
            "front with cache_policy='stable' to grow the cacheable prefix."
        )
    heavy_low_value = [
        i for i in included
        if i.tokens > 1500 and (i.relevance_score or 0) < 0.4 and i.priority < 60
    ]
    for i in heavy_low_value:
        recs.append(f"Item '{i.name}' is token-heavy ({i.tokens:,}) but low relevance — consider compress or exclude.")
    for name, lvl in sorted(item_risk.items()):
        if lvl in ("high", "critical"):
            recs.append(f"Item '{name}' carries {lvl} risk — redact or exclude before sending.")
    if any(v.get("severity") in ("error", "critical") for v in policy_violations):
        recs.append("Critical policy violations present — resolve before this context is used in production.")
    high_pri_excluded = [e for e in excluded if e.priority >= 80]
    for e in high_pri_excluded:
        recs.append(f"High-priority item '{e.name}' was excluded ({e.excluded_reason}); raise the budget or compress it.")
    return recs


def _deterministic_run_id(compiled: CompiledPack, task: str | None) -> str:
    import hashlib

    parts = [compiled.model, task or ""]
    for it in compiled.included_items:
        parts.append(f"{it.name}:{compiled.included_tokens.get(it.name, 0)}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"bom-{digest[:16]}"


def _bom_markdown(bom: ContextBOM) -> str:
    lines: list[str] = []
    lines.append("# Context Bill of Materials")
    lines.append("")
    lines.append(f"- **Run ID:** `{bom.run_id}`")
    lines.append(f"- **Model:** `{bom.model}`")
    if bom.task:
        lines.append(f"- **Task:** {bom.task}")
    lines.append(f"- **ctxbudgeter:** v{bom.package_version}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Total tokens | {bom.total_tokens:,} |")
    lines.append(f"| Available context tokens | {bom.available_context_tokens:,} |")
    lines.append(f"| Cacheable tokens | {bom.cacheable_tokens:,} |")
    lines.append(f"| Cache efficiency | {bom.cache_efficiency_score}/100 |")
    lines.append(f"| Context health | {bom.context_health_score}/100 |")
    lines.append(f"| Risk score | {bom.risk_score}/100 |")
    lines.append(f"| Included items | {len(bom.included_items)} |")
    lines.append(f"| Excluded items | {len(bom.excluded_items)} |")
    lines.append(f"| Redacted items | {len(bom.redacted_items)} |")
    lines.append(f"| Policy violations | {len(bom.policy_violations)} |")
    lines.append("")

    if bom.included_items:
        lines.append("## Included")
        lines.append("")
        lines.append("| Name | Kind | Tokens | Priority | Cache | Trust | Risk | Reason |")
        lines.append("|------|------|-------:|---------:|-------|-------|------|--------|")
        for i in bom.included_items:
            risk = f"**{i.risk_level}**" if i.risk_level in ("high", "critical") else i.risk_level
            lines.append(
                f"| `{i.name}` | {i.kind} | {i.tokens:,} | {i.priority} | {i.cache_policy} "
                f"| {i.trust_level or '-'} | {risk} | {i.included_reason} |"
            )
        lines.append("")

    if bom.excluded_items:
        lines.append("## Excluded")
        lines.append("")
        lines.append("| Name | Kind | Tokens | Priority | Reason |")
        lines.append("|------|------|-------:|---------:|--------|")
        for e in bom.excluded_items:
            lines.append(
                f"| `{e.name}` | {e.kind} | {e.tokens:,} | {e.priority} | {e.excluded_reason} |"
            )
        lines.append("")

    if bom.policy_violations:
        lines.append("## Policy Violations")
        lines.append("")
        for v in bom.policy_violations:
            who = f" (`{v.get('item_name')}`)" if v.get("item_name") else ""
            lines.append(f"- **[{v.get('severity', 'warning')}]** {v.get('rule')}: {v.get('message')}{who}")
        lines.append("")

    if bom.scanner_findings:
        lines.append("## Scanner Findings (masked)")
        lines.append("")
        for f in bom.scanner_findings:
            lines.append(
                f"- **[{f.get('severity')}]** {f.get('category')}: `{f.get('matched_preview')}` "
                f"— {f.get('recommendation')}"
            )
        lines.append("")

    if bom.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for r in bom.recommendations:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)

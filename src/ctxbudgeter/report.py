"""Reporting: render a CompiledPack as plain text, markdown, or JSON.

The plain-text format mirrors the spec example so it can be dropped into CI logs.
Markdown is for human-readable PR comments or docs. JSON is for tooling.
"""

from __future__ import annotations

import json

from .compiler import CompiledPack


def _included_in_prompt_order(pack: CompiledPack) -> list:
    """Return ItemDecision rows for included items, in actual prompt order."""
    by_name = {d.name: d for d in pack.decisions}
    return [by_name[it.name] for it in pack.included_items if it.name in by_name]


def _sensitivity_tag(d) -> str:
    if d.sensitivity == "secret":
        return " [!secret]"
    if d.sensitivity == "public":
        return " [public]"
    return ""


def to_text(pack: CompiledPack) -> str:
    """Plain-text report — fits the spec layout, drops into CI logs cleanly."""
    included = _included_in_prompt_order(pack)
    excluded = [d for d in pack.decisions if d.status == "excluded"]

    lines: list[str] = []
    lines.append("Included:")
    if not included:
        lines.append("  (none)")
    for d in included:
        notes: list[str] = []
        if d.required:
            notes.append("required")
        if d.cache_policy == "stable":
            notes.append("stable cache prefix")
        if d.status == "compressed":
            notes.append(f"compressed {d.original_tokens:,}→{d.tokens:,}")
        elif d.status == "truncated":
            notes.append(f"truncated {d.original_tokens:,}→{d.tokens:,}")
        elif d.status == "redacted":
            notes.append("REDACTED")
        notes.append(d.kind)
        tag = _sensitivity_tag(d)
        lines.append(f"  - {d.name}: {d.tokens:,} tokens, {', '.join(notes)}{tag}")

    if excluded:
        lines.append("")
        lines.append("Excluded:")
        for d in excluded:
            lines.append(f"  - {d.name}: {d.reason}")

    if pack.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in pack.warnings:
            lines.append(f"  ! {w}")

    lines.append("")
    lines.append(f"Estimated input tokens: {pack.used_tokens:,}")
    lines.append(f"Reserved output tokens: {pack.reserved_output_tokens:,}")
    lines.append(f"Cacheable prefix: {pack.cacheable_prefix_tokens:,} tokens")
    lines.append(
        f"Token budget: {pack.token_budget:,} "
        f"(utilization {pack.utilization * 100:.1f}%)"
    )
    lines.append(f"Context health score: {pack.health_score}/100")
    if pack.health_breakdown:
        deductions = ", ".join(f"{k}: {v:+d}" for k, v in sorted(pack.health_breakdown.items()))
        lines.append(f"  breakdown: {deductions}")
    lines.append(f"Tokenizer: {pack.tokenizer_backend}")
    return "\n".join(lines)


def to_markdown(pack: CompiledPack) -> str:
    """Markdown report — good for PR comments and docs."""
    included = _included_in_prompt_order(pack)
    excluded = [d for d in pack.decisions if d.status == "excluded"]

    lines: list[str] = []
    lines.append("# Context Compilation Report")
    lines.append("")
    lines.append(f"- **Model:** `{pack.model}`")
    lines.append(f"- **Health score:** **{pack.health_score}/100**")
    if pack.health_breakdown:
        bits = ", ".join(f"`{k}: {v:+d}`" for k, v in sorted(pack.health_breakdown.items()))
        lines.append(f"  - Breakdown: {bits}")
    lines.append(
        f"- **Tokens:** {pack.used_tokens:,} used / {pack.available_tokens:,} available "
        f"(budget {pack.token_budget:,}, output reserved {pack.reserved_output_tokens:,})"
    )
    lines.append(f"- **Cacheable prefix:** {pack.cacheable_prefix_tokens:,} tokens")
    lines.append(f"- **Utilization:** {pack.utilization * 100:.1f}%")
    lines.append(f"- **Tokenizer:** `{pack.tokenizer_backend}`")
    if pack.has_secrets:
        lines.append("- ⚠️ **Contains items tagged `sensitivity=secret`** — review before sending to external models.")
    lines.append("")

    if included:
        lines.append("## Included")
        lines.append("")
        lines.append(
            "| Name | Kind | Tokens | Cache | Priority | Sensitivity | Status | Source |"
        )
        lines.append(
            "|------|------|-------:|-------|---------:|-------------|--------|--------|"
        )
        for d in included:
            tok = (
                f"{d.tokens:,}"
                if d.tokens == d.original_tokens
                else f"{d.tokens:,} ({d.original_tokens:,} orig)"
            )
            src = d.source or ""
            sens = f"**{d.sensitivity}** ⚠️" if d.sensitivity == "secret" else d.sensitivity
            lines.append(
                f"| `{d.name}` | {d.kind} | {tok} | {d.cache_policy} | {d.priority} | {sens} | {d.status} | {src} |"
            )
        lines.append("")

    if excluded:
        lines.append("## Excluded")
        lines.append("")
        lines.append("| Name | Kind | Tokens | Priority | Reason |")
        lines.append("|------|------|-------:|---------:|--------|")
        for d in excluded:
            lines.append(
                f"| `{d.name}` | {d.kind} | {d.original_tokens:,} | {d.priority} | {d.reason} |"
            )
        lines.append("")

    if pack.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in pack.warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    return "\n".join(lines)


def to_json(pack: CompiledPack, *, indent: int = 2) -> str:
    """Machine-readable JSON report (round-trippable via compiled_pack_from_dict)."""
    return json.dumps(pack.to_dict(), indent=indent, default=str)

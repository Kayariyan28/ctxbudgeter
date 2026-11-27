"""ctxbudgeter.skills — portable context-management expertise (Level 1: instructional).

The ``context-management`` skill ships as package data in the standard SKILL.md
layout (frontmatter + body + ``references/``) so it can be:

* exported into any skills directory (Claude Code ``~/.claude/skills``, a project's
  ``.claude/skills``, or another harness that reads SKILL.md) with
  :func:`export_skill` / ``ctxbudgeter skill export``;
* rendered into a system-prompt section with :func:`render_guidance`, specialised to
  the live policy thresholds, the agent's autonomy level, and *behavioural
  adaptations* learned for its model (Level 2 — see :mod:`ctxbudgeter.adaptive`).
"""

from __future__ import annotations

import re
import shutil
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..governance.autonomy import AutonomyLevel
    from ..governance.edits import ContextEditPolicy

SKILL_NAME = "context-management"

#: Behavioural adaptations: targeted guidance added for a model whose observed
#: behaviour shows a specific failure mode. Keys are referenced by
#: :class:`ctxbudgeter.adaptive.AdaptivePolicy.guidance`.
ADAPTATIONS: dict[str, str] = {
    "timing_instruction": (
        "Timing: consider compacting only once usage reaches {trigger:.0%} of the budget "
        "(the floor is {floor:.0%}). Your recent sessions compacted while plenty of space "
        "remained; that discarded detail you later needed. Prefer to keep working."
    ),
    "compaction_expected": (
        "When you receive a context-pressure warning, compact completed work and continue. "
        "Do not end the task or answer early because the window is full — your recent "
        "sessions answered instead of compacting and lost accuracy."
    ),
    "preserve_detail": (
        "Summaries: aim for at most {max_ratio}x compression and keep every number, "
        "identifier and decision. Your recent summaries were compressed hard enough that "
        "archived originals had to be reopened often."
    ),
    "pin_before_compact": (
        "Before compacting, pin the events that hold evidence your final answer depends on. "
        "Recent compactions dropped task-critical facts."
    ),
    "use_archive_search": (
        "Before answering, search the archive for any entity or figure the question names "
        "that you no longer see (context_search_archive, then context_recall). Recent "
        "answers missed facts that were archived but never recalled."
    ),
    "resolve_conflicts": (
        "When a later event corrects an earlier one, keep only the corrected value. Run "
        "context_diagnose and archive 'conflict' findings before summarizing."
    ),
}

_AUTONOMY_TEXT = {
    0: "Autonomy L0 (observe): context edits are disabled; read-only calls still work.",
    1: "Autonomy L1 (recommend): your edits are recorded as recommendations; the harness decides.",
    2: "Autonomy L2 (supervised): your edits queue for approval before they apply.",
    3: "Autonomy L3 (governed): permitted edits apply immediately, with automatic checkpoints.",
    4: "Autonomy L4 (adaptive): permitted edits apply; policies tuned to your past behaviour are active.",
}


def skill_dir() -> Path:
    """Filesystem path of the bundled skill directory."""
    return Path(str(resources.files("ctxbudgeter.skills").joinpath(SKILL_NAME)))


def load_skill(*, include_frontmatter: bool = False) -> str:
    text = (skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    return text if include_frontmatter else strip_frontmatter(text)


def skill_metadata() -> dict[str, str]:
    text = (skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    meta: dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            key, sep, val = line.partition(":")
            if sep:
                meta[key.strip()] = val.strip()
    return meta


def strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\n.*?\n---\n+", "", text, count=1, flags=re.DOTALL)


def load_reference(name: str) -> str:
    path = skill_dir() / "references" / (name if name.endswith(".md") else f"{name}.md")
    return path.read_text(encoding="utf-8")


def export_skill(dest: str | Path, *, overwrite: bool = False) -> Path:
    """Copy the skill folder to ``dest/context-management`` and return its path."""
    target = Path(dest) / SKILL_NAME
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
        shutil.rmtree(target)
    shutil.copytree(skill_dir(), target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return target


def render_guidance(
    *,
    policy: ContextEditPolicy | None = None,
    autonomy: AutonomyLevel | int | None = None,
    adaptations: list[str] | tuple[str, ...] = (),
    brief: bool = False,
) -> str:
    """Render the skill as a system-prompt section.

    ``brief=True`` returns a compact (~300 token) operating summary for tight budgets;
    otherwise the full SKILL.md body is used. Policy thresholds, autonomy level and any
    learned adaptations are appended as a "Your operating parameters" section.
    """
    from ..governance.edits import ContextEditPolicy

    pol = policy or ContextEditPolicy()
    unknown = [a for a in adaptations if a not in ADAPTATIONS]
    if unknown:
        raise ValueError(f"unknown adaptations {unknown}; expected some of {sorted(ADAPTATIONS)}")
    body = _BRIEF if brief else load_skill()
    lines = [
        "## Your operating parameters",
        f"- Floor: don't compact below {pol.min_usage_to_compact:.0%} usage. Trigger: compact at a "
        f"boundary from {pol.compaction_trigger:.0%}. Hard limit: compact now at {pol.hard_limit:.0%}.",
        f"- The {pol.keep_recent_events} most recent events stay verbatim; pins may use up to "
        f"{pol.max_pinned_fraction:.0%} of the budget.",
    ]
    if pol.max_compression_ratio:
        lines.append(f"- Summaries may compress at most {pol.max_compression_ratio:g}x.")
    if pol.min_fact_retention:
        lines.append(f"- Summaries must keep at least {pol.min_fact_retention:.0%} of task-critical facts.")
    if autonomy is not None:
        lines.append(f"- {_AUTONOMY_TEXT[int(autonomy)]}")
    for a in adaptations:
        lines.append("- " + ADAPTATIONS[a].format(
            trigger=pol.compaction_trigger, floor=pol.min_usage_to_compact,
            max_ratio=f"{pol.max_compression_ratio:g}" if pol.max_compression_ratio else "4"))
    return body.rstrip() + "\n\n" + "\n".join(lines) + "\n"


_BRIEF = """# Context management (brief)

Your context is a governed budget. Every event shows a tag like [e7 tool_result];
summaries carry range tags like 3..17, and originals stay recoverable by tag.

Loop, when usage is high or a sub-task finishes:
1. context_budget — follow its timing verdict: block (space remains, keep working),
   defer (finish the current derivation), allow / compact_now (go ahead).
2. context_diagnose — archive duplicates and superseded/conflicting facts first.
3. context_pin the evidence your answer needs; then context_plan / context_compact
   finished work. Never target protected events (system, task, requirements).
4. Summaries keep every number, identifier, path, decision and correction (latest value
   only), with source tags. Aim for 3-5x compression.
5. Check the result (status, fact_retention). context_rollback if something was lost.
6. On a pressure warning: compact and continue — never stop early because the window is full.
7. Before answering, context_search_archive / context_recall anything the question
   names that you no longer see.
Denied edits explain why (e.g. premature_compaction, over_compression): adjust, don't retry.
"""

__all__ = [
    "ADAPTATIONS",
    "SKILL_NAME",
    "export_skill",
    "load_reference",
    "load_skill",
    "render_guidance",
    "skill_dir",
    "skill_metadata",
    "strip_frontmatter",
]

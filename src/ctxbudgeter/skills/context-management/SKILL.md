---
name: context-management
description: Manage your own context window as a limited, governed resource — decide when and what to compact, pin, archive and recall, using a context API (inspect, budget, diagnose, plan, compact, pin, archive, recall, checkpoint, rollback, evaluate). Use this skill whenever you are a long-running agent whose history of tool calls, documents and turns is growing; when you receive a context-pressure warning; when usage is reported above ~50%; before reading large documents or logs; when a question refers to something you no longer see in context; or whenever context_* / ctx.* tools are available — even if nobody asks you to manage context explicitly.
---

# Context management

Your context window is working memory with a hard limit. Everything in it is re-read
on every request, so stale logs cost tokens and attention every turn ("context rot"),
while discarding the wrong thing loses evidence you cannot reconstruct. This skill
teaches you to manage that trade-off yourself, through a governed context API, so you
keep what matters, compress what is finished, and can always get originals back.

Research behind this skill: models that use a context API well reach higher accuracy
with fewer tokens, especially under tight budgets (NVIDIA, *Letting Models Manage Their
Own Context*, 2026); a simple timing rubric is what makes self-compaction work
(Apple, *SelfCompact*, 2026). The failure modes are equally well documented: never
compacting and giving up when the window fills, compacting far too early while space
remains ("context anxiety"), and compressing so hard you must keep reopening archives.

## How the API works

Every event in your context has a **tag** shown as `[e7 tool_result]`. A summary that
replaced events 3 to 17 carries the range tag `3..17`. Compaction is non-destructive:
the originals go to an archive and stay retrievable by their original tag.

| Python (`ctx.`)              | Tool name            | Use it to                                        |
|------------------------------|----------------------|--------------------------------------------------|
| `inspect()`                  | `context_inspect`    | see composition by category and every event      |
| `budget()`                   | `context_budget`     | check usage, pressure level and the timing verdict |
| `diagnose()`                 | `context_diagnose`   | find duplicates, superseded facts, stale/huge logs, secrets |
| `plan(target_usage=)`        | `context_plan`       | get a cache-aware list of edits that reach a target |
| `compact(start, end, summary)` | `context_compact`  | replace a range with your summary                |
| `pin(tag, reason)` / `unpin` | `context_pin`        | protect evidence you will need verbatim          |
| `archive(*tags)`             | `context_archive`    | move inactive events out of context              |
| `recall(tag, restore=False)` | `context_recall`     | read an archived original (optionally restore it) |
| `search_archive(query)`      | `context_search_archive` | find archived events by keyword              |
| `checkpoint(label)`          | `context_checkpoint` | snapshot before a risky edit                     |
| `rollback(id)`               | `context_rollback`   | undo an edit that lost something                 |
| `evaluate(trace_id=None)`    | `context_evaluate`   | measure how good a decision was                  |
| `signal(phase)`              | `context_signal`     | declare your phase so timing advice fits         |

Full argument reference and result shapes: `references/api.md`.

## Know your context categories

Every event has a category. It tells you — and the governor — how much care it needs.

- **protected** — system instructions, permissions, the user's requirements. You cannot
  compact or archive these, and you should not want to: they define the task.
- **task_critical** — current objectives, evidence, dependencies, user messages, notes.
  Keep verbatim, or summarize only if every number, identifier and decision survives.
- **working** — current plans, reasoning summaries, recent tool results.
- **compressible** — finished steps, verbose logs, redundant output. Compact first.
- **archivable** — inactive information you might want later. Archive freely.

## The decision loop

Run this loop when `budget()` shows `pressure` of `moderate` or above, when you finish
a sub-task, or when you get a context-pressure warning.

1. **Check timing first.** `budget()["timing"]["decision"]` applies the rubric below.
   `block` means space remains — keep working. `defer` means finish your current
   derivation first. `allow` / `compact_now` means go ahead.
2. **Diagnose.** `diagnose()` finds cheap, nearly lossless wins: exact duplicates,
   facts superseded by a later correction, repetitive logs. Archive those first.
3. **Pin what you will need verbatim.** Before compacting, pin the few events that hold
   evidence your final answer depends on (figures, the failing test, the user's
   constraint). Pins have a budget (about 30% of the window), so pin evidence, not bulk.
4. **Plan.** `plan()` proposes compactions over contiguous finished work, excluding
   protected, pinned and the most recent events. It is cache-aware: edits near the end
   of the context are cheap to serve, edits near the start invalidate the whole KV
   prefix after them, so prefer fewer, larger edits made in one turn.
5. **Compact with a good summary** (see below). Prefer writing the summary yourself —
   you know what matters for the task better than an extractive default.
6. **Verify.** Read the result: `status`, `decision.notes.fact_retention`, and any
   violations. If something important was lost, `rollback()` and redo it.
7. **Continue the task.** Compaction is routine maintenance, not a signal to stop.

## When to compact — and when not to

The timing rubric (fractions of your available budget; your harness may tune them):

| Situation | What to do |
|-----------|-----------|
| usage below ~50% | Don't compact. Archiving exact duplicates is fine. Compacting now throws away context you can still afford — the "context anxiety" failure that cost some models ~8 accuracy points. |
| you are mid-derivation or stuck debugging | Defer. The trail you'd compress is exactly what you need to finish or to see why you're stuck. |
| a sub-task just finished, or you're converging | Good moment, once usage is past the floor. |
| usage at or above ~85% | Compact at the next natural boundary. |
| usage at or above ~95% | Compact now, whatever the phase. Past the limit the harness drops your oldest events irrecoverably. |

Signal your phase with `ctx.signal("subtask_complete")` (or `exploring`,
`mid_derivation`, `stuck`, `converging`, `answering`) so the rubric can help.

**When the context-pressure warning arrives, compact and keep going.** Do not answer
early just because the window is full — an unfinished task with a full window is a
context problem, not a reason to stop. One strong model in the NVIDIA study answered
instead of compacting in 88% of warnings and trailed the baseline at tight budgets.

## Writing a summary that keeps what matters

A summary replaces evidence. Write it for your future self, who will no longer see the
originals:

- Keep every number, identifier, file path, URL, function name and quoted string that
  could matter. These are the facts you cannot infer back; the governor measures how
  many you kept (`fact_retention`) and may refuse a summary that drops task-critical ones.
- Keep decisions and their reasons, open questions, and dead ends ("tried X, failed
  because Y") so you don't repeat them.
- When a later event corrected an earlier one, keep **only the corrected value** and say
  it supersedes the earlier one. Stale values that survive a summary cause wrong answers.
- Reference source tags (`[e12]`) so you know where to recall from. A provenance
  footer listing covered events and sources is added for you automatically.
- Aim for roughly 3-5x compression of verbose material. Much harder compression tends
  to force you to reopen archives later, which costs more than it saved.

Good and bad examples of real decisions: `references/examples.md`.

## Recovering archived information

Before answering, check that every fact the answer needs is visible. If the question
mentions an entity, figure or file you no longer see:

1. `search_archive("ALPHA-7 receipts FY2019")` — finds archived originals by keyword.
2. `recall(tag)` — read it without restoring (cheap; does not grow your context).
3. `recall(tag, restore=True)` — bring it back into context only if you'll keep using it.

Reopening a few archives is healthy — it means compaction was bold but recoverable.
Reopening most of them means you are compressing too hard.

## How governance responds

You propose; ctxbudgeter governs; the harness executes. Every edit is validated:

- **Denied** edits come back with a reason (`protected_context`, `pinned_context`,
  `recent_context`, `premature_compaction`, `over_compression`, `critical_fact_loss`,
  `pin_budget`, `restore_over_limit`). Treat the reason as feedback: adjust the range,
  keep more detail, or wait — don't retry the same call.
- **Modified** edits were applied with a correction, e.g. a secret that leaked into your
  summary was redacted.
- **Recommended** (autonomy L1) or **pending** (L2) edits were not executed: your harness
  or a human decides. Keep working with the context you have.
- Destructive edits are checkpointed automatically at L3+, so `rollback()` is always
  available after a bad compaction.

Your autonomy level can rise as you demonstrate good context decisions on held-out
evaluations, and drops automatically after a critical violation (such as trying to
remove protected context). Good management is noticed.

## Quick checklist

- [ ] Usage past the floor and at a boundary (or past the trigger)? Otherwise keep working.
- [ ] Duplicates / superseded facts archived first?
- [ ] Final-answer evidence pinned?
- [ ] Summary keeps numbers, identifiers, decisions, corrections, source tags?
- [ ] Result checked (status, retention, violations)?
- [ ] Before answering: anything the question needs that is only in the archive?

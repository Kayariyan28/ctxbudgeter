# Context API reference

Every call returns JSON. Mutating calls return an *operation result*:

```json
{
  "op": "compact",
  "status": "applied | denied | recommended | pending | not_found | noop",
  "message": "why it was denied / queued, if it was",
  "trace_id": "9f2c...",
  "proposal_id": "p4 (only when pending)",
  "decision": {"verdict": "approve | modify | deny", "violations": [...], "notes": {...}},
  "details": {"summary_tag": "3..9", "summary_tokens": 212, "covers": ["3", "...", "9"]},
  "tokens_before": 27410,
  "tokens_after": 19876
}
```

## Contents

1. Read operations: inspect, budget, diagnose, plan, search_archive
2. Write operations: compact, pin, unpin, archive, recall, checkpoint, rollback, signal
3. evaluate
4. Violation rules

---

## 1. Read operations

### `context_inspect()` / `ctx.inspect(preview_chars=120)`
Returns `budget`, `composition.by_category` and `composition.by_kind` (events and
tokens), `events` (tag, kind, category, tokens, pinned, source, step, covers, preview),
`archived` count, `phase`, `autonomy`, and `pending_approvals`.

### `context_budget()` / `ctx.budget()`
`budget`, `reserved_output_tokens`, `available`, `used`, `free`, `usage` (0-1 of
available), `pressure` (`low` < floor ≤ `moderate` < trigger ≤ `high` < hard ≤
`critical`), `pinned_tokens`, `thresholds`, and `timing` — the rubric verdict
(`block | defer | allow | compact_now`) with reasons.

### `context_diagnose()` / `ctx.diagnose()`
List of findings, most severe first:

| kind | meaning | suggested op |
|------|---------|--------------|
| `sensitive` | secret / PII resent every request | ask the harness to redact |
| `duplicate` | identical content appears later | archive older copies |
| `conflict` | same `metadata.key`, values disagree with the latest | archive the stale ones |
| `superseded` | same key, superseded by a later event | archive |
| `oversized` | one event > 15% of the budget | compact it |
| `repetitive` | mostly repeated lines (logs) | compact it |
| `near_duplicate` | ≥85% shingle overlap with a later event | archive |
| `stale` | old tool output never referenced since | compact |

Each finding has `tags`, `tokens_recoverable`, `message`, `suggestion`.

### `context_plan(target_usage=None, cache_aware=True)` / `ctx.plan(...)`
`current_usage`, `target_usage`, `projected_usage`, `tokens_to_free`, `timing`, `ops`
(each `{op, args, est_tokens_saved, recompute_cost, score, rationale}`) and `notes`.
`recompute_cost` is the tokens a prefix-caching engine recomputes because of the edit.
Execute with `ctx.apply_plan(plan)` or call each op yourself.

### `context_search_archive(query, limit=5)`
Archived originals ranked by keyword match: `tag, score, tokens, kind, source, preview`.

## 2. Write operations

### `context_compact(start, end, summary=None, target_tokens=None)`
Replaces active events `start..end` (inclusive, by tag, in prompt order) with one
summary event tagged `start..end`. Omit `summary` to use the deterministic extractive
summarizer (keeps the sentences richest in numbers, identifiers and decisions).
Originals are archived and recallable by their tags. `decision.notes` reports
`compression_factor`, `fact_retention`, `critical_fact_retention`, and the rubric verdict.

### `context_pin(tag, reason)` / `context_unpin(tag)`
Pinned events cannot be compacted or archived. Pins are limited to
`max_pinned_fraction` of the budget (default 30%).

### `context_archive(tags)`
Moves events out of active context, recoverable by tag. Archiving task-critical events
below the usage floor is refused.

### `context_recall(tag, restore=False)`
Returns `content`, `kind`, `source`, `covers` and `originals` (the original tags behind
a summary). `restore=True` re-inserts the event at its original position, subject to
the hard limit. Recalling marks the compaction that archived it as *reopened*.

### `context_checkpoint(label=None)` / `context_rollback(checkpoint_id=None)`
Checkpoints snapshot the active context. `rollback()` without an id restores the most
recent checkpoint (including the automatic ones taken before destructive edits at
autonomy L3+). The archive is append-only, so rollback never loses archived originals.

### `context_signal(phase)`
Declare `exploring | mid_derivation | stuck | subtask_complete | converging | answering`.
Returns the new phase and the updated timing verdict. Signalling `subtask_complete`
when you finish a unit of work lets the rubric allow compaction past the floor;
`mid_derivation` / `stuck` defer it until the hard limit.

## 3. `context_evaluate(trace_id=None)`

With a trace id: `usage_at_decision`, `timing`, `compression_factor`,
`fact_retention`, `critical_fact_retention`, `reopened`, `violations`.

Without: session totals — `compaction_attempts`, `compactions_applied`,
`median_compression_factor`, `mean_fact_retention`, `reopen_rate`,
`first_compaction_usage`, `early_compaction_attempts`, `denied`,
`committed_violations`, `critical_violations`, `overflow_events_dropped`,
`fallback_compactions`, and prefill `cost` under prefix caching vs suffix KV reuse.

## 4. Violation rules

| rule | severity | fix |
|------|----------|-----|
| `protected_context` | critical | never target system / task / requirement events |
| `pinned_context` | error | unpin first, or leave it |
| `recent_context` | error | compact older events; the newest stay verbatim |
| `premature_compaction` | error | wait until usage passes the floor |
| `premature_archive` | error | same, for task-critical events |
| `over_compression` | error | keep more detail in the summary |
| `critical_fact_loss` | error | keep task-critical numbers / identifiers |
| `compaction_span` | error | compact a smaller range |
| `pin_budget` | error | pin less, or unpin something |
| `restore_over_limit` | error | recall without restore, or compact first |
| `mutation_rate` | error | too many edits this turn; continue the task |
| `secret_in_summary_redacted` | warning | your summary contained a secret; it was redacted |

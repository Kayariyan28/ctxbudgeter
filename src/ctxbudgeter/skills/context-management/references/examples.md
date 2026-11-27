# Context decisions: good and bad

Each example shows a state, a decision, and why it is good or bad. They mirror the
behaviours measured in NVIDIA's study and the failure modes the governor checks for.

---

## 1. Compacting at a sub-task boundary (good)

State: usage 87%, phase `subtask_complete`. Events 4-15 are tool results from reading
six budget documents; you have extracted the three figures you need.

```python
ctx.pin("9", "FY2019 receipts figure for ALPHA-7, needed in the final answer")
ctx.compact("4", "15", summary=(
    "Read FY2019 Treasury statements for agencies ALPHA-3..ALPHA-9 [e4-e15]. "
    "ALPHA-7 receipts FY2019 = 3,512 million USD (corrected in [e13]; supersedes 3,461). "
    "ALPHA-3 = 1,204; ALPHA-5 = 988. Outlays tables in [e6], [e11] not needed. "
    "Open question: whether ALPHA-7 includes trust-fund transfers (see [e14])."))
```

Why it's good: it happens at a natural boundary past the trigger. It keeps every figure
plus the correction (and drops the superseded value), records an open question, points
at source tags, and pins the one event the answer depends on.

## 2. Compacting at 30% usage (bad — context anxiety)

State: usage 31%, phase `exploring`.

```python
ctx.compact("3", "10")   # -> denied: premature_compaction
```

Why it's bad: plenty of space remains. Compacting now throws away detail you can still
afford and will likely need. In the NVIDIA study, models that compacted this early
without a timing instruction scored about 8 points lower at large budgets. Keep working;
archive only exact duplicates if you want to tidy.

## 3. Answering instead of compacting (bad)

State: context-pressure warning at 93%; the task needs two more documents.

Bad: "Given the context limit, here is my best answer so far…"

Good:

```python
plan = ctx.plan()
ctx.apply_plan(plan)          # frees ~40% at once, in one turn (one KV-prefix invalidation)
# …then read the remaining two documents and answer.
```

Why: a full window is a maintenance task, not a stopping condition.

## 4. Over-compression (bad)

```python
ctx.compact("3", "40", summary="Read lots of documents about receipts.")
# -> denied: over_compression (38x) and critical_fact_loss (0% of task-critical facts kept)
```

Why: the summary keeps none of the numbers or identifiers. Even if allowed, you would
have to reopen nearly every archived event later.

## 5. Keeping a stale value (bad)

Events: `[e8]` "ALPHA-7 receipts FY2019 of 3,461 million USD", later `[e19]`
"Correction: … 3,512 million USD (supersedes earlier figure)".

Bad summary: "ALPHA-7 receipts were 3,461 million USD." The correction is gone and the
final answer will be wrong.

Good: archive `[e8]` (diagnose reports it as `conflict`), keep `[e19]`.

## 6. Recovering before answering (good)

Question: "Compare ALPHA-7 and ALPHA-12 FY2019 receipts." ALPHA-12 is not visible.

```python
hits = ctx.search_archive("ALPHA-12 receipts FY2019")
ctx.recall(hits[0]["tag"])            # read it; no need to restore for a one-off lookup
```

## 7. Trying to remove protected context (critical)

```python
ctx.archive("1")   # the system instructions
# -> denied: protected_context (critical). Counts against your autonomy level.
```

## 8. Rolling back a bad edit (good)

```python
r = ctx.compact("20", "31")
if r.decision["notes"]["critical_fact_retention"] < 0.8:
    ctx.rollback()                  # auto-checkpoint taken before the compaction
    ctx.pin("27", "test failure trace")
    ctx.compact("20", "26")
```

# Context Intelligence — the Adaptive Context Engine (0.4)

> **The agent proposes. ctxbudgeter governs. The harness executes. The evaluator teaches.**

Up to 0.3, ctxbudgeter decided what an agent was *allowed to know* before each model
call. 0.4 adds the other half: letting agents manage their **own** context while it
evolves — and making that safe, measurable and learnable, independent of the model.

This document covers the architecture, the research it implements, every module, and
how to integrate it. Benchmark results are in [`docs/ctxbench/`](ctxbench/).

## Contents

1. [Why](#why)
2. [Architecture](#architecture)
3. [The context API (`ctxbudgeter.runtime`)](#the-context-api)
4. [Governance and graduated autonomy (`ctxbudgeter.governance`)](#governance-and-graduated-autonomy)
5. [The context-management skill (`ctxbudgeter.skills`)](#the-context-management-skill)
6. [Behavioural learning (`ctxbudgeter.adaptive`)](#behavioural-learning)
7. [CTXBench (`ctxbudgeter.bench`)](#ctxbench)
8. [Training data (`ctxbudgeter.train`)](#training-data)
9. [Integrations (`ctxbudgeter.adapters`)](#integrations)
10. [Limitations](#limitations)

## Why

NVIDIA's *Letting Models Manage Their Own Context: Where We Stand* (arXiv 2611.34147)
compared harness-driven compaction (summarize at a threshold) with model-driven context
management through a context API, across 8K–512K budgets:

* Models that use the API well reach higher accuracy, often with fewer tokens, most at
  tight budgets — GPT-6.1 Sol nearly triples its score at 32K on OfficeQA Pro v2.
* Models differ widely: some barely use the API (Claude Opus 5.5 compacts in 4% of tasks
  and answers early under pressure), some compact hard and keep reopening archives, some
  compact far too early when not told when to ("context anxiety").
* Inference engines can make edits cheaper by reusing the KV cache of the unchanged
  suffix after an edit (up to 15% less prefill).

Complementary work: SelfCompact (Apple) shows a lightweight timing rubric is what makes
self-compaction beat fixed-interval summarization; Context Language Models (UW/Meta)
treat the context as an editable file and add Suffix Cache Reuse; MEMENTO trains models
to fold their own reasoning.

The gap these leave for infrastructure: a **model-agnostic** layer that gives any agent a
context API, keeps it from destroying what it must not, measures how well it manages
context, and improves that behaviour for the specific model in use — without retraining.

## Architecture

```text
             ┌──────────── skills (L1: instructional) ─────────────┐
             │ SKILL.md + references, rendered with live policy,    │
             │ autonomy level and model-specific adaptations        │
             └──────────────────────────┬──────────────────────────┘
                                        │ guidance
  agent / model ──tool calls──▶ adapters (OpenAI · Anthropic · MCP · ContextHarness)
                                        │ proposals
                                        ▼
                      runtime.ContextRuntime ──▶ governance.EditGovernor
                      events · archive · rubric      ContextEditPolicy + AutonomyLevel
                      planner · cache cost model     approve / modify / deny / queue
                                        │
                              DecisionTrace (JSONL)
                     ┌──────────────────┼───────────────────┐
                     ▼                  ▼                   ▼
     adaptive (L2: behavioural)   bench (CTXBench)    train (L3: parametric)
     profile → tune → optimize    4 modes · CIS       SFT · DPO pairs · rewards
     → validate → version         autonomy evidence
                     │
                     └──▶ tuned ContextEditPolicy + guidance back into the runtime
```

`ctxbudgeter.core` remains the deterministic compile-time layer;
`ContextRuntime.to_context_pack()` bridges live context into it, so ContextPolicy,
scanning, BOM, cache planning and Context MRI apply to a live agent context too.

## The context API

`ContextRuntime` holds the agent's working context as tagged `ContextEvent`s
(`[e7 tool_result]`). A summary that replaced events 3–17 is tagged `3..17` (NOOA's
convention), and originals go to an append-only archive, recoverable by tag.

| Operation | What it does |
|---|---|
| `inspect()` | composition by category and kind, every event |
| `budget()` | used / free / usage, pressure level, rubric timing verdict |
| `diagnose()` | duplicates, superseded and conflicting facts, stale, oversized, repetitive and sensitive events — each with recoverable tokens and a suggested op |
| `plan()` | cache-aware list of compact/archive ops reaching a target; never touches protected, pinned or recent events; avoids re-summarizing summaries |
| `compact(start, end, summary)` | governed range → summary; originals archived; provenance footer added |
| `pin` / `unpin` | protect evidence (pin budget applies) |
| `archive(*tags)` | move out of context, recoverable |
| `recall(tag, restore=False)` | read an archived original; `restore=True` re-inserts it (governed) |
| `search_archive` / `recall_hints` | find archived originals by keyword / by the question's identifiers |
| `checkpoint` / `rollback` | snapshot and revert (auto-checkpoint before destructive edits at L3+) |
| `evaluate(trace_id=None)` | one decision (compression, retention, timing, reopened) or the session |
| `signal(phase)` | declare `exploring / mid_derivation / stuck / subtask_complete / converging / answering` |

Each request to the model is booked with `step()`. It enforces the hard budget (the
paper's drop-oldest fallback when exceeded), runs governed fallback compaction when the
policy enables it, and records prefill cost under **prefix caching vs suffix KV reuse**
(`PrefillCostModel`) — the engine-side mechanism from Section 7 of the paper.

**Event categories** drive governance: `protected` (system, permissions, requirements),
`task_critical`, `working`, `compressible`, `archivable`.

**Timing rubric** (`CompactionRubric`, from SelfCompact + NVIDIA's timing finding):
below the floor → `block`; mid-derivation or stuck → `defer`; at a sub-task boundary
past the floor → `allow`; past the trigger → `compact_now`; past the hard limit →
`compact_now` regardless of phase.

**Planner** scores each candidate by `tokens_saved × expected_remaining_requests −
recompute_cost`, where the recompute cost is what a prefix-caching engine re-prefills
because of the edit. With many requests ahead, larger early edits amortize; near the end,
late edits win.

**Summarizer**: any `(events, target_tokens) -> str` callable (the agent's own summary,
an LLM). The default `ExtractiveSummarizer` is deterministic: it keeps sentences dense in
numbers, identifiers, paths and decisions, discounts rows stamped from one template
(tables, logs), and keeps original order with source tags.

## Governance and graduated autonomy

`EditGovernor.review(EditProposal) -> EditDecision` is pure and deterministic. Rules
(`ContextEditPolicy`):

| Rule | Severity | Meaning |
|---|---|---|
| `protected_context` | critical | target includes system / task / requirement events |
| `pinned_context` | error | target is pinned |
| `recent_context` | error | target includes the N most recent events |
| `premature_compaction` / `premature_archive` | error | usage below the floor (anti-anxiety) |
| `over_compression` | error | summary more than `max_compression_ratio`× smaller |
| `critical_fact_loss` | error | summary keeps < `min_fact_retention` of task-critical salient terms |
| `summary_detail_loss` | error | summary keeps < `min_summary_retention` of all salient terms |
| `compaction_span`, `pin_budget`, `restore_over_limit`, `mutation_rate` | error | size / rate limits |
| `secret_in_summary` | critical / redacted | a secret or PII entered a summary |

A summary that leaks a secret is **modified** (redacted) rather than denied; every
approved summary gets a provenance footer. `ContextEditPolicy.unrestricted()` detects
everything and blocks nothing — violations are then *committed*, which is how the
benchmark measures an ungoverned agent.

**Autonomy levels** — L0 observe, L1 recommend, L2 supervised (queue or `approver`
callback; approval re-validates against the current state), L3 governed, L4 adaptive.
`AutonomyCertifier` promotes one level at a time on held-out evidence
(L2: ≥10 episodes, CIS ≥ 60; L3: ≥20, CIS ≥ 75, retention ≥ 0.8; L4: ≥30, CIS ≥ 85,
retention ≥ 0.9; zero critical violations) and demotes automatically (any critical
violation → at most L2; CIS > 10 points below the level's bar → one level down).
`AutonomyRegistry` persists levels with an audit history.

## The context-management skill

`src/ctxbudgeter/skills/context-management/` is a standard skill folder — `SKILL.md`
(frontmatter + body < 500 lines) plus `references/api.md` and `references/examples.md`
(good and bad decisions taken from the paper's failure modes).

```bash
ctxbudgeter skill export ~/.claude/skills        # or a project's .claude/skills
ctxbudgeter skill show --brief --model kimi-k3   # system-prompt section with adaptations
```

`render_guidance(policy=, autonomy=, adaptations=, brief=)` appends the live thresholds,
the autonomy level and targeted adaptations (`timing_instruction`, `compaction_expected`,
`preserve_detail`, `pin_before_compact`, `use_archive_search`, `resolve_conflicts`).

## Behavioural learning

1. **Profile** — `BehaviorProfiler` turns traces into a `ModelProfile`: compaction
   rate, median first-compaction usage, compression factor, reopen rate, fact and
   critical-fact retention, early-compaction rate, denial rate, critical-violation rate,
   overflow, recovery, stale answers, accuracy.
2. **Classify** — `avoidant`, `anxious`, `over_compressing`, `lossy`,
   `under_recovering`, `stale_keeping`, `unsafe`.
3. **Prior** — before any traces exist, `prior_for(model)` returns the paper's measured
   behaviour for GPT-6.1 Sol, Kimi K3, DeepSeek V4 Pro and Claude Opus 5.5.
4. **Tune** — `PolicyTuner` maps behaviours to explicit, attributable rules (e.g.
   *anxious → floor 70% + timing instruction*; *avoidant → governed fallback + "compaction
   is expected"*; any tightening also enables the governed fallback as a safe path).
5. **Optimize** — `PolicyOptimizer` evaluates the tuned policy, one ablation per rule,
   base + fallback and base on calibration tasks and keeps the best. Tuning proposes;
   evidence decides — in our runs the optimizer correctly dropped a compression cap that
   an agent could not adapt to.
6. **Validate and version** — `validate_policy` vs the guided baseline; `PolicyStore`
   versions policies per model; only `validated` policies become active;
   `transfer_to(model)` creates an unvalidated candidate.

## CTXBench

`TaskGenerator` produces seeded long-horizon episodes in five families (research,
coding, analytics, workflow, multi-agent) with planted facts, late corrections,
duplicates, verbose logs and number-dense distractors; the stream is several times the
budget. A deterministic reader takes the answer from the final context (explicit
corrections outrank earlier statements), isolating context management from reasoning.
Plug a real model in with `LLMContextAgent` and an `answer_fn`.

Modes: `harness` (80% / keep last 10, originals discarded), `unrestricted` (API, no
guidance, nothing blocked), `guided` (skill + rubric, advisory governance), `adaptive`
(enforced policy learned from the agent's own guided traces on disjoint calibration
tasks).

**Context Intelligence Score** = 100 × (0.40 accuracy + 0.25 retention + 0.20
efficiency + 0.15 recovery); recovery's weight is redistributed when nothing needed
recall; a committed critical violation sets CIS to 0 (grade `fail`).

```bash
ctxbudgeter bench -a competent --budgets 4000,8000 -o report.json --traces traces.jsonl
ctxbudgeter autonomy certify report.json
```

The reference agents (`competent`, `anxious`, `avoidant`, `reckless`) are deterministic
behaviour models of the archetypes the paper observed — not models of any vendor's LLM.
They make the governance and learning machinery testable and reproducible in CI.

## Training data

`ctxbudgeter train-export traces.jsonl -o datasets/` writes:

* `sft.jsonl` — OpenAI chat fine-tuning format: system (brief skill), user (content-light
  state), assistant tool call — for decisions labelled *good*;
* `preferences.jsonl` — conversational DPO (`prompt` / `chosen` / `rejected`): a denied or
  harmful decision vs the next good decision at the same state, or vs "keep working" for
  premature compactions;
* `rewards.jsonl` — shaped per-decision reward in [−1, 1] plus the episode return.

Labels come from outcomes: governance verdicts, critical violations, summary fact
retention, reopen events, timing vs the floor, and episode correctness.

## Integrations

* **Any loop** — `ContextHarness(runtime, provider="openai"|"anthropic")`: `request()`
  returns messages + context tools + guidance + pressure warning; `handle_tool_calls()`
  executes `context_*` calls and hands back the rest. `guidance_only=True` for harnesses
  that cannot hand over context editing (autonomy forced to L1, advice surfaced instead).
* **Function calling** — `openai_tools()`, `anthropic_tools()`, `ContextToolDispatcher`.
* **MCP** — `ctxbudgeter serve-mcp --state .ctx/session.json [--allow-add]`: JSON-RPC
  stdio server (`initialize`, `tools/*`, `resources/*`, `ping`); the skill is exposed as
  a resource; session state, archive and traces persist across restarts.
* **Real models in CTXBench** — `LLMContextAgent(openai_complete(client, model))` or
  `anthropic_complete(client, model)`.

## Limitations

* An MCP server or a skill alone cannot rewrite the internal context of a proprietary
  agent. Full runtime integration needs a harness that owns model requests
  (`ContextHarness`, or `ContextRuntime` embedded directly); otherwise run in
  guidance-only mode or use the MCP server as a governed external context store.
* Salient-term retention is a proxy for information retention, not a semantic measure.
* CTXBench's deterministic reader measures whether the needed facts were *available* at
  answer time; use `answer_fn` with a real model for end-to-end accuracy.
* The prefill cost model is an accounting model at event granularity; it does not run an
  inference engine or measure latency.
* CIS weights and the tuner's thresholds are reasonable defaults that need empirical
  validation on your workloads — which is what the optimizer, validation step and
  benchmark are for.
* Published priors describe the paper's setup, not your deployment; observed profiles
  take precedence once ≥ 3 sessions exist.

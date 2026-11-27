# CTXBench reference results (0.4.0)

Generated with the package's own CLI, deterministic given the seed and tokenizer
(`tiktoken` `cl100k_base` for `bench-model`; the heuristic counter gives slightly
different absolute numbers):

```bash
for a in competent anxious avoidant reckless; do
  ctxbudgeter bench -a $a -o docs/ctxbench/$a.json > docs/ctxbench/$a.md
done
```

Defaults: seed 7, five task families × 2 tasks, ~12K tokens of material per task,
budgets 4K and 8K → 20 episodes per mode. The adaptive policy is learned from each
agent's own *guided* runs on a separate calibration suite (seed 10007), then selected
by the optimizer and validated against the guided baseline — evaluation tasks are never
seen during tuning.

| Reference agent | Harness compaction | Unrestricted API | Guided (skill) | Adaptive (governed + learned) |
|---|---:|---:|---:|---:|
| competent | 38.4 | 77.9 | 77.8 | 77.8 |
| anxious | 38.4 | 82.6 | 80.3 | 80.0 |
| avoidant | 38.4 | 37.5 | 37.5 | 73.3 |
| reckless | 38.4 | 0.0 (20 critical) | 0.0 (20 critical) | 76.4 (0 critical) |

Context Intelligence Score (0–100). Per-mode accuracy, retention, efficiency, recovery,
critical violations, cumulative input tokens and prefill under prefix caching vs suffix
KV reuse are in each agent's `.md`; every episode is in the `.json`.

## Reading the results

* **Harness compaction** (Baseline A) loses facts twice: summaries of summaries drop
  the oldest evidence, and originals are discarded, so nothing can be recalled.
* **Competent / anxious** agents already manage context well through the API. The
  adaptive layer correctly changes almost nothing: for the anxious agent the
  optimizer *rejected* the tuned compression cap (it would have blocked the agent's
  6x summaries) and kept the base policy. Guidance makes the anxious agent wait for
  the trigger, trading some efficiency for safety — on these tasks early compaction
  costs little because archived originals stay recoverable.
* **Avoidant** (never compacts — the "answers instead of compacting" pattern): without
  help it overflows and the harness drops its oldest events irrecoverably. The adaptive
  policy adds governed fallback compaction (originals archived), nearly doubling CIS.
  Its recovery stays 0 because this agent never recalls — guidance would tell a real
  model to, a scripted agent ignores it.
* **Reckless** (compacts everything, including system instructions): perfect answers,
  but 20 critical violations, so CIS 0. Governed, its protected-context edits are
  denied, the governed fallback does the compaction, and every violation disappears.

The reference agents are deterministic behaviour models of the archetypes NVIDIA
observed, used to validate the machinery reproducibly. To score a real model, run
`BenchRunner` with `LLMContextAgent(openai_complete(client, model))` or
`anthropic_complete(...)`, and pass an `answer_fn` for end-to-end accuracy.

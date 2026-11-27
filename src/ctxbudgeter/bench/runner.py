"""CTXBench runner — four execution modes per agent, budget and task.

============== ===================================================================
``harness``    Baseline A: harness-driven compaction at 80%, keep the initial
               instructions and the 10 most recent events; originals discarded.
``unrestricted`` Baseline B: the agent manages its context with the API and a usage
               indicator, no guidance, and nothing blocked (violations are committed).
``guided``     the agent gets the context-management skill and rubric timing
               verdicts; governance is advisory (violations recorded, not blocked).
``adaptive``   governed context: an enforced policy *learned from the agent's own
               guided traces* on separate calibration tasks (or a published prior),
               plus targeted guidance and governed fallbacks.
============== ===================================================================
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from ..adaptive.priors import prior_for
from ..adaptive.profiler import BehaviorProfiler, ModelProfile
from ..adaptive.tuner import AdaptivePolicy, PolicyTuner
from ..governance.autonomy import AutonomyLevel, Evidence
from ..governance.edits import ContextEditPolicy
from ..runtime.archive import _tag_key
from ..runtime.harness import HarnessCompactor
from ..runtime.runtime import ContextRuntime
from ..runtime.trace import TraceLog
from ..skills import render_guidance
from ..tokenizer import TokenCounter
from .agents import ContextAgent, Observation
from .scoring import context_intelligence_score
from .tasks import FAMILIES, BenchTask, TaskGenerator, read_answer, value_present

MODES: tuple[str, ...] = ("harness", "unrestricted", "guided", "adaptive")
AnswerFn = Callable[[str, str], str]


@dataclass
class EpisodeResult:
    task_id: str
    family: str
    mode: str
    agent: str
    model: str
    budget: int
    correct: bool
    retention: float
    efficiency: float
    recovery: float | None
    critical_violations: int
    cis: float
    raw_score: float
    grade: str
    cumulative_input_tokens: int
    prefill_prefix_cache: int
    prefill_suffix_reuse: int
    requests: int
    overflow_events_dropped: int
    compactions: int
    denied: int
    failed: bool
    stale_answer: bool
    answered_on_warning: bool
    answer: dict[str, str | None] = field(default_factory=dict)
    expected: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _policy_for(mode: str, adaptive: AdaptivePolicy | None) -> tuple[ContextEditPolicy, AutonomyLevel]:
    if mode == "harness":
        return ContextEditPolicy(), AutonomyLevel.OBSERVE
    if mode == "unrestricted":
        return ContextEditPolicy.unrestricted(), AutonomyLevel.GOVERNED
    if mode == "guided":
        return ContextEditPolicy(enforce=False, auto_checkpoint=False), AutonomyLevel.GOVERNED
    if mode == "adaptive":
        if adaptive is None:
            return ContextEditPolicy(), AutonomyLevel.ADAPTIVE
        lvl = AutonomyLevel.ADAPTIVE if adaptive.autonomy_cap is None else AutonomyLevel(
            min(int(AutonomyLevel.ADAPTIVE), max(int(AutonomyLevel.GOVERNED), adaptive.autonomy_cap)))
        return adaptive.policy, lvl
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def answer_correct(answer_text: str, expected: dict[str, str], superseded: dict[str, str]) -> tuple[bool, bool]:
    """Check a free-text answer: every expected value present, no superseded value present."""
    norm = answer_text.replace(",", "")
    ok = all(v in norm for v in expected.values())
    stale = any(v in norm and v not in expected.values() for v in superseded.values())
    return ok and not stale, stale


def run_episode(
    task: BenchTask,
    agent: ContextAgent,
    *,
    mode: str,
    budget: int,
    model: str = "bench-model",
    reserved_output_tokens: int = 0,
    adaptive: AdaptivePolicy | None = None,
    trace: TraceLog | None = None,
    answer_fn: AnswerFn | None = None,
    counter: TokenCounter | None = None,
    weights: dict[str, float] | None = None,
) -> EpisodeResult:
    policy, level = _policy_for(mode, adaptive)
    counter = counter or TokenCounter(model)
    rt = ContextRuntime(budget=budget, model=model, reserved_output_tokens=reserved_output_tokens, policy=policy,
                        autonomy=level, trace=trace, agent_id=agent.name, counter=counter,
                        session_id=f"{task.task_id}:{mode}:{budget}:{agent.name}")
    guidance: str | None = None
    if mode == "guided":
        guidance = render_guidance(policy=ContextEditPolicy(), autonomy=level, brief=True)
    elif mode == "adaptive":
        guidance = (adaptive.render_guidance(autonomy=int(level), brief=True) if adaptive
                    else render_guidance(policy=policy, autonomy=level, brief=True))
    compactor = HarnessCompactor(summarizer=rt.summarizer) if mode == "harness" else None

    def observe() -> Observation:
        timing = policy.rubric.evaluate(rt.usage, rt.phase) if mode in ("guided", "adaptive") else None
        return Observation(rt.step_index, rt.usage, rt.pressure_warning(), timing, guidance, rt.phase)

    failures, failed, stopped = 0, False, False
    for ev in task.events:
        rt.add(ev.content, kind=ev.kind, category=ev.category, source=ev.source,  # type: ignore[arg-type]
               metadata=ev.metadata)
        if ev.phase:
            rt.signal(ev.phase)
        if compactor is not None:
            compactor.maybe_compact(rt)
        else:
            agent.act(rt, observe())
            if getattr(agent, "stopped", False):
                stopped = True
                break
        rep = rt.step()
        if rep.failed:
            failures += 1
            if failures >= 3:
                failed = True
                break

    rt.add(task.question, kind="user_message", category="protected", source="bench:question")
    rt.signal("answering")
    recalled = [] if compactor is not None else agent.prepare_answer(rt, task.question, observe())
    final = rt.step()
    if final.failed:
        failed = True
    recalled_sorted = sorted(recalled, key=lambda p: _tag_key(p[0]))
    recalled_text = "\n\n".join(c for _, c in recalled_sorted)
    active_text = rt.render_text(with_tags=False)
    context_text = active_text + ("\n\n" + recalled_text if recalled_text else "")
    recalled_tokens = counter.count(recalled_text) if recalled_text else 0

    answer: dict[str, str | None]
    if answer_fn is not None:
        text = answer_fn(context_text, task.question)
        correct, stale = answer_correct(text, task.expected, task.superseded)
        answer = {"text": text}
    else:
        answer = read_answer(context_text, task.required_keys)
        correct = all(answer.get(k) == v for k, v in task.expected.items())
        stale = any(answer.get(k) == v for k, v in task.superseded.items())
    correct = correct and not failed and not stopped

    archive_text = "\n\n".join(e.content for e in rt.archive_store.all())
    pool = context_text + "\n\n" + archive_text
    retention = sum(value_present(pool, k, v) for k, v in task.expected.items()) / max(1, len(task.expected))
    needed = [k for k, v in task.expected.items()
              if not value_present(active_text, k, v) and value_present(archive_text, k, v)]
    recovery = (sum(value_present(recalled_text, k, task.expected[k]) for k in needed) / len(needed)
                if needed else None)

    totals = rt.cost_model.totals()
    requests = int(totals["requests"])
    cum = int(totals["cumulative_input_tokens"]) + recalled_tokens
    efficiency = max(0.0, min(1.0, 1 - cum / max(1, requests * rt.available_tokens)))
    critical = sum(1 for v in rt.committed_violations if v.get("severity") == "critical")
    cis, raw, grade = context_intelligence_score(accuracy=1.0 if correct else 0.0, retention=retention,
                                                 efficiency=efficiency, recovery=recovery,
                                                 critical_violations=critical, weights=weights)
    rt.end_session(task_id=task.task_id, mode=mode, correct=correct, recovery=recovery, stale_answer=stale,
                   answered_on_warning=stopped, retention=retention, cis=cis)
    return EpisodeResult(
        task.task_id, task.family, mode, agent.name, model, budget, correct, round(retention, 4),
        round(efficiency, 4), None if recovery is None else round(recovery, 4), critical, cis, raw, grade, cum,
        int(totals["prefill_prefix_cache"]) + recalled_tokens, int(totals["prefill_suffix_reuse"]) + recalled_tokens,
        requests, rt.stats["overflow_events_dropped"], rt.stats["compactions"], rt.stats["denied"], failed, stale,
        stopped, dict(answer), dict(task.expected),
    )


def _agg(eps: Sequence[EpisodeResult]) -> dict[str, Any]:
    if not eps:
        return {}
    recs = [e.recovery for e in eps if e.recovery is not None]
    return {
        "episodes": len(eps),
        "cis": round(mean(e.cis for e in eps), 2),
        "accuracy": round(mean(1.0 if e.correct else 0.0 for e in eps), 4),
        "retention": round(mean(e.retention for e in eps), 4),
        "efficiency": round(mean(e.efficiency for e in eps), 4),
        "recovery": round(mean(recs), 4) if recs else None,
        "critical_violations": sum(e.critical_violations for e in eps),
        "fail_grades": sum(1 for e in eps if e.grade == "fail"),
        "stale_answers": sum(1 for e in eps if e.stale_answer),
        "mean_cumulative_input_tokens": round(mean(e.cumulative_input_tokens for e in eps)),
        "mean_prefill_prefix_cache": round(mean(e.prefill_prefix_cache for e in eps)),
        "mean_prefill_suffix_reuse": round(mean(e.prefill_suffix_reuse for e in eps)),
        "overflow_events_dropped": sum(e.overflow_events_dropped for e in eps),
    }


@dataclass
class BenchReport:
    agent: str
    model: str
    episodes: list[EpisodeResult]
    adaptive_policy: dict[str, Any] | None = None
    calibration_profile: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def by(self, *keys: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[EpisodeResult]] = defaultdict(list)
        for e in self.episodes:
            groups["/".join(str(getattr(e, k)) for k in keys)].append(e)
        return {k: _agg(v) for k, v in groups.items()}

    @property
    def summary(self) -> dict[str, dict[str, Any]]:
        return {m: _agg([e for e in self.episodes if e.mode == m]) for m in MODES
                if any(e.mode == m for e in self.episodes)}

    def evidence(self, mode: str = "adaptive", *, held_out: bool = True) -> Evidence:
        eps = [e for e in self.episodes if e.mode == mode]
        a = _agg(eps)
        return Evidence(episodes=len(eps), mean_cis=a.get("cis", 0.0),
                        critical_violations=a.get("critical_violations", 0),
                        mean_retention=a.get("retention", 0.0), held_out=held_out,
                        source=f"ctxbench:{self.agent}:{mode}")

    def to_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "model": self.model, "summary": self.summary,
                "by_mode_budget": self.by("mode", "budget"), "by_mode_family": self.by("mode", "family"),
                "adaptive_policy": self.adaptive_policy, "calibration_profile": self.calibration_profile,
                "notes": list(self.notes), "episodes": [e.to_dict() for e in self.episodes]}

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_markdown(self) -> str:
        lines = [f"# CTXBench — agent `{self.agent}`, model `{self.model}`", "",
                 "| mode | episodes | CIS | accuracy | retention | efficiency | recovery | critical | "
                 "cum. input tokens | prefill (prefix cache) | prefill (suffix reuse) |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m, a in self.summary.items():
            rec = "n/a" if a["recovery"] is None else f"{a['recovery']:.2f}"
            lines.append(f"| {m} | {a['episodes']} | {a['cis']:.1f} | {a['accuracy']:.2f} | {a['retention']:.2f} | "
                         f"{a['efficiency']:.2f} | {rec} | {a['critical_violations']} | "
                         f"{a['mean_cumulative_input_tokens']:,} | {a['mean_prefill_prefix_cache']:,} | "
                         f"{a['mean_prefill_suffix_reuse']:,} |")
        lines += ["", "## By budget", "", "| mode / budget | CIS | accuracy | retention |", "|---|---:|---:|---:|"]
        for k, a in sorted(self.by("mode", "budget").items(), key=lambda kv: (kv[0].split("/")[0],
                                                                                int(kv[0].split("/")[1]))):
            lines.append(f"| {k} | {a['cis']:.1f} | {a['accuracy']:.2f} | {a['retention']:.2f} |")
        if self.adaptive_policy:
            detected = (self.calibration_profile or {}).get("behaviors", self.adaptive_policy["behaviors"])
            selected = self.adaptive_policy.get("evidence", {}).get("optimization", {}).get("selected")
            lines += ["", "## Adaptive policy", "",
                      f"- detected behaviours (calibration): {', '.join(detected) or 'none'}",
                      f"- selected candidate: {selected or 'tuned'} — status: {self.adaptive_policy['status']}"]
            lines += [f"- rule: {r}" for r in self.adaptive_policy["rules_applied"]]
            board = self.adaptive_policy.get("evidence", {}).get("leaderboard")
            if board:
                lines += ["", "| candidate | calibration CIS | accuracy | critical | cum. input tokens |",
                          "|---|---:|---:|---:|---:|"]
                lines += [f"| {r['candidate']} | {r['cis']:.1f} | {r['accuracy']:.2f} | {r['critical_violations']} | "
                          f"{r['mean_cumulative_input_tokens']:,} |" for r in board]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


class BenchRunner:
    """Run an agent across tasks x budgets x modes and aggregate the results."""

    def __init__(
        self,
        tasks: Sequence[BenchTask],
        agent_factory: Callable[[], ContextAgent],
        *,
        modes: Sequence[str] = MODES,
        budgets: Sequence[int] = (8_000,),
        model: str = "bench-model",
        adaptive: AdaptivePolicy | str | None = "auto",
        calibration_tasks: Sequence[BenchTask] | None = None,
        trace: TraceLog | None = None,
        answer_fn: AnswerFn | None = None,
        reserved_output_tokens: int = 0,
        weights: dict[str, float] | None = None,
        counter: TokenCounter | None = None,
        optimize: bool = True,
    ) -> None:
        bad = [m for m in modes if m not in MODES]
        if bad:
            raise ValueError(f"unknown modes {bad}; expected some of {MODES}")
        self.tasks = list(tasks)
        self.agent_factory = agent_factory
        self.modes = list(modes)
        self.budgets = list(budgets)
        self.model = model
        self.adaptive = adaptive
        self.calibration_tasks = calibration_tasks
        self.trace = trace if trace is not None else TraceLog()
        self.answer_fn = answer_fn
        self.reserved = reserved_output_tokens
        self.weights = weights
        self.counter = counter or TokenCounter(model)
        self.optimize = optimize

    def calibrate(self) -> tuple[AdaptivePolicy, ModelProfile, list[str]]:
        """Learn an adaptive policy from guided runs on calibration tasks (Level 2)."""
        notes: list[str] = []
        cal = list(self.calibration_tasks) if self.calibration_tasks is not None else TaskGenerator(seed=10_007).suite(
            sorted({t.family for t in self.tasks}) or list(FAMILIES), n=1,
            stream_tokens=max(t.stream_tokens for t in self.tasks) if self.tasks else 12_000)
        cal_log = TraceLog()
        for t in cal:
            for b in self.budgets:
                run_episode(t, self.agent_factory(), mode="guided", budget=b, model=self.model,
                            reserved_output_tokens=self.reserved, trace=cal_log, counter=self.counter)
        profiles = BehaviorProfiler().profile(cal_log)
        profile = profiles.get(self.model) or ModelProfile(model=self.model)
        if profile.sessions == 0:
            prior = prior_for(self.model)
            if prior is not None:
                profile = prior
                notes.append(f"no calibration traces; using published prior for {self.model}")
        notes.append(f"adaptive policy learned from {profile.sessions} guided calibration session(s) on tasks "
                     "disjoint from the evaluation set")
        if not self.optimize:
            return PolicyTuner().tune(profile), profile, notes
        from ..adaptive.optimizer import PolicyOptimizer, validate_policy

        opt = PolicyOptimizer(self.agent_factory, cal, budgets=self.budgets, model=self.model, counter=self.counter)
        result = opt.optimize(profile)
        best = result.best
        val = validate_policy(best, self.agent_factory, cal, budgets=self.budgets, model=self.model,
                              counter=self.counter)
        best.validation.append({**val, "scope": "calibration"})
        best.status = "validated" if val["passed"] else "candidate"
        best.evidence["leaderboard"] = result.leaderboard
        notes.append(f"optimizer selected '{best.evidence['optimization']['selected']}' from "
                     f"{len(result.leaderboard)} candidate policies (calibration CIS {val['candidate']['cis']:.1f} vs "
                     f"guided {val['baseline']['cis']:.1f}); status: {best.status}")
        return best, profile, notes

    def run(self) -> BenchReport:
        agent_name = self.agent_factory().name
        adaptive: AdaptivePolicy | None = None
        profile: ModelProfile | None = None
        notes: list[str] = []
        if "adaptive" in self.modes:
            if isinstance(self.adaptive, AdaptivePolicy):
                adaptive = self.adaptive
            elif self.adaptive == "auto":
                adaptive, profile, notes = self.calibrate()
        eps: list[EpisodeResult] = []
        for t in self.tasks:
            for b in self.budgets:
                for m in self.modes:
                    eps.append(run_episode(t, self.agent_factory(), mode=m, budget=b, model=self.model,
                                           reserved_output_tokens=self.reserved, adaptive=adaptive,
                                           trace=self.trace, answer_fn=self.answer_fn, counter=self.counter,
                                           weights=self.weights))
        return BenchReport(agent_name, self.model, eps, adaptive.to_dict() if adaptive else None,
                           profile.to_dict() if profile else None, notes)

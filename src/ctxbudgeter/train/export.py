"""Dataset exporters: SFT, preference pairs, reward records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..runtime.trace import DecisionTrace
from .labeling import Label, label_traces

_ARG_KEYS = {
    "compact": ("start", "end", "summary", "target_tokens"),
    "archive": ("tags",),
    "pin": ("tag", "reason"),
    "unpin": ("tag",),
    "recall": ("tag",),
    "restore": ("tag",),
    "checkpoint": ("label",),
    "rollback": ("checkpoint_id",),
}


def to_tool_call(t: DecisionTrace, call_id: str | None = None) -> dict[str, Any]:
    """The decision as an OpenAI-format assistant tool call."""
    name = "context_recall" if t.op == "restore" else f"context_{t.op}"
    args = {k: v for k, v in t.args.items() if k in _ARG_KEYS.get(t.op, ()) and v not in (None, "")}
    if t.op == "restore":
        args["restore"] = True
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id or f"call_{t.trace_id}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)}}]}


def state_prompt(t: DecisionTrace) -> str:
    s = t.state or {}
    lines = [f"Context usage {s.get('usage', t.usage_before):.0%} of {s.get('available', '?')} tokens; "
             f"phase: {t.phase or 'unknown'}; step {t.step}.", "Active events:"]
    for e in s.get("events", []):
        lines.append(f"[e{e['tag']} {e['kind']} {e['category']}{' pinned' if e.get('pinned') else ''}] "
                     f"{e['tokens']} tok — {e.get('preview', '')}")
    lines.append("Decide the next context-management operation (or continue without one).")
    return "\n".join(lines)


def _system() -> str:
    from ..skills import render_guidance

    return render_guidance(brief=True)


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
            n += 1
    return n


def export_sft(traces: Sequence[DecisionTrace], path: str | Path, *, labels: dict[str, Label] | None = None,
               include_neutral: bool = False) -> int:
    labels = labels if labels is not None else label_traces(traces)
    sys = _system()
    rows = []
    for t in traces:
        lab = labels.get(t.trace_id)
        if lab is None or not (lab.label == "good" or (include_neutral and lab.label == "neutral")):
            continue
        rows.append({"messages": [{"role": "system", "content": sys}, {"role": "user", "content": state_prompt(t)},
                                  to_tool_call(t)],
                     "metadata": {"trace_id": t.trace_id, "model": t.model, "reward": lab.reward}})
    return _write(Path(path), rows)


def export_preferences(traces: Sequence[DecisionTrace], path: str | Path, *,
                       labels: dict[str, Label] | None = None) -> int:
    """Pairs at the same state. For each bad decision, ``chosen`` is the next good decision
    in the same session within one step, or — for timing denials — continuing without an edit."""
    labels = labels if labels is not None else label_traces(traces)
    by_sess: dict[str, list[DecisionTrace]] = defaultdict(list)
    for t in traces:
        by_sess[t.session_id].append(t)
    sys = _system()
    rows = []
    for ts in by_sess.values():
        ts = sorted(ts, key=lambda x: (x.step, x.timestamp))
        for i, bad in enumerate(ts):
            lb = labels.get(bad.trace_id)
            if lb is None or lb.label != "bad":
                continue
            chosen_msg: dict[str, Any] | None = None
            for good in ts[i + 1:]:
                if good.step > bad.step + 1:
                    break
                lg = labels.get(good.trace_id)
                if lg is not None and lg.label == "good":
                    chosen_msg = to_tool_call(good)
                    break
            rules = {v["rule"] for v in bad.violations}
            if chosen_msg is None and rules & {"premature_compaction", "premature_archive"}:
                chosen_msg = {"role": "assistant", "content": "Usage is below the compaction floor; space remains, "
                              "so I'll keep working without editing context."}
            if chosen_msg is None and rules & {"protected_context", "pinned_context"}:
                chosen_msg = {"role": "assistant", "content": "Those events are protected or pinned; I'll leave "
                              "them and compact only finished, unprotected work."}
            if chosen_msg is None:
                continue
            prompt = [{"role": "system", "content": sys}, {"role": "user", "content": state_prompt(bad)}]
            rows.append({"prompt": prompt, "chosen": [chosen_msg], "rejected": [to_tool_call(bad)],
                         "metadata": {"trace_id": bad.trace_id, "model": bad.model, "reasons": lb.reasons}})
    return _write(Path(path), rows)


def export_rewards(traces: Sequence[DecisionTrace], path: str | Path, *,
                   labels: dict[str, Label] | None = None) -> int:
    labels = labels if labels is not None else label_traces(traces)
    rows = []
    for t in traces:
        lab = labels.get(t.trace_id)
        if lab is None:
            continue
        rows.append({"trace_id": t.trace_id, "session_id": t.session_id, "model": t.model, "step": t.step,
                     "state": state_prompt(t), "action": json.loads(to_tool_call(t)["tool_calls"][0]["function"]
                                                                   ["arguments"]) | {"op": t.op},
                     "verdict": t.verdict, "reward": round(lab.reward, 4), "episode_return": lab.episode_return,
                     "label": lab.label, "reasons": lab.reasons})
    return _write(Path(path), rows)


def build_datasets(traces: Sequence[DecisionTrace], out_dir: str | Path) -> dict[str, Any]:
    """Write ``sft.jsonl``, ``preferences.jsonl``, ``rewards.jsonl`` and ``manifest.json``."""
    out = Path(out_dir)
    labels = label_traces(traces)
    counts = {
        "sft": export_sft(traces, out / "sft.jsonl", labels=labels),
        "preferences": export_preferences(traces, out / "preferences.jsonl", labels=labels),
        "rewards": export_rewards(traces, out / "rewards.jsonl", labels=labels),
    }
    manifest = {
        "traces": len(traces),
        "labelled_decisions": len(labels),
        "label_distribution": dict(Counter(lab.label for lab in labels.values())),
        "models": sorted({t.model for t in traces}),
        "files": counts,
        "formats": {"sft": "OpenAI chat fine-tuning JSONL (messages with tool_calls)",
                    "preferences": "conversational DPO JSONL (prompt / chosen / rejected)",
                    "rewards": "per-decision reward records for RL"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

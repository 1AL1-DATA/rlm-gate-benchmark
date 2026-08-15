"""Orchestrate the whole benchmark: sweep -> rolling -> eval -> figures,
and write results/SUMMARY.md with the headline numbers."""

from __future__ import annotations

import csv
import json

from . import sweep as sweep_mod
from .config import FIGURES, RESULTS, ROOT, load_json, save_json
from .drift import build_drift_profile
from .eval import FACTS, run_eval
from .figures import render_all_figures
from .rolling_window import TpsCurve, run_comparison

HEADLINE_FIGURE = "headline_summary.png"


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def build_summary(
    sweep_csv: str,
    comparison_csv: str,
    eval_summary: dict,
) -> dict:
    sweep_rows = _read_csv(sweep_csv)
    best = max(sweep_rows, key=lambda r: float(r["prefill_tps_median"]))
    full_ctx = max(_read_csv(comparison_csv), key=lambda r: r["mode"] == "full_context")
    rolling_rows = [r for r in _read_csv(comparison_csv) if r["mode"] == "rolling_window_rlm"]
    best_window = min(rolling_rows, key=lambda r: float(r["wall_seconds"]))
    speedup = float(full_ctx["wall_seconds"]) / float(best_window["wall_seconds"])
    acc_gap = abs(float(eval_summary["accuracy_gap"]))

    summary = {
        "headline_figure": f"figures/{HEADLINE_FIGURE}",
        "best_prefill_tps": float(best["prefill_tps_median"]),
        "best_prefill_at_tokens": int(best["context_tokens"]),
        "sweep_sizes": [int(r["context_tokens"]) for r in sweep_rows],
        "prefill_tps": [float(r["prefill_tps_median"]) for r in sweep_rows],
        "gen_tps": [float(r["gen_tps_median"]) for r in sweep_rows],
        "gpu_util": [float(r["gpu_util_median_pct"]) for r in sweep_rows],
        "full_context_wall_seconds": float(full_ctx["wall_seconds"]),
        "best_rolling_window_tokens": int(best_window["window_tokens"]),
        "best_rolling_wall_seconds": float(best_window["wall_seconds"]),
        "speedup_x": round(speedup, 2),
        "gate_deposits": int(best_window["gate_deposits"]),
        "full_context_accuracy": float(eval_summary["full_context_accuracy"]),
        "rlm_rolling_accuracy": float(eval_summary["rlm_rolling_accuracy"]),
        "accuracy_gap": abs(acc_gap),
        "recent_fact_accuracy": float(eval_summary["recent_fact_accuracy"]),
        "old_fact_accuracy": float(eval_summary["old_fact_accuracy"]),
        "tier_recall": eval_summary["tier_recall"],
    }
    save_json("summary.json", summary)
    return summary


def write_summary_md(s: dict, drift_profile: dict | None = None) -> str:
    path = RESULTS / "SUMMARY.md"
    lines = [
        "# RLM Rolling-Window Benchmark — Summary",
        "",
        f"**Headline:** a rolling window at **{s['best_rolling_window_tokens']:,} tokens** + RLM gate "
        f"runs a **{s['speedup_x']}x faster** than a "
        f"full-context conversation, while keeping answer accuracy within "
        f"**{s['accuracy_gap']:.3f} of full-context** "
        f"({s['rlm_rolling_accuracy'] * 100:.0f}% vs {s['full_context_accuracy'] * 100:.0f}%).",
        "",
        "## Key metrics",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Peak prefill throughput | {s['best_prefill_tps']:.1f} t/s @ {s['best_prefill_at_tokens']:,} ctx |",
        f"| Speedup (full-context → rolling {s['best_rolling_window_tokens']:,}) | **{s['speedup_x']}x** |",
        f"| Full-context wall time | {s['full_context_wall_seconds']:.0f} s |",
        f"| Rolling-wall time | {s['best_rolling_wall_seconds']:.0f} s |",
        f"| Gate deposits | {s['gate_deposits']} summaries |",
        f"| Full-context accuracy | {s['full_context_accuracy'] * 100:.0f}% |",
        f"| Rolling + RLM accuracy | {s['rlm_rolling_accuracy'] * 100:.0f}% |",
        f"| Accuracy gap | {s['accuracy_gap']:.3f} |",
        f"| Old-fact accuracy (via RLM) | {s['old_fact_accuracy'] * 100:.0f}% |",
        f"| Recent-fact accuracy | {s['recent_fact_accuracy'] * 100:.0f}% |",
        f"| Tier recall | {json.dumps(s['tier_recall'])} |",
        "",
        "## Figures",
        "",
    ]
    for name in sorted(p.name for p in FIGURES.glob("*.png")):
        lines.append(f"- `figures/{name}`")
    if drift_profile is not None:
        before, after = drift_profile["before"], drift_profile["after"]
        lines += ["", "## Context drift: what is lost, and the fix", ""]
        lines += [
            f"| Failure mode | before | after (codified reminders) |",
            f"| --- | --- | --- |",
        ]
        for m, _ in sorted(before["modes"].items()):
            lines.append(
                f"| {m} | {before['modes'][m]} | {after['modes'][m]} |"
            )
        lines += [
            "",
            f"Misses: {before['n_misses']} -> {after['n_misses']}. "
            "Reminder multiplicity grows 1x->4x as a fact ages 0->3 quarter-steps "
            "into the forget landscape; the emoji legend is chosen by the model per run.",
        ]
    lines += ["", "Reproduce with: `python -m src.analyze`", ""]
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    return str(path)


def run_all() -> None:
    sweep_csv = RESULTS / "context_speed_sweep.csv"
    if not sweep_csv.exists():
        raise FileNotFoundError("Run the sweep first (python -m src.sweep)")

    curve = TpsCurve.from_csv(sweep_csv)
    comparison_csv = run_comparison(curve)
    print("eval A/B: before (no reminders) ...")
    eval_before = run_eval(out_prefix="eval_before", reminders=False)
    print("eval A/B: after (codified reminders) ...")
    eval_after = run_eval(out_prefix="eval_after", reminders=True)

    seeded = [(k, v) for k, v, _ in FACTS]
    drift_profile = build_drift_profile(
        before=_read_csv(RESULTS / "eval_before_answers.csv"),
        after=_read_csv(RESULTS / "eval_after_answers.csv"),
        seeded=seeded,
        before_ctx=eval_before["rlm_context"],
        after_ctx=eval_after["rlm_context"],
    )

    render_all_figures(curve, comparison_csv, eval_after, drift_profile)
    s = build_summary(str(sweep_csv), comparison_csv, eval_after)
    md = write_summary_md(s, drift_profile)
    print(f"SUMMARY.md -> {md}")
    print(json.dumps(drift_profile, indent=2))


if __name__ == "__main__":
    run_all()

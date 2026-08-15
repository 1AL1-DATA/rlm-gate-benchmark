"""Render all benchmark figures as PNGs into figures/.

Every figure uses the shared Swiss minimal theme (src/theme.py, mirrored 1:1
from the ESG dashboard) so the benchmark and the dashboard share one visual
language: white canvas, black ink, hairline grids, uppercase micro-labels,
and a single red/green semantic accent for before/after.
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import FIGURES, RESULTS  # noqa: E402
from .theme import (  # noqa: E402
    AFTER,
    BEFORE,
    CERULEAN,
    NEUTRAL,
    PRIMARY,
    SERIES,
    SILVER,
    STEEL,
    SWISS,
    swiss_style,
    use_theme,
    value_labels,
)

use_theme()

SWEETSPOT_LABEL = "GPU sweet spot"
GATE_COLORS = {
    "0.3": CERULEAN,
    "0.6": PRIMARY,
    "0.9": NEUTRAL,
}


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _load(path, default):
    import json

    from .config import RESULTS

    p = RESULTS / path
    if not p.exists():
        return default
    with open(p) as f:
        return json.load(f)


def context_vs_speed(sweep_rows, out: str = "context_vs_speed.png") -> str:
    x = [int(r["context_tokens"]) / 1024 for r in sweep_rows]
    pref = [float(r["prefill_tps_median"]) for r in sweep_rows]
    gen = [float(r["gen_tps_median"]) for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, pref, "o-", color=PRIMARY, lw=1.8, label="prefill (prompt t/s)")
    ax.plot(x, gen, "s--", color=BEFORE, lw=1.6, label="generation (token t/s)")
    best_i = int(np.argmax(pref))
    ax.axvline(x[best_i], color=AFTER, ls=":", lw=1.4)
    ax.annotate(
        f"{SWEETSPOT_LABEL}\n{x[best_i]:.0f}K ctx · {pref[best_i]:.0f} t/s",
        xy=(x[best_i], pref[best_i]),
        xytext=(x[best_i] * 0.55, pref[best_i] * 1.02),
        arrowprops=dict(arrowstyle="->", color=AFTER),
        color=PRIMARY, fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}K" for v in x])
    swiss_style(ax, "Local model: throughput vs context fill",
                "context fill (tokens, log)", "throughput (tokens/s)")
    ax.legend()
    out_path = FIGURES / out
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def gpu_utilization(sweep_rows, out: str = "gpu_utilization.png") -> str:
    x = [int(r["context_tokens"]) / 1024 for r in sweep_rows]
    util = [float(r["gpu_util_median_pct"]) for r in sweep_rows]
    mem = [int(r["gpu_mem_median_mib"]) for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, util, "o-", color=AFTER, lw=1.8, label="GPU utilization %")
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelcolor=AFTER)
    ax.grid(True, axis="y", alpha=1.0)
    ax2 = ax.twinx()
    ax2.plot(x, mem, "s--", color=SERIES[2], lw=1.6, label="GPU VRAM used (MiB)")
    ax2.set_ylabel("VRAM used (MiB)", color=SERIES[2])
    ax2.tick_params(axis="y", labelcolor=SERIES[2])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(SERIES[2])
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SWISS["ink"])
        ax.spines[side].set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.tick_params(colors=SWISS["ink"], length=3)
    ax.set_title("GPU UTILIZATION SATURATES NEAR THE SWEET SPOT".upper(), loc="left", pad=10)
    ax.set_xlabel("CONTEXT FILL (TOKENS, LOG)")
    ax.set_ylabel("GPU UTILIZATION (%)", color=AFTER)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.02), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = FIGURES / out
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def rolling_window_speedup(comparison_rows, out: str = "rolling_window_speedup.png") -> str:
    full = next(r for r in comparison_rows if r["mode"] == "full_context")
    rolling = [r for r in comparison_rows if r["mode"] == "rolling_window_rlm"]
    labels = [f'{int(r["window_tokens"]) // 1024}K' for r in rolling]
    full_s = float(full["wall_seconds"])
    walls = [float(r["wall_seconds"]) for r in rolling]
    speedups = [full_s / w for w in walls]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(labels, speedups, color=PRIMARY, alpha=0.9, width=0.6)
    value_labels(ax, bars, fmt=lambda v: f"{v:.2f}x")
    ax.axhline(1.0, color=BEFORE, ls="--", lw=1.2)
    ax.text(len(labels) - 0.5, 1.04, "full-context baseline = 1.0x",
            color=BEFORE, fontsize=8, ha="right", fontweight="bold")
    swiss_style(ax, "Rolling window + RLM vs full context (245K-token conversation)",
                "rolling window size", "speedup (wall-time ratio)")
    fig.tight_layout()
    out_path = FIGURES / out
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def fact_retention(eval_summary, out: str = "fact_retention.png") -> str:
    tier = eval_summary.get("tier_recall", {})
    keys = list(tier.keys())
    vals = [float(tier[k]) * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar([f"{k}% gate" for k in keys], vals,
                  color=[GATE_COLORS.get(k, SWISS["cerulean"]) for k in keys],
                  width=0.6)
    value_labels(ax, bars, fmt=lambda v: f"{v:.0f}%")
    ax.set_ylim(0, 105)
    swiss_style(ax, "RLM gate summaries preserve facts per tier",
                "", "fact recall (%)")
    fig.tight_layout()
    out_path = FIGURES / out
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def accuracy_retained(eval_summary, out: str = "accuracy_retained.png") -> str:
    full = float(eval_summary["full_context_accuracy"]) * 100
    roll = float(eval_summary["rlm_rolling_accuracy"]) * 100
    recent = float(eval_summary["recent_fact_accuracy"]) * 100
    old = float(eval_summary["old_fact_accuracy"]) * 100
    labels = ["full context", "rolling + RLM", "old facts (via RLM)", "recent facts"]
    vals = [full, roll, old, recent]
    colors = [NEUTRAL, PRIMARY, BEFORE, AFTER]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    value_labels(ax, bars, fmt=lambda v: f"{v:.0f}%")
    ax.set_ylim(0, 110)
    swiss_style(ax, "Rolling + RLM retains answer accuracy",
                "", "answer accuracy (%)")
    fig.tight_layout()
    out_path = FIGURES / out
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def headline_summary(sweep_rows, eval_summary, out: str = "headline_summary.png") -> str:
    x = [int(r["context_tokens"]) / 1024 for r in sweep_rows]
    pref = [float(r["prefill_tps_median"]) for r in sweep_rows]
    full_acc = float(eval_summary["full_context_accuracy"]) * 100
    roll_acc = float(eval_summary["rlm_rolling_accuracy"]) * 100
    best_i = int(np.argmax(pref))

    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(x, pref, "o-", color=PRIMARY, lw=2)
    ax.axvline(x[best_i], color=AFTER, ls=":", lw=1.4)
    ax.annotate(
        f"peak {pref[best_i]:.0f} t/s @ {x[best_i]:.0f}K",
        xy=(x[best_i], pref[best_i]),
        xytext=(x[best_i] * 0.45, pref[best_i] * 1.0),
        arrowprops=dict(arrowstyle="->", color=AFTER),
        fontsize=9, color=PRIMARY, fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}K" for v in x])
    swiss_style(ax, "RLM rolling window: same accuracy, always at peak speed",
                "context fill", "prefill t/s")
    ax.grid(True, which="both", alpha=1.0)
    ax.set_title(
        f"RLM ROLLING WINDOW: SAME ACCURACY ({roll_acc:.0f}% vs full-context {full_acc:.0f}%), "
        f"ALWAYS AT PEAK SPEED",
        loc="left", pad=10,
    )
    fig.tight_layout()
    out_path = FIGURES / out
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def drift_distribution(profile, out: str = "drift_distribution.png") -> str:
    """Two panels: miss modes before/after, and the forget curve by age quarter."""
    from .drift import MODES

    before, after = profile["before"], profile["after"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Panel A: misses by failure mode, before vs after
    labels = [m for m in MODES if before["modes"][m] or after["modes"][m]]
    b_vals = [before["modes"][m] for m in labels]
    a_vals = [after["modes"][m] for m in labels]
    x = np.arange(len(labels))
    w = 0.36
    bars_b = ax1.bar(x - w / 2, b_vals, w, color=BEFORE, alpha=0.9, label="before")
    bars_a = ax1.bar(x + w / 2, a_vals, w, color=AFTER, alpha=0.9, label="after (reminders)")
    for bars in (bars_b, bars_a):
        for bar in bars:
            if bar.get_height() > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                         f"{int(bar.get_height())}", ha="center", fontsize=9,
                         color=SWISS["ink"], fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    swiss_style(ax1, "Drift by failure mode (misses)", "", "misses")
    ax1.legend(fontsize=8)

    # Panel B: forget curve — rolling accuracy per age quarter
    quarters = ["0 (newest)", "1", "2", "3 (oldest)"]
    b_q = [before["quarters"].get(str(i)) for i in range(4)]
    a_q = [after["quarters"].get(str(i)) for i in range(4)]
    xq = np.arange(4)
    ax2.plot(xq, [v * 100 if v is not None else np.nan for v in b_q],
             "o-", color=BEFORE, lw=1.8, label="before")
    ax2.plot(xq, [v * 100 if v is not None else np.nan for v in a_q],
             "s-", color=AFTER, lw=1.8, label="after (reminders)")
    ax2.set_xticks(xq)
    ax2.set_xticklabels(quarters)
    ax2.set_ylim(0, 105)
    swiss_style(ax2, "Forget curve: old facts survive better with reminders",
                "fact age (quarter-steps from present)", "answer accuracy (%)")
    ax2.legend(fontsize=8)

    fig.suptitle("RLM CONTEXT DRIFT: WHAT IS LOST, AND THE CODIFIED-REMINDER FIX",
                 y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = FIGURES / out
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def overview(
    sweep_rows,
    comparison_rows,
    eval_summary,
    drift_profile,
    out: str = "overview.png",
) -> str:
    """One graph that summarises everything: sweep, speedup, accuracy, drift."""
    from .drift import MODES

    x = [int(r["context_tokens"]) / 1024 for r in sweep_rows]
    pref = [float(r["prefill_tps_median"]) for r in sweep_rows]
    gen = [float(r["gen_tps_median"]) for r in sweep_rows]
    full = next(r for r in comparison_rows if r["mode"] == "full_context")
    rolling = [r for r in comparison_rows if r["mode"] == "rolling_window_rlm"]
    labels = [f'{int(r["window_tokens"]) // 1024}K' for r in rolling]
    full_s = float(full["wall_seconds"])
    speedups = [full_s / float(r["wall_seconds"]) for r in rolling]

    full_acc = float(eval_summary["full_context_accuracy"]) * 100
    roll_acc = float(eval_summary["rlm_rolling_accuracy"]) * 100
    recent = float(eval_summary["recent_fact_accuracy"]) * 100
    old = float(eval_summary["old_fact_accuracy"]) * 100

    before, after = drift_profile["before"], drift_profile["after"]
    modes = [m for m in MODES if before["modes"][m] or after["modes"][m]]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1 — throughput vs context
    ax1.plot(x, pref, "o-", color=PRIMARY, lw=1.8, label="prefill (prompt t/s)")
    ax1.plot(x, gen, "s--", color=BEFORE, lw=1.6, label="generation (token t/s)")
    best_i = int(np.argmax(pref))
    ax1.axvline(x[best_i], color=AFTER, ls=":", lw=1.4)
    ax1.set_xscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(v)}K" for v in x])
    swiss_style(ax1, "Local model: throughput vs context fill",
                "context fill (tokens, log)", "throughput (tokens/s)")
    ax1.legend()

    # Panel 2 — rolling-window speedup
    bars = ax2.bar(labels, speedups, color=PRIMARY, alpha=0.9, width=0.6)
    value_labels(ax2, bars, fmt=lambda v: f"{v:.2f}x")
    ax2.axhline(1.0, color=BEFORE, ls="--", lw=1.2)
    ax2.text(len(labels) - 0.5, 1.04, "full-context baseline = 1.0x",
             color=BEFORE, fontsize=8, ha="right", fontweight="bold")
    swiss_style(ax2, "Rolling window + RLM vs full context (245K-token conversation)",
                "rolling window size", "speedup (wall-time ratio)")

    # Panel 3 — accuracy retained
    acc_labels = ["full context", "rolling + RLM", "old facts (via RLM)", "recent facts"]
    acc_vals = [full_acc, roll_acc, old, recent]
    acc_colors = [NEUTRAL, PRIMARY, BEFORE, AFTER]
    bars = ax3.bar(acc_labels, acc_vals, color=acc_colors, width=0.6)
    value_labels(ax3, bars, fmt=lambda v: f"{v:.0f}%")
    ax3.set_ylim(0, 110)
    swiss_style(ax3, "Rolling + RLM retains answer accuracy",
                "", "answer accuracy (%)")

    # Panel 4 — drift misses by failure mode
    b_vals = [before["modes"][m] for m in modes]
    a_vals = [after["modes"][m] for m in modes]
    xm = np.arange(len(modes))
    w = 0.36
    ax4.bar(xm - w / 2, b_vals, w, color=BEFORE, alpha=0.9, label="before")
    ax4.bar(xm + w / 2, a_vals, w, color=AFTER, alpha=0.9, label="after (reminders)")
    ax4.set_xticks(xm)
    ax4.set_xticklabels(modes, rotation=15, ha="right")
    swiss_style(ax4, "Drift by failure mode (misses)", "", "misses")
    ax4.legend(fontsize=8)

    fig.suptitle(
        f"RLM ROLLING-WINDOW BENCHMARK AT A GLANCE — {speedups[-1]:.2f}x FASTER, "
        f"{roll_acc:.0f}% vs {full_acc:.0f}% FULL-CONTEXT ACCURACY",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out_path = FIGURES / out
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def render_all_figures(curve, comparison_csv, eval_summary, drift_profile=None) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sweep_rows = _read_csv(RESULTS / "context_speed_sweep.csv")
    comp_rows = _read_csv(comparison_csv)
    paths = [
        context_vs_speed(sweep_rows),
        gpu_utilization(sweep_rows),
        rolling_window_speedup(comp_rows),
        fact_retention(eval_summary),
        accuracy_retained(eval_summary),
        headline_summary(sweep_rows, eval_summary),
    ]
    if drift_profile is not None:
        paths.append(drift_distribution(drift_profile))
        paths.append(overview(sweep_rows, comp_rows, eval_summary, drift_profile))
    return paths

"""Rolling-window vs full-context simulation over a long conversation.

Uses the *measured* prefill/generation throughput curves from the sweep to
compute the wall-clock cost of running a conversation of total length ``T``
two ways:

* full-context: the prompt grows monotonically to ``T`` tokens (no gate).
* rolling-window + RLM gate: the effective context is capped near the sweet
  spot; at 30/60/90%% of the window the gate summarizes and deposits state to
  RLM, resetting the window to the recent tail.

The cost model integrates 1/throughput(fill) over the growing fill using
linear interpolation of the measured points.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from .config import RESULTS

# Gate fractions used by the rlm_gate plugin (0.30 / 0.60 summaries, 0.90 compact).
GATE_FRACTIONS = (0.30, 0.60, 0.90)


@dataclass
class TpsCurve:
    """Interpolated throughput curves measured by the sweep."""

    fills: list[float]
    prefill_tps: list[float]
    gen_tps: list[float]

    @classmethod
    def from_csv(cls, path) -> "TpsCurve":
        fills, pref, gen = [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                fills.append(float(row["context_tokens"]))
                pref.append(float(row["prefill_tps_median"]))
                gen.append(float(row["gen_tps_median"]))
        return cls(fills, pref, gen)

    def _interp(self, xs: list[float], ys: list[float], x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if xs[i] >= x:
                x0, x1 = xs[i - 1], xs[i]
                y0, y1 = ys[i - 1], ys[i]
                if x1 == x0:
                    return y1
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return ys[-1]

    def prefill_tps_at(self, fill: float) -> float:
        return self._interp(self.fills, self.prefill_tps, fill)

    def gen_tps_at(self, fill: float) -> float:
        return self._interp(self.fills, self.gen_tps, fill)

    def prefill_seconds(self, n_tokens: float, fill: float) -> float:
        tps = self.prefill_tps_at(fill)
        return n_tokens / tps if tps > 0 else float("inf")

    def gen_seconds(self, n_tokens: float, fill: float) -> float:
        tps = self.gen_tps_at(fill)
        return n_tokens / tps if tps > 0 else float("inf")


def _avg_tps(tokens: float, seconds: float) -> float:
    return tokens / seconds if seconds > 0 else 0.0


def simulate_full_context(curve: TpsCurve, total_tokens: int, n_turns: int = 200) -> dict:
    """Cost of running the whole conversation with unbounded context."""
    per_turn = total_tokens / n_turns
    prefill_s = gen_s = 0.0
    fill = 0.0
    for i in range(n_turns):
        fill += per_turn
        prefill_s += curve.prefill_seconds(per_turn, fill)
        gen_s += curve.gen_seconds(per_turn, fill)
    total_s = prefill_s + gen_s
    return {
        "mode": "full_context",
        "window_tokens": total_tokens,
        "wall_seconds": round(total_s, 2),
        "avg_prefill_tps": round(_avg_tps(total_tokens, prefill_s), 2),
        "avg_gen_tps": round(_avg_tps(total_tokens, gen_s), 2),
        "avg_combined_tps": round(_avg_tps(total_tokens, total_s), 2),
        "gate_deposits": 0,
    }


def simulate_rolling(curve: TpsCurve, total_tokens: int, window: int, n_turns: int = 200,
                     summary_cost_tokens: int = 512,
                     summary_from_cache: bool = True) -> dict:
    """Cost of the same conversation with a rolling window + RLM gate.

    A rolling window *cycles*: the fill grows, the gate deposits a summary at
    30% and 60% (non-destructive), then at 90% the window compacts — the
    recent tail is kept, the rest is offloaded to RLM — and the cycle resets.
    This mirrors the rlm_gate plugin, which resets its fired-gate set after
    every 90% compaction.

    ``summary_from_cache`` models the gateway behaviour of summarizing from the
    warm KV cache (cache_prompt=True), so a deposit only pays for the summary
    generation, not a re-prefill of the window.
    """
    per_turn = total_tokens / n_turns
    prefill_s = gen_s = 0.0
    window_fill = 0.0
    deposits = 0
    crossed: set[float] = set()
    for i in range(n_turns):
        window_fill += per_turn
        prefill_s += curve.prefill_seconds(per_turn, window_fill)
        gen_s += curve.gen_seconds(per_turn, window_fill)
        for frac in GATE_FRACTIONS:
            if frac not in crossed and window_fill >= frac * window:
                crossed.add(frac)
                deposits += 1
                # write the summary (generation); re-prefill only if the KV
                # cache was not reused
                if not summary_from_cache:
                    prefill_s += curve.prefill_seconds(window_fill, window_fill)
                gen_s += curve.gen_seconds(summary_cost_tokens, window_fill)
                if frac == GATE_FRACTIONS[-1]:
                    # 90% compact: reset the window to the recent tail and
                    # arm the gates for the next cycle.
                    window_fill = summary_cost_tokens
                    crossed = set()
    total_s = prefill_s + gen_s
    return {
        "mode": "rolling_window_rlm",
        "window_tokens": window,
        "wall_seconds": round(total_s, 2),
        "avg_prefill_tps": round(_avg_tps(total_tokens, prefill_s), 2),
        "avg_gen_tps": round(_avg_tps(total_tokens, gen_s), 2),
        "avg_combined_tps": round(_avg_tps(total_tokens, total_s), 2),
        "gate_deposits": deposits,
    }


def run_comparison(
    curve: TpsCurve,
    total_tokens: int = 245_760,
    windows: tuple[int, ...] = (8192, 16384, 32768, 65536),
    out: str = "rolling_window_comparison.csv",
) -> str:
    """Compare full-context vs rolling-window across window sizes."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / out
    full = simulate_full_context(curve, total_tokens)
    rows = [full]
    for w in windows:
        rows.append(simulate_rolling(curve, total_tokens, window=w, summary_from_cache=True))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return str(out_path)

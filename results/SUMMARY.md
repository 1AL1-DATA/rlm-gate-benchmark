# RLM Rolling-Window Benchmark — Summary

**Headline:** a rolling window at **32,768 tokens** + RLM gate runs a **1.67x faster** than a full-context conversation, while keeping answer accuracy within **0.000 of full-context** (92% vs 92%).

## Key metrics

| Metric | Value |
| --- | --- |
| Peak prefill throughput | 308.3 t/s @ 8,192 ctx |
| Speedup (full-context → rolling 32,768) | **1.67x** |
| Full-context wall time | 20242 s |
| Rolling-wall time | 12148 s |
| Gate deposits | 24 summaries |
| Full-context accuracy | 92% |
| Rolling + RLM accuracy | 92% |
| Accuracy gap | 0.000 |
| Old-fact accuracy (via RLM) | 90% |
| Recent-fact accuracy | 100% |
| Tier recall | {"0.3": 0.167, "0.6": 0.25, "0.9": 0.167} |

## Figures

- `figures/accuracy_retained.png`
- `figures/context_vs_speed.png`
- `figures/drift_distribution.png`
- `figures/fact_retention.png`
- `figures/gpu_utilization.png`
- `figures/headline_summary.png`
- `figures/rolling_window_speedup.png`

## Context drift: what is lost, and the fix

| Failure mode | before | after (codified reminders) |
| --- | --- | --- |
| absent | 0 | 0 |
| collision | 2 | 1 |
| hallucinated | 0 | 0 |
| paraphrased | 0 | 0 |
| stale | 0 | 0 |
| unbound | 1 | 0 |

Misses: 3 -> 1. Reminder multiplicity grows 1x->4x as a fact ages 0->3 quarter-steps into the forget landscape; the emoji legend is chosen by the model per run.

Reproduce with: `python -m src.analyze`

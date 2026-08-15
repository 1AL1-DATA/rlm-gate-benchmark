# Data

This benchmark is **fully synthetic** — no external dataset downloads are
required. Every measurement is produced against the local llama.cpp server
(`localhost:8082`, model `Qwen3.6-35B-A3B-UD-IQ2_M.gguf`).

## Layout

| Path | Contents |
| --- | --- |
| `src/sweep.py` | context-vs-speed sweep prompts (procedurally generated) |
| `src/eval.py` | synthetic fact-retention conversation + eval prompts |
| `results/context_speed_sweep.csv` | measured throughput at each context fill |
| `results/rolling_window_comparison.csv` | simulated wall-time comparison |
| `results/eval_*.json/csv` | model-performance eval outputs |

## Reproducibility

```bash
python -m src.sweep --repeats 2   # requires llama-server on :8082
python -m src.eval                # requires llama-server on :8082
python -m src.analyze             # full pipeline: comparison + figures + SUMMARY
```

The sweep and eval prompts are seeded (`seed=42` / `seed=7`) so the synthetic
data is identical across runs.

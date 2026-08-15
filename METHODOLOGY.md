# Methodology

This document describes how the benchmark was performed, in enough detail that
someone with a similar background could reproduce the work without reading the code.

If you only want to **run** the benchmark, follow the `README.md` "Reproducing"
section. If you want to **understand** the analysis, read this.

## Research question

Can a long-running model conversation be kept inside a small working window — rolling
the context forward and re-injecting important facts from a durable memory — without
losing answer accuracy, while gaining wall-clock speed?

Operationalised as three sub-questions:

1. **Speed**: how does prefill / generation / GPU utilisation change as the context
   size grows?
2. **Wall time**: for the same 245,760-token conversation, is a rolling window + RLM
   context gate faster than a single full context, and at which window size?
3. **Accuracy**: does fact-recall accuracy survive the roll, and can a drift-aware
   reminder mechanism close the gap to full context?

## Setup

### Hardware / server

- GPU: **8 GB VRAM** (single mid-range consumer card).
- Server: llama.cpp `llama-server` on `http://localhost:8082`, launched with `--metrics`,
  `-c 250000` (reported `n_ctx = 250112`), KV cache quantised to `q8_0`, MoE FFN
  experts offloaded to CPU (all other layers on GPU).
- Model: `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (35B MoE, IQ2_M quant, 11.5 GB).
- Generation parameters: temperature 0.0 (deterministic where possible), `n_predict`
  16 for the sweep probes.

### Software

- Python ≥ 3.10 with `numpy`, `matplotlib`, `requests` (no pandas, no pytest).
- Tests run via a self-contained inline importlib harness (see README).

## Speed sweep (sub-question 1)

`src/sweep.py`

- A synthetic conversation is built for each context size in
  `[8192, 16384, 32768, 65536, 131072, 196608, 245760]`.
- For each size the server is asked to prefill and generate **16 tokens**; the run is
  repeated twice and the best (min-time) run is kept.
- Reported metrics per size: **prefill t/s**, **generation t/s**, **GPU utilisation %**
  (from the server's `/metrics` endpoint during the probe).
- **Sweep cap**: the 245,760 point is unreachable. Prompt construction overshoots the
  server's `n_ctx` (250,112) by ~8% (~265 K actual), and the server returns HTTP 400.
  The practical maximum is **196,608**; the 245,760 entry is dropped from the curve.

Result: `results/context_speed_sweep.csv`.

## Rolling-window wall-time comparison (sub-question 2)

`src/rolling_window.py`

- A single 245,760-token conversation is simulated twice:
  - **Full context**: every turn sent to the server with the whole conversation.
  - **Rolling + RLM**: conversation runs in a fixed window; when context fill crosses
    **30% / 60% / 90%**, the oldest portion is compacted into a summary (a *gate
    deposit*) which is re-injected into the next window. The gate re-arms after the
    90% compaction.
- Because the 245,760 sweep point is unreachable, the simulated conversation is
  clamped at 196,608 tokens; the full-context baseline is measured at the server's
  actual limit.
- Each rolling configuration is run with `summary_from_cache=True` — summaries are
  generated from the warm KV cache (the realistic serving path), so summarisation does
  not re-prefill history.
- Wall time is the total server round-trip time; combined t/s = tokens / wall time.
- Window sizes compared: `[8192, 16384, 32768, 65536]` plus the full-context baseline.

Result: `results/rolling_window_comparison.csv`.

## Accuracy eval (sub-question 3)

`src/eval.py`

### Conversation

A synthetic 12-fact conversation (6,012 tokens) describing a fictional system
deployment — codename, port, KV-cache dtype, offload target, context size, gate
threshold, memory system, rolling window, queue discipline, GPU size, model, backup
location. Facts have controlled shapes: `number:unit`, `word`, `abbrev`,
`letter_number:context`, etc. Ten facts are "old" (seeded early), two are "recent"
(seeded late) — see `old_fact_keys` / `recent_fact_keys` in the eval summary JSON.

### Conditions

- **Full context**: the whole conversation, no rolling.
- **Rolling + RLM**: the conversation runs through the rolling gate; the RLM memory is
  queried for the surviving facts.
- **Rolling + RLM + reminders**: same, plus drift-aware reminders (below).
- **Rolling, no reminders**: the control that shows what rolling loses.

### Inference

The model is a **thinking model**, so the eval uses the `/chat/completions` endpoint
with `"chat_template_kwargs": {"enable_thinking": false}` and reads `message.content`
(the thinking-token path otherwise returns empty content). Fact questions are asked
one per fact; an answer matches if it equals the seeded value (normalised).

### Accuracy metrics

- `full_context_accuracy` — fraction of 12 facts answered correctly in full context.
- `rlm_rolling_accuracy` — same, through the rolling + RLM pipeline.
- `accuracy_gap` = full − rolling.
- `old_fact_accuracy`, `recent_fact_accuracy` — recall split by seeding position.
- `tier_recall` — recall per forget-quarter tier (0.3 / 0.6 / 0.9), aggregated across
  all compaction cycles.

## Drift analysis and reminders

`src/drift.py`, `src/eval.py::pick_emojis / build_reminders`

### Miss taxonomy

Every wrong answer is classified along two axes:

- **mode** — `absent` (fact not in context), `unbound` (value present but lost its
  key/attribute), `collision` (two facts merged into one value), `paraphrased`
  (rewritten but recoverable), `stale` (superseded), `hallucinated` (invented).
- **shape** — the answer type: `number:unit`, `word`, `abbrev`, `letter_number:context`.

### Forget-landscape reminders

- Each fact is codified as **key — emoji — value**. The emoji for each key is chosen by
  the model at the start of the run (`pick_emojis`, output format `key|emoji`) — the
  codification is **not hardcoded**; it adapts every run.
- The **age quarter** of a fact is its position in the conversation mapped to a quarter
  (0 = newest, 3 = oldest).
- **Multiplicity rule**: for each ¼ the fact has aged into the forget landscape, add
  the codified pair once more to the context:

  > multiplicity = 1 + age_quarters

  A fact at quarter 3 is reminded 4×; a recent fact at quarter 0 once. Both the RLM
  memory and the context own the reminder — the pair appears in the context *and* in
  the RLM summary.
- `build_reminders()` emits these; `run_eval(out_prefix, reminders=True/False)` toggles
  them.

### Drift profile

`src/drift.py::build_drift_profile` aggregates misses across both eval conditions into
`results/drift_profile.json`: per condition, `n_misses`, `modes`, `shapes`,
`quarters` (recall per forget-quarter), and the individual miss records
(`fact_key`, `expected`, `shape`, `mode`, `answer`).

## Figures

`src/figures.py`, Pumpkin Spice theme in `src/theme.py`.

- `context_vs_speed.png` — sweep: prefill/gen/GPU vs context size.
- `gpu_utilization.png` — GPU utilisation across the sweep.
- `rolling_window_speedup.png` — wall time / speedup vs window size.
- `fact_retention.png` — accuracy by fact / condition.
- `accuracy_retained.png` — rolling accuracy vs full-context baseline.
- `headline_summary.png` — the headline comparison table as an image.
- `drift_distribution.png` — miss modes before vs after reminders, per forget-quarter.

All figures use the **Pumpkin Spice theme** (`src/theme.py`): pumpkin `#ff6700` =
before, rich cerulean `#3a6ea5` = after, steel azure `#004e98` = primary, platinum
`#ebebeb` / silver `#c0c0c0` = neutrals, black `#1a1a1a` = numbers and strings.
`scripts/check_figures.py` verifies every PNG is valid, well-proportioned, and uses
only theme colours.

## Reproducing everything

```bash
pip install -r requirements.txt
python -m src.sweep
python -m src.rolling_window
python -m src.analyze      # A/B evals + drift profile + figures + SUMMARY.md
python scripts/check_figures.py
```

`src/analyze.py::run_all` orchestrates the eval A/B, the drift profile, figure
rendering, and `results/SUMMARY.md`.

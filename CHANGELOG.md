# Changelog

All notable changes to this repository are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `src/theme.py` — Pumpkin Spice theme (palette tokens + matplotlib `use_theme`/
  `swiss_style`/`value_labels`), single source of truth for the figure palette.
- All figures restyled to the Pumpkin Spice palette (pumpkin = before, rich
  cerulean = after, steel azure = primary, platinum/silver = neutrals, black
  `#1a1a1a` = numbers and strings) with a DejaVu Sans fallback when Helvetica
  families are unavailable.
- `scripts/check_figures.py` — structural verification of every generated PNG
  (valid image, Pumpkin-Spice-only colours incl. antialiasing blends, aspect ratio).
- `figures/overview.png` — one graph summarising everything (throughput, rolling
  speedup, accuracy, drift); featured front-line in the README.
- `README.md`, `linkedin_post.md`, `blog_post.md`, `METHODOLOGY.md`,
  `research_report.md`, `arxiv_paper.tex` (draft with agent-actionable roadmap
  R1--R7), `AUTHORS`, `LICENSE` (MIT), `CITATION.cff`, `CHANGELOG.md`,
  `CONTRIBUTING.md`.

## [0.1.0] — 2026-08-15

### Added

- Context-size speed sweep (`src/sweep.py`) → `results/context_speed_sweep.csv`:
  prefill / generation / GPU utilisation across 8,192 → 196,608 tokens
  (2 repeats, best kept). Peak prefill 308.3 t/s @ 8 K; generation 24.4 → 8.7 t/s.
- Rolling-window wall-clock comparison (`src/rolling_window.py`) →
  `results/rolling_window_comparison.csv`: full context vs 8/16/32/64 K windows
  over a 245,760-token conversation. 32,768-token window: 1.67× speedup
  (12,148 s vs 20,242 s), 24 gate deposits. Summaries generated from the warm KV
  cache; gate deposits re-arm after each 90% compaction.
- Accuracy eval (`src/eval.py`): 12-fact synthetic conversation, full-context vs
  rolling + RLM, `/chat/completions` with `enable_thinking: false`.
  `results/eval_{before,after}_summary.json`, answers CSVs, summaries, legend.
  Without reminders: 75% (3 misses). With drift-aware reminders: 92% (1 miss),
  gap 0.000 vs full context.
- Drift analysis (`src/drift.py`): miss taxonomy (absent / unbound / collision /
  paraphrased / stale / hallucinated), value-shape classifier (`number:unit`,
  `word`, `abbrev`, `letter_number:context`), forget-quarter recall →
  `results/drift_profile.json`.
- Forget-landscape reminders: facts codified as `key|emoji|value` with the emoji
  chosen by the model per run (`pick_emojis`); multiplicity = 1 + age-quarters.
- Figures (`src/figures.py`): 8 Pumpkin-Spice-themed PNGs incl. `overview.png`
  and `drift_distribution.png` (miss modes before vs after, per forget-quarter).
- `results/SUMMARY.md` — headline ("1.67x faster … within 0.000 of full-context
  (92% vs 92%)"), key-metrics table, drift section.
- Tests (`tests/test_benchmark.py`): 21 tests — drift taxonomy, value shapes,
  classify-miss modes, quarter boundaries, reminder multiplicity, quarter
  accuracy, classify-rows; self-contained importlib harness (no pytest).

### Fixed

- Sweep: 245,760 point dropped (unreachable — prompt construction overshoots the
  250,112-token `n_ctx`, HTTP 400).
- `simulate_rolling`: gate-cycle bug (crossed set never reset, gates fired once
  per conversation) — fired set now resets after the 90% compaction.
- Eval: switching to `/chat/completions` (thinking model returned empty content
  via `/completion`); tier-recall aggregation now accumulates across cycles.
- `SUMMARY.md` headline: `int()` truncation of speedup (1.67 → 1x) replaced with
  the formatted float.

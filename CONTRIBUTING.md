# Contributing

Thanks for your interest in this project. There are several ways to contribute,
ordered from easiest to most involved.

## Reporting a bug

Open an issue on the GitHub issue tracker with:
- A minimal reproducible example (the script, the data subset, the actual output, the expected output).
- Your Python and dependency versions (`python --version`, `pip freeze`).
- The full traceback if the script crashes.

## Fixing a bug

Open a pull request with:
- A test that fails before your fix and passes after (see `tests/test_benchmark.py`).
- A clear commit message in the imperative mood ("fix off-by-one in eval.py", not "fixed bug").
- No unrelated formatting changes.

## Adding a feature

1. Open an issue first to discuss. This is a small project; we want to make sure
   new features fit the existing architecture before you spend time on them.
2. Add unit tests in `tests/`.
3. Update `README.md` if the feature changes the public surface.
4. Update the `CHANGELOG.md`.

## Adding a new drift mode or shape

The drift taxonomy lives in `src/drift.py` (`classify_miss` and `value_shape`).
To extend it:

1. Add the new mode/shape to the classifier and to the `EXPECTED` metadata used by
   `build_drift_profile` (it aggregates into `results/drift_profile.json`).
2. Add at least one unit test with a hand-computed expected classification.
3. Re-run `python -m src.analyze` — the drift profile and `drift_distribution.png`
   pick the new category up automatically.
4. Update the mode/shape tables in `METHODOLOGY.md` and `research_report.md`.

## Adding a benchmark condition (model, GPU, conversation)

1. Model / GPU: the only hardcoded dependency is `src/config.py` (server URL,
   `N_CTX`, sweep sizes). Everything else reads from the server; run a second
   server instance and point `RLM_BENCH_SERVER` at it.
2. New conversation: add the seeded facts to `src/eval.py` (they must keep the
   controlled shapes the drift classifier expects). Facts are seeded by position,
   so `old_fact_keys` / `recent_fact_keys` derive automatically.
3. Add a results row to the comparison: re-run `python -m src.rolling_window`.

## Coding style

- Type hints on all public functions.
- Docstrings on all public functions.
- Tests for all non-trivial functions.
- No pandas / pytest dependencies — the repo runs on numpy, matplotlib, requests
  plus the stdlib. Tests run via the inline importlib harness (see README).
- `src/theme.py` is the single source of the Pumpkin Spice palette. If you change
  the palette, change it here and regenerate all figures with `python -m src.analyze`.

## What we will *not* accept

- Breaking changes to the public API of `src/drift.py::classify_miss` or
  `src/eval.py::run_eval` (this breaks every downstream analysis).
- Refactoring that doesn't change behavior or performance (this is research code;
  readability and reproducibility matter more than DRY).
- New dependencies outside `numpy`, `matplotlib`, `requests`. If you need
  something exotic, please discuss in an issue first.
- Hardcoded emoji codifications — the whole point of `pick_emojis` is that the
  model chooses the codification each run.

## Code of conduct

Be kind, be patient, and assume good faith. This is a research artifact;
the goal is to share findings honestly, not to win arguments.

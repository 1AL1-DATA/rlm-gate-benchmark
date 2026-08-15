"""Shared configuration for the RLM rolling-window benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"

SERVER_URL = os.environ.get("RLM_BENCH_SERVER", "http://localhost:8082")
MODEL_NAME = "Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
N_CTX = 250112  # llama.cpp server n_ctx (from /props)

# Context-fill levels (prompt tokens) for the speed sweep.
SWEEP_SIZES = [8192, 16384, 32768, 65536, 131072, 196608, 245760]

# Generation length for each speed probe (tokens).
SWEEP_N_PREDICT = 16

# Rolling window sizes to compare (effective context the model operates in).
ROLLING_WINDOWS = [4096, 8192, 16384, 32768]

# Default full-context baseline for the rolling-window comparison.
FULL_CONTEXT_SIZE = 65536


def load_json(name: str, fallback=None):
    p = RESULTS / name
    if not p.exists():
        return fallback
    with open(p) as f:
        return json.load(f)


def save_json(name: str, data) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    return p

"""Context-vs-speed sweep against a llama.cpp server.

Measures prefill throughput (prompt_per_second) and generation throughput
(predicted_per_second) at increasing context-fill levels, together with GPU
utilization sampled from nvidia-smi during each probe.  The RLM gate keeps the
model operating in a small rolling window; these curves quantify how much
speed is recovered versus running the same model at a large context fill.

Uses the native ``/completion`` endpoint so llama.cpp reports exact timings.
Each context level is probed ``repeats`` times (median reported) to damp
server noise from concurrent slots.
"""

from __future__ import annotations

import csv
import statistics
import subprocess
import threading
import time

import requests

from .config import RESULTS, SERVER_URL

FILLER = (
    "The architecture separates the memory tier from the reasoning tier so "
    "that context growth does not degrade generation throughput. Each module "
    "keeps a bounded rolling window and offloads durable state to external "
    "storage. Token estimation must stay cheap because it runs on every turn."
)

TIMEOUT_S = 3600

SWEEP_SIZES_DEFAULT = [8192, 16384, 32768, 65536, 131072, 196608, 245760]


class GpuSampler:
    """Samples nvidia-smi GPU util%% + memory in a background thread."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.util: list[float] = []
        self.mem: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.util = []
        self.mem = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                util, mem = out.stdout.strip().split(",")
                self.util.append(float(util))
                self.mem.append(int(mem))
            except Exception:
                pass
            time.sleep(self.interval)

    def median(self) -> tuple[float, int] | None:
        if not self.util:
            return None
        return round(statistics.median(self.util), 1), int(statistics.median(self.mem))


def build_prompt(n_tokens: int) -> str:
    """Build a synthetic prompt of approximately n_tokens (English words)."""
    block_tokens = len(FILLER.split())
    blocks = max(1, n_tokens // block_tokens)
    return " ".join([FILLER] * blocks)


def server_idle(server: str = SERVER_URL) -> bool:
    """True if all slots are free (no live gateway contention)."""
    try:
        slots = requests.get(f"{server}/slots", timeout=10).json()
        for s in slots:
            state = s.get("state")
            if state not in (None, 0, "idle", "IDLE", "free"):
                return False
            if s.get("n_past") not in (None, 0):
                return False
            if s.get("cache_tokens"):
                return False
        return True
    except Exception:
        return True


def probe(
    prompt: str,
    n_predict: int,
    sample_gpu: bool = False,
    server: str = SERVER_URL,
    retries: int = 3,
) -> dict:
    """One completion call; returns normalized timing + usage fields.

    Retries on 5xx / connection errors so shared-server contention (the live
    hermes gateway on the same instance) doesn't kill a whole sweep.
    """
    sampler = GpuSampler() if sample_gpu else None
    if sampler:
        sampler.start()
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": False,
        "seed": 42,
        "temperature": 0.0,
        "timings_per_token": False,
    }
    last_error = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = requests.post(f"{server}/completion", json=payload, timeout=TIMEOUT_S)
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = str(e)
            time.sleep(5 * (attempt + 1))
            continue
        except requests.exceptions.HTTPError as e:
            last_error = str(e)
            time.sleep(5 * (attempt + 1))
            continue
        wall = time.time() - t0
        body = resp.json()
        timings = body.get("timings", {})
        gpu = sampler.median() if sampler else None
        if sampler:
            sampler.stop()
        return {
            "prompt_n": int(timings.get("prompt_n", 0)),
            "prompt_per_second": float(timings.get("prompt_per_second", 0.0)),
            "predicted_n": int(timings.get("predicted_n", 0)),
            "predicted_per_second": float(timings.get("predicted_per_second", 0.0)),
            "wall_seconds": wall,
            "gpu_util_median": gpu[0] if gpu else None,
            "gpu_mem_median": gpu[1] if gpu else None,
        }
    if sampler:
        sampler.stop()
    raise RuntimeError(f"probe failed after {retries} attempts: {last_error}")


def _median(values: list[float]) -> float:
    return statistics.median([v for v in values if v > 0])


def run_sweep(
    sizes: list[int] | None = None,
    n_predict: int = 96,
    repeats: int = 2,
    sample_gpu: bool = True,
    server: str = SERVER_URL,
    out: str = "context_speed_sweep.csv",
    verbose: bool = True,
) -> str:
    """Run the context-vs-speed sweep; returns the output CSV path."""
    sizes = sizes or SWEEP_SIZES_DEFAULT
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / out
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "context_tokens",
                "repeats",
                "prefill_tps_median",
                "prefill_tps_all",
                "gen_tps_median",
                "gen_tps_all",
                "gpu_util_median_pct",
                "gpu_mem_median_mib",
                "wall_seconds_total",
            ]
        )
        for size in sizes:
            prompt = build_prompt(size)
            prefill_all: list[float] = []
            gen_all: list[float] = []
            gpu_utils: list[float] = []
            gpu_mems: list[int] = []
            wall_total = 0.0
            for _ in range(repeats):
                row = probe(prompt, n_predict=n_predict, sample_gpu=sample_gpu, server=server)
                prefill_all.append(row["prompt_per_second"])
                gen_all.append(row["predicted_per_second"])
                if row["gpu_util_median"] is not None:
                    gpu_utils.append(row["gpu_util_median"])
                    gpu_mems.append(row["gpu_mem_median"])
                wall_total += row["wall_seconds"]
            prefill_m = _median(prefill_all)
            gen_m = _median(gen_all)
            gpu_m = statistics.median(gpu_utils) if gpu_utils else 0.0
            mem_m = int(statistics.median(gpu_mems)) if gpu_mems else 0
            if verbose:
                print(
                    f"ctx={size:>7}  prefill={prefill_m:7.2f} t/s  gen={gen_m:6.2f} t/s  "
                    f"gpu={gpu_m:5.1f}% {mem_m}MiB  wall={wall_total:8.1f}s (n={repeats})",
                    flush=True,
                )
            writer.writerow(
                [
                    size,
                    repeats,
                    round(prefill_m, 4),
                    json_prettify(prefill_all),
                    round(gen_m, 4),
                    json_prettify(gen_all),
                    round(gpu_m, 1),
                    mem_m,
                    round(wall_total, 2),
                ]
            )
    return str(out_path)


def json_prettify(values: list[float]) -> str:
    import json

    return json.dumps([round(v, 4) for v in values])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--server", type=str, default=SERVER_URL)
    args = ap.parse_args()
    print(run_sweep(sizes=args.sizes, n_predict=args.n_predict, repeats=args.repeats,
                    sample_gpu=not args.no_gpu, server=args.server))

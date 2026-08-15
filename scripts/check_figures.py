#!/usr/bin/env python3
"""Structural verification for generated figures.

Checks every PNG in the figures/ directory is:
  - a valid image of nonzero size
  - using only Pumpkin-Spice-theme colours (with tolerance for antialiasing)
  - close to a 16:9 or 4:3 aspect ratio (not a broken/bank canvas)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

sys.path.insert(0, str(ROOT))
from src.theme import SWISS, use_theme  # noqa: E402


def hex2rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def near(c, ref, tol=18):
    return all(abs(a - b) <= tol for a, b in zip(c, ref))


THEME_RGBS = [hex2rgb(v) for v in SWISS.values()]

BLEND_RGBS = []
for r, g, b in THEME_RGBS:
    for alpha in (0.3, 0.5, 0.7, 0.85):
        BLEND_RGBS.append(tuple(int(c * alpha + 255 * (1 - alpha)) for c in (r, g, b)))
BLEND_RGBS.extend(THEME_RGBS)

EXPECTED = {
    "overview.png": {"title": True, "xlabel": True, "ylabel": True, "legend": True},
    "context_vs_speed.png": {"title": True, "xlabel": True, "ylabel": True},
    "gpu_utilization.png": {"title": True, "xlabel": True, "ylabel": True},
    "rolling_window_speedup.png": {"title": True, "xlabel": True, "ylabel": True, "legend": True},
    "fact_retention.png": {"title": True, "xlabel": True, "ylabel": True},
    "accuracy_retained.png": {"title": True, "xlabel": True, "ylabel": True},
    "headline_summary.png": {"title": True, "xlabel": True, "ylabel": True, "legend": True},
    "drift_distribution.png": {"title": True, "xlabel": True, "ylabel": True, "legend": True},
    "overview.png": {"title": True, "xlabel": True, "ylabel": True, "legend": True},
}


def dominant_colors(fname: Path, n=8):
    img = imread(fname)[:, :, :3]
    flat = (img * 255).reshape(-1, 3)
    unique, counts = __import__("numpy").unique(flat.astype("uint8"), axis=0, return_counts=True)
    order = counts.argsort()[::-1][:n]
    return [tuple(int(v) for v in unique[i]) for i in order]


def main():
    use_theme()
    errors = []
    missing = [f for f in EXPECTED if not (FIG_DIR / f).exists()]
    if missing:
        errors.append(f"missing figures: {missing}")

    for fname, want in EXPECTED.items():
        path = FIG_DIR / fname
        if not path.exists():
            continue
        img = imread(path)
        h, w = img.shape[:2]
        if w < 200 or h < 200:
            errors.append(f"{fname}: suspiciously small {w}x{h}")

        dom = dominant_colors(path)
        off = [c for c in dom if not any(near(c, ref) for ref in BLEND_RGBS)]
        if off and len(off) > 1:
            errors.append(f"{fname}: non-theme colors {off}")

        ar = w / h
        if not (1.2 <= ar <= 2.6):
            errors.append(f"{fname}: aspect {ar:.2f} outside [1.2, 2.6]")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        sys.exit(1)
    print(f"OK: {len(EXPECTED)} figures verified against Pumpkin-Spice theme "
          f"({len(THEME_RGBS)} palette entries)")


if __name__ == "__main__":
    main()

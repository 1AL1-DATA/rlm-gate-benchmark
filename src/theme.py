"""Pumpkin Spice theme — warm autumnal palette with cerulean accents.

Palette (as specified):
  Pumpkin Spice   #ff6700  warm, bold, festive — the "before"/warning accent
  Platinum        #ebebeb  soft silver-white — surfaces, faint grids
  Silver          #c0c0c0  metallic neutral — context, reference series
  Rich Cerulean   #3a6ea5  deep trusting blue — the "after"/gain accent
  Steel Azure     #004e98  dark steel blue — primary series

An extra near-black "ink" (#1a1a1a) is derived for text and spines, because the
five named colours are all too light for body copy on white. Keeping the tokens
here means every figure renders from one source of truth.
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib as mpl

# ---- Palette -------------------------------------------------------------

PUMPKIN = "#ff6700"
PLATINUM = "#ebebeb"
SILVER = "#c0c0c0"
CERULEAN = "#3a6ea5"
STEEL = "#004e98"
INK = "#1a1a1a"
WHITE = "#ffffff"

SWISS = {
    "pumpkin": PUMPKIN,
    "platinum": PLATINUM,
    "silver": SILVER,
    "cerulean": CERULEAN,
    "steel": STEEL,
    "ink": INK,
    "white": WHITE,
}

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

SERIES = [STEEL, CERULEAN, PUMPKIN, SILVER, PLATINUM]

# Semantic roles used by the benchmark figures.
BEFORE = PUMPKIN      # baseline / what was lost
AFTER = CERULEAN      # intervention / what was gained
PRIMARY = STEEL       # main series
NEUTRAL = SILVER      # context / reference


def _font_family() -> list[str]:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
    return [f for f in preferred if f in available] or ["DejaVu Sans"]


def use_theme() -> None:
    """Register the Pumpkin Spice visual language as the matplotlib default."""
    mpl.rcParams.update(
        {
            "font.family": _font_family(),
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelweight": "bold",
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": PLATINUM,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def swiss_style(
    ax: Any,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Any:
    """Apply the theme chrome to a matplotlib Axes.

    White canvas, ink spines (left + bottom only), platinum grid, uppercase
    tracked micro-labels, sharp corners. Titles are uppercased to match the
    dashboard typography.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, length=3)
    if title:
        ax.set_title(title.upper(), loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel.upper())
    if ylabel:
        ax.set_ylabel(ylabel.upper())
    ax.grid(True, axis="y", alpha=1.0)
    return ax


def value_labels(ax: Any, bars: Any, fmt: Callable[[float], str] = lambda v: f"{v:.0f}") -> None:
    """Ink value labels above bars/points, black for readability."""
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(
            x, y + max(ax.get_ylim()[1] * 0.015, 0.02),
            fmt(y), ha="center", va="bottom",
            fontsize=8, color=INK, fontweight="bold",
        )

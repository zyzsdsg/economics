"""Shared matplotlib style for notebooks and reports.

Series colours follow a fixed categorical order (never cycled by rank); use at most
three colours in one panel and facet beyond that. Text stays in neutral ink.
"""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#8a8984"  # reference lines, synthetic "truth", confidence bands


def use_lab_style() -> None:
    """Apply the project style to all subsequent matplotlib figures."""
    mpl.rcParams.update(
        {
            "figure.figsize": (9, 4),
            "figure.dpi": 110,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 4,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "legend.frameon": False,
            "font.size": 10,
        }
    )

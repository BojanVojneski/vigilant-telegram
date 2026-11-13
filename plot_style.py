"""Shared visual style for VerA charts."""

from __future__ import annotations

import matplotlib
from cycler import cycler

NAVY = "#153B5B"
BLUE = "#477EA8"
TEAL = "#4D9078"
GOLD = "#C28A3A"
ORANGE = "#C98245"
RED = "#B85C5C"
SLATE = "#718096"
INK = "#263746"
MUTED = "#667085"
GRID = "#DCE3EA"


def apply_plot_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#AAB4BF",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(
                color=[BLUE, ORANGE, TEAL, RED, GOLD, SLATE]
            ),
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 180,
        }
    )

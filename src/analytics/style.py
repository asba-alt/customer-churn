"""Shared plotting configuration so every chart in the project looks the same."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe: never tries to open a window

import matplotlib.pyplot as plt
import seaborn as sns

import config

PALETTE = {
    "primary": "#1f3b73",
    "accent": "#e4572e",
    "teal": "#2a9d8f",
    "amber": "#e9c46a",
    "grey": "#8d99ae",
    "danger": "#c1121f",
    "retained": "#2a9d8f",
    "churned": "#e4572e",
}

CHURN_COLORS = {"No": PALETTE["retained"], "Yes": PALETTE["churned"], 0: PALETTE["retained"], 1: PALETTE["churned"]}

SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"


def apply_style() -> None:
    """Apply the project-wide matplotlib/seaborn theme (idempotent)."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#c9ccd3",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": PALETTE["primary"],
            "axes.labelsize": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
        }
    )


def new_figure(width: float = 9.0, height: float = 5.0):
    """Create a styled figure/axes pair."""
    apply_style()
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def save(fig, filename: str) -> str:
    """Save a figure into data/exports/figures and return its repo-relative path."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / filename
    fig.savefig(path)
    plt.close(fig)
    return path.relative_to(config.ROOT).as_posix()


def title_with_caption(ax, title: str, caption: str | None = None) -> None:
    """Consistent title plus a small explanatory caption under the axes."""
    ax.set_title(title, loc="left", pad=12)
    if caption:
        ax.figure.text(0.01, -0.04, caption, ha="left", va="top", fontsize=8, color="#5c6370")


def annotate_bars(ax, values, fmt: str = "{:,.0f}", offset: float = 0.01) -> None:
    """Label each bar with its value (used on count/percentage charts)."""
    xmax = ax.get_xlim()[1]
    for patch, value in zip(ax.patches, values):
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + xmax * offset,
            y,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

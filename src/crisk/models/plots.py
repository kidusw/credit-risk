"""Static charts for the README (PNG). Palette: the validated categorical slots 1-5.

Three of the five slots sit below 3:1 contrast on the light surface, so identity never
rests on colour alone: each model family has its own line style and marker, every line is
labelled at its end, and the README carries the same numbers as a table.
"""

from itertools import pairwise
from pathlib import Path

import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# model -> (label, colour, linestyle, marker); colour order fixed, never by rank.
SERIES = {
    "gbm_with_pricing": ("LightGBM + pricing", "#2a78d6", "-", "o"),
    "gbm_no_pricing": ("LightGBM, borrower only", "#eb6834", "-", "s"),
    "scorecard_with_pricing": ("Scorecard + pricing", "#1baf7a", "--", "^"),
    "scorecard_no_pricing": ("Scorecard, borrower only", "#eda100", "--", "D"),
    "baseline_lc_sub_grade": ("LC sub-grade baseline", "#e87ba4", ":", "v"),
}


def auc_by_vintage(by_vintage: pd.DataFrame, path: Path, valid_end: str = "2015Q1") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wide = by_vintage.pivot(index="issue_quarter", columns="variant", values="auc")
    quarters = list(wide.index)
    x = range(len(quarters))

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    low, high = float(wide.min().min()), float(wide.max().max())
    ax.set_ylim(low - 0.004, high + 0.008)
    split_at = quarters.index(valid_end) + 0.5
    ax.axvline(split_at, color=AXIS, linewidth=1)
    ax.text(split_at - 0.1, low - 0.0025, "validation", ha="right", color=MUTED, fontsize=9)
    ax.text(split_at + 0.1, low - 0.0025, "test", ha="left", color=MUTED, fontsize=9)

    ends = []
    for model, (label, colour, style, marker) in SERIES.items():
        if model not in wide:
            continue
        y = wide[model].to_numpy()
        ax.plot(
            x,
            y,
            color=colour,
            linestyle=style,
            linewidth=2,
            label=label,
            marker=marker,
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ends.append([float(y[-1]), float(y[-1])])  # [true value, label position]
    # Spread end labels so close values (e.g. two models at 0.700) don't overprint.
    gap = (high - low) * 0.06
    ends.sort(key=lambda e: e[0])
    for prev, cur in pairwise(ends):
        cur[1] = max(cur[1], prev[1] + gap)
    for value, pos in ends:
        ax.annotate(
            f"{value:.3f}",
            (len(quarters) - 1, value),
            xytext=(len(quarters) - 0.75, pos),
            textcoords="data",
            va="center",
            fontsize=9,
            color=INK_2,
        )

    ax.set_xticks(list(x), quarters)
    ax.set_xlim(-0.3, len(quarters) - 0.3)
    ax.set_ylabel("ROC-AUC", color=INK_2)
    ax.set_title(
        "Ranking quality (ROC-AUC) on held-out vintages", color=INK, loc="left", fontsize=12
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        ncols=3,
        handlelength=3,
    )
    fig.subplots_adjust(left=0.09, right=0.9, top=0.92, bottom=0.24)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path

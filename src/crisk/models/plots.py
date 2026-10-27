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


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def reliability(table: pd.DataFrame, path: Path, model: str = "gbm_with_pricing") -> Path:
    """Predicted vs observed default rate per PD decile on test: raw vs calibrated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table[table["model"] == model]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    top = float(max(t["mean_pd"].max(), t["default_rate"].max())) * 1.08
    ax.plot(
        [0, top], [0, top], color=AXIS, linewidth=1, linestyle="--", label="perfect calibration"
    )
    styles = {
        "raw": ("Raw PD", "#eb6834", "s", ":"),
        "platt": ("Calibrated (Platt)", "#2a78d6", "o", "-"),
        "isotonic": ("Calibrated (isotonic)", "#2a78d6", "o", "-"),
    }
    for version, g in t.groupby("version", sort=False):
        label, colour, marker, style = styles[version]
        ax.plot(
            g["mean_pd"],
            g["default_rate"],
            color=colour,
            linestyle=style,
            linewidth=2,
            marker=marker,
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
    _style(
        ax,
        "Test reliability by PD decile (2015Q2-2016Q1)",
        "Mean predicted PD",
        "Observed default rate",
    )
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def profit_curves(curves: pd.DataFrame, cutoffs: pd.DataFrame, path: Path, split="test") -> Path:
    """Two panels sharing the x axis: total profit and return on lent vs approval rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = curves[curves["split"] == split]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    panels = [
        ("profit", "Total profit ($M)", 1e-6),
        ("return_on_lent", "Return on amount lent", 100),
    ]
    for ax, (col, title, scale) in zip(axes, panels, strict=True):
        for model in ("gbm_with_pricing", "scorecard_with_pricing"):
            label, colour, style, _ = SERIES[model]
            g = c[c["model"] == model]
            ax.plot(
                g["approval_rate"] * 100,
                g[col] * scale,
                color=colour,
                linestyle=style,
                linewidth=2,
                label=label,
            )
        for r in cutoffs.itertuples():
            ax.plot(
                r.approval_rate * 100,
                getattr(r, col) * scale,
                linestyle="none",
                marker=r.marker,
                markersize=12,
                color=r.colour,
                markeredgecolor=INK,
                markeredgewidth=1.0,
                label=r.label,
                zorder=5,
            )
        _style(ax, title, "Approval rate (%)", "$M" if col == "profit" else "%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=9, labelcolor=INK_2
    )
    fig.suptitle(
        f"Approve the lowest-PD loans first ({split} vintages)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.27, wspace=0.22)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path

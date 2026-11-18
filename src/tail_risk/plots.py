import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

MARKET_COLOURS = {"sp500": "#1f6fb4", "euronext": "#e07b22"}
STOCK_MARKERS = ["o", "s", "^", "D", "v"]

K_LINESTYLES = {0.02: "dashed", 0.035: "solid", 0.05: "dotted"}
K_LINEWIDTHS = {0.02: 1.0, 0.035: 2.0, 0.05: 1.0}


def shade_k_band(ax, k_start: float = 0.02, k_end: float = 0.05, alpha: float = 0.15):
    label = "k range 2-5%"
    ax.axvspan(
        k_start, k_end, color="grey", alpha=alpha, linewidth=0, zorder=0, label=label
    )


def plot_hill_curves(
    market_results: dict[dict],
    stock_results: list[tuple],
    k_band: tuple[float, float] = (0.02, 0.05),
    k_floor: int = 10,
    shade_alpha: float = 0.15,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the shaded band of k-values of interest
    shade_k_band(ax, *k_band, alpha=shade_alpha)
    # Plot the stable-alpha boundary
    ax.axhline(2.0, color="0.3", linestyle="--", linewidth=1, label="α = 2")

    # Plot the single stocks as supplemental background
    for (label, share, hill_curve), marker in zip(stock_results, STOCK_MARKERS):
        ax.plot(
            share[k_floor:],
            hill_curve[k_floor:],
            color="0.6",
            linewidth=0.8,
            marker=marker,
            markersize=5,
            markevery=40,
            label=label,
        )

    for market, result in market_results.items():
        share = result["share"][k_floor:]
        quarter, half, three_quart = result["quartiles"][:, k_floor:]

        colour = MARKET_COLOURS[market]
        ax.fill_between(
            share, quarter, three_quart, color=colour, alpha=0.2, linewidth=0
        )
        ax.plot(
            share, half, color=colour, linewidth=2, label=f"{market} median (IQR band)"
        )

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("k/T (sample of the share treated as tail)")
    ax.set_ylabel(r"Hill estimate $\hat{\alpha}(k)$")
    ax.grid(axis="y", color="0.9")

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


def plot_ecdf(ecdf_results: dict[dict], main_pct: float = 0.035):
    fig, ax = plt.subplots(figsize=(8, 5))

    # The alpha = 2 line in background
    ax.axvline(2, color="0.3", linestyle="dashdot", alpha=0.85, zorder=0)
    # The median line at 50%
    ax.axhline(0.5, color="0.85", linewidth=0.8, zorder=0)

    market_handles = []
    for market, results in ecdf_results.items():
        colour = MARKET_COLOURS[market]
        for pct, alpha_hats in results.items():
            N = alpha_hats.shape[0]
            x = np.sort(alpha_hats)
            y = np.arange(1, N + 1) / N

            ax.step(
                x,
                y,
                where="post",
                color=colour,
                linestyle=K_LINESTYLES[pct],
                linewidth=K_LINEWIDTHS[pct],
            )

        median_alpha = np.median(results[main_pct])
        ax.plot(median_alpha, 0.5, marker="o", color=colour, markersize=6, zorder=3)

        market_label = (
            f"{market} ({N=}), median " + r"$\hat{\alpha}$ = " + f"{median_alpha:.2f}"
        )
        market_handles.append(
            Line2D([], [], color=colour, linewidth=2, label=market_label)
        )

    k_handles = [
        Line2D(
            [],
            [],
            color="0.4",
            linestyle=K_LINESTYLES[pct],
            linewidth=K_LINEWIDTHS[pct],
            label=f"k/T = {pct:.1%}",
        )
        for pct in K_LINESTYLES
    ]
    boundary_handle = Line2D(
        [], [], color="0.3", linestyle="dashdot", label=r"$\alpha = 2$"
    )
    all_handles = market_handles + k_handles + [boundary_handle]

    ax.legend(handles=all_handles, frameon=False, fontsize=8, loc="lower right")

    ax.set_xlabel(r"Hill (left) tail index $\hat{\alpha}$")
    ax.set_ylabel(r"share of stock with $\hat{\alpha} \leq x$")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

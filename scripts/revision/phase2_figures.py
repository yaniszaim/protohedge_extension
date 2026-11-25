#!/usr/bin/env python3
"""Figures for the Phase 2 diagnostics (reads the CSVs written by phase2_diagnostics.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / ".deephedging_real_runs" / "revision_v1" / "phase2_diagnostics"
DST = REPO / "paper" / "revision_2026_11" / "phase2"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5df"
SERIES = {  # fixed order; colour follows the entity in every figure
    "dh": ("Deep Hedging", "#2a78d6"),
    "ph_tail": ("ProtoHedge (tail-selected)", "#eb6834"),
    "spot_delta_listed": ("Spot-Delta (listed call's delta)", "#1baf7a"),
    "liability_delta": ("Liability delta hedge", "#eda100"),
}
TITLES = {"european_call": "European call", "asian_call": "Asian call"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def position_profiles():
    df = pd.read_csv(SRC / "position_profiles.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for i, liability in enumerate(("european_call", "asian_call")):
        for j, (col, name) in enumerate((("spot_position", "shares held"), ("call_position", "listed calls held"))):
            ax = axes[i, j]
            style(ax)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            for key, (label, color) in SERIES.items():
                d = df[(df.liability == liability) & (df.model == key)].sort_values("step")
                ax.plot(d.step + 1, d[col], color=color, linewidth=2, solid_capstyle="round",
                        label=label)
                ax.plot(d.step.iloc[-1] + 1, d[col].iloc[-1], "o", color=color, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.set_title(f"{TITLES[liability]}: {name}", fontsize=9.5, color=INK, loc="left")
            ax.set_ylim(-0.3, 1.0)
            ax.set_xlim(1, 20.6)
            ax.set_xticks([1, 5, 10, 15, 20])
            if i == 1:
                ax.set_xlabel("decision step (trading day of the episode)", fontsize=8.5, color=MUTED)
            if j == 0:
                ax.set_ylabel("mean position", fontsize=8.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Mean test-set position by step, v4 models (ten tickers, equal weight; three seeds for learned models)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(DST / "position_profiles.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def risk_by_ticker():
    df = pd.read_csv(SRC / "scoreboard_by_ticker.csv")
    tickers = sorted(df.ticker.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, liability in zip(axes, ("european_call", "asian_call")):
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        for key, (label, color) in SERIES.items():
            d = df[(df.liability == liability) & (df.model == key)].set_index("ticker").loc[tickers]
            ax.plot(d.he_std * 100, range(len(tickers)), "o", color=color, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_title(TITLES[liability], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("std of hedging error, % of initial spot (lower is better)", fontsize=8.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Test-set hedging-error dispersion by ticker, v4 models and rule-based hedges",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(DST / "risk_by_ticker.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    position_profiles()
    risk_by_ticker()
    print("wrote", DST)
    sys.exit(0)

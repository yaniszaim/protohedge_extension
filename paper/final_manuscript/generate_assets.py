#!/usr/bin/env python3
"""Build the paper-facing evidence archive, tables, and figures.

This script is intentionally read-only with respect to completed experiment
directories. It copies the small set of paper-relevant inputs into
``source_data`` and derives every manuscript asset from that archive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/protohedge-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DATA = HERE / "source_data"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"

NAVY = "#17324D"
TEAL = "#178F8A"
ORANGE = "#D97732"
BLUE = "#3E6FB0"
RED = "#B54A4A"
GOLD = "#CAA43D"
GRAY = "#6B7280"
LIGHT = "#E8EEF2"


SOURCES = {
    "historical_dataset_episode_summary.csv": ROOT
    / "paper/submission_rerun_scientific_v4/combined/tables/dataset_episode_summary.csv",
    "historical_descriptive_panel_averages.csv": ROOT
    / "paper/submission_rerun_scientific_v4/combined/tables/european_asian_panel_averages.csv",
    "historical_original_cvar50_bootstrap.csv": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/original_cvar50_utility_bootstrap.csv",
    "historical_original_cvar50_per_ticker.csv": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/original_cvar50_utility_per_ticker.csv",
    "historical_spot_delta_bootstrap.csv": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/proto_vs_spot_delta_bootstrap.csv",
    "historical_spot_delta_per_ticker.csv": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/proto_vs_spot_delta_per_ticker.csv",
    "historical_checkpoint_audit.csv": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/dh_checkpoint_audit.csv",
    "historical_checkpoint_audit_summary.csv": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/dh_checkpoint_audit_summary.csv",
    "historical_selected_checkpoint_summary.csv": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/selected_model_checkpoint_summary.csv",
    "historical_benchmark_tuning_fairness.json": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/benchmark_tuning_fairness.json",
    "historical_archive_manifest.json": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/archive_manifest.json",
    "historical_locked_protocol.json": ROOT
    / "paper/submission_rerun_scientific_v4/locked_protocol.json",
    "historical_raw_output_checksums.sha256": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/raw_output_checksums.sha256",
    "historical_panel_input_checksums.sha256": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/panel_input_checksums.sha256",
    "historical_source_snapshot_checksums.sha256": ROOT
    / "paper/submission_rerun_scientific_v4/reproducibility/source_snapshot_checksums.sha256",
    "panel_summary.json": ROOT / "Data/NEW_PANEL_DECISION_V2/panel_summary.json",
    "checkpoint_sensitivity_bootstrap.csv": ROOT
    / ".deephedging_real_runs/checkpoint_selection_sensitivity_v1/hybrid_checkpoint_sensitivity_bootstrap.csv",
    "checkpoint_sensitivity_completed_fits.csv": ROOT
    / ".deephedging_real_runs/checkpoint_selection_sensitivity_v1/completed_fit_comparisons.csv",
    "spy_softclip_metrics.csv": ROOT
    / ".deephedging_real_runs/spy_european_softclip_confirmation_v1/spy_model_metrics.csv",
    "spy_softclip_bootstrap.csv": ROOT
    / ".deephedging_real_runs/spy_european_softclip_confirmation_v1/spy_paired_block_bootstrap.csv",
    "spy_softclip_protocol.json": ROOT
    / ".deephedging_real_runs/spy_european_softclip_confirmation_v1/protocol.json",
    "historical_family_inventory.csv": ROOT
    / ".deephedging_real_runs/spy_european_softclip_confirmation_v1/historical_headline_family_inventory.csv",
    "historical_family_counts.json": ROOT
    / ".deephedging_real_runs/spy_european_softclip_confirmation_v1/historical_family_counts.json",
    "paper_faithful_validation_choices.csv": ROOT
    / ".deephedging_real_runs/spy_european_softclip_confirmation_v1/paper_faithful_validation_choices.csv",
    "spy_confirmation_environment.json": ROOT
    / ".deephedging_real_runs/spy_european_softclip_confirmation_v1/environment.json",
    "black_scholes_summary.csv": ROOT
    / "paper/black_scholes_notebook_confirmation/synthetic_summary.csv",
    "black_scholes_protocol.json": ROOT
    / "paper/black_scholes_notebook_confirmation/protocol.json",
    "black_scholes_assessment.json": ROOT
    / "paper/black_scholes_notebook_confirmation/reproduction_assessment.json",
    "black_scholes_environment.json": ROOT
    / "paper/black_scholes_notebook_confirmation/launcher_environment.json",
    "original_synthetic_summary.csv": ROOT
    / "paper/original_synthetic_reproduction/synthetic_summary.csv",
    "original_synthetic_protocol.json": ROOT
    / "paper/original_synthetic_reproduction/protocol.json",
    "original_synthetic_assessment.json": ROOT
    / "paper/original_synthetic_reproduction/reproduction_assessment.json",
    "qqq_european_effective_size.csv": ROOT
    / "paper/submission_rerun_scientific_v4/european_call/interpretability/tables/qqq_prototype_effective_size.csv",
    "qqq_asian_effective_size.csv": ROOT
    / "paper/submission_rerun_scientific_v4/asian_call/interpretability/tables/qqq_prototype_effective_size.csv",
    "qqq_european_concentration.csv": ROOT
    / "paper/submission_rerun_scientific_v4/european_call/interpretability/tables/qqq_prototype_concentration.csv",
    "qqq_asian_concentration.csv": ROOT
    / "paper/submission_rerun_scientific_v4/asian_call/interpretability/tables/qqq_prototype_concentration.csv",
    "qqq_european_regime_prototypes.csv": ROOT
    / "paper/submission_rerun_scientific_v4/european_call/interpretability/tables/qqq_dominant_prototype_by_regime.csv",
    "qqq_asian_regime_prototypes.csv": ROOT
    / "paper/submission_rerun_scientific_v4/asian_call/interpretability/tables/qqq_dominant_prototype_by_regime.csv",
    "qqq_european_nearest_states.csv": ROOT
    / "paper/submission_rerun_scientific_v4/european_call/interpretability/tables/qqq_nearest_historical_states.csv",
    "qqq_asian_nearest_states.csv": ROOT
    / "paper/submission_rerun_scientific_v4/asian_call/interpretability/tables/qqq_nearest_historical_states.csv",
    "qqq_european_nearest_episodes.png": ROOT
    / "paper/submission_rerun_scientific_v4/european_call/interpretability/figures/qqq_nearest_historical_episodes.png",
    "qqq_asian_nearest_episodes.png": ROOT
    / "paper/submission_rerun_scientific_v4/asian_call/interpretability/figures/qqq_nearest_historical_episodes.png",
    "final_confirmation.log": ROOT / "paper/final_confirmation.log",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def archive_sources() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for archive_name, source in SOURCES.items():
        if not source.is_file():
            missing.append(str(source))
            continue
        destination = SOURCE_DATA / archive_name
        shutil.copy2(source, destination)
        rows.append(
            {
                "archive_name": archive_name,
                "original_path": str(source.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    if missing:
        raise FileNotFoundError("Missing paper inputs:\n" + "\n".join(missing))

    manifest = pd.DataFrame(rows).sort_values("archive_name")
    manifest.to_csv(SOURCE_DATA / "source_manifest.csv", index=False)
    with (SOURCE_DATA / "source_checksums.sha256").open("w") as handle:
        for row in manifest.itertuples(index=False):
            handle.write(f"{row.sha256}  {row.archive_name}\n")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status_at_generation": git_output("status", "--short").splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__)),
        "source_file_count": len(rows),
    }
    (HERE / "generation_environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


LATEX_REPLACEMENTS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in LATEX_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def write_tabular(
    filename: str,
    headers: list[str],
    rows: list[list[object]],
    alignment: str,
) -> None:
    path = TABLES / filename
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def signed(value: float, digits: int = 6) -> str:
    return f"{float(value):+.{digits}f}"


def plain(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def ci_text(low: float, high: float, digits: int = 6) -> str:
    return f"[{signed(low, digits)}, {signed(high, digits)}]"


def selection_label(value: str) -> str:
    return {
        "best_screened_proto_mean": "Validation-Mean",
        "best_screened_proto_cvar05": "Validation-Tail",
    }[value]


def liability_label(value: str) -> str:
    return {"european_call": "European", "asian_call": "Asian"}[value]


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def build_dataset_assets() -> dict:
    panel = pd.read_csv(SOURCE_DATA / "historical_dataset_episode_summary.csv")
    summary = json.loads((SOURCE_DATA / "panel_summary.json").read_text())
    totals = panel[["n_episodes", "n_train_episodes", "n_val_episodes", "n_test_episodes"]].sum()
    assert int(totals["n_episodes"]) == 34395
    assert int(totals["n_train_episodes"]) == 20537
    assert int(totals["n_val_episodes"]) == 6881
    assert int(totals["n_test_episodes"]) == 6977
    assert summary["n_observations"] == 21 and summary["n_steps"] == 20

    out = panel[
        ["ticker", "n_train_episodes", "n_val_episodes", "n_test_episodes", "n_episodes"]
    ].copy()
    out.columns = ["ticker", "train", "validation", "test", "total"]
    out.loc[len(out)] = ["Total", *[int(out[c].sum()) for c in ["train", "validation", "test", "total"]]]
    out.to_csv(TABLES / "dataset_episode_counts.csv", index=False)
    rows = [
        [row.ticker, f"{int(row.train):,}", f"{int(row.validation):,}", f"{int(row.test):,}", f"{int(row.total):,}"]
        for row in out.itertuples(index=False)
    ]
    write_tabular(
        "dataset_episode_counts.tex",
        ["Ticker", "Train", "Validation", "Test", "Total"],
        rows,
        "lrrrr",
    )

    plot = out.iloc[:-1].set_index("ticker")[["train", "validation", "test"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    left = np.zeros(len(plot))
    colors = [NAVY, TEAL, ORANGE]
    for column, color in zip(plot.columns, colors):
        ax.barh(plot.index, plot[column], left=left, color=color, label=column.title())
        left += plot[column].to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("Contract-consistent episodes")
    ax.set_title("Historical panel after chronological splitting and embargo")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    save_figure(fig, "dataset_episode_counts")
    return {"total": int(totals["n_episodes"]), "test": int(totals["n_test_episodes"])}


def build_synthetic_assets() -> dict:
    bs = pd.read_csv(SOURCE_DATA / "black_scholes_summary.csv")
    original = pd.read_csv(SOURCE_DATA / "original_synthetic_summary.csv")
    sv = original[original["environment"] == "stochastic_volatility"].copy()
    paper_gap = {"Black--Scholes": -0.00023, "Stochastic volatility": -0.00030}
    records = []
    for label, frame in [("Black--Scholes", bs), ("Stochastic volatility", sv)]:
        dh = frame.loc[frame["model"] == "deep_hedging"].iloc[0]
        ph = frame.loc[frame["model"] == "protohedge"].iloc[0]
        gap = float(ph["expected_utility_frozen_training_y"] - dh["expected_utility_frozen_training_y"])
        records.append(
            {
                "environment": label,
                "deep_hedging": float(dh["expected_utility_frozen_training_y"]),
                "protohedge": float(ph["expected_utility_frozen_training_y"]),
                "proto_minus_dh": gap,
                "paper_proto_minus_dh": paper_gap[label],
                "dh_parameters": int(dh["trainable_parameters"]),
                "ph_parameters": int(ph["trainable_parameters"]),
                "test_samples": int(dh["test_samples"]),
            }
        )
    table = pd.DataFrame(records)
    assert table.loc[0, "proto_minus_dh"] < 0
    assert table.loc[1, "proto_minus_dh"] < 0
    table.to_csv(TABLES / "synthetic_reproduction.csv", index=False)
    write_tabular(
        "synthetic_reproduction.tex",
        ["Environment", "Deep Hedging", "ProtoHedge", "PH-DH", "Paper PH-DH"],
        [
            [r.environment, plain(r.deep_hedging), plain(r.protohedge), signed(r.proto_minus_dh), signed(r.paper_proto_minus_dh)]
            for r in table.itertuples(index=False)
        ],
        "lrrrr",
    )

    protocol = json.loads((SOURCE_DATA / "black_scholes_protocol.json").read_text())
    notebook = protocol["notebook_reference"]["saved_expected_utility"]
    bs_dh = table.iloc[0]["deep_hedging"]
    bs_ph = table.iloc[0]["protohedge"]
    alignment = pd.DataFrame(
        [
            {"model": "Deep Hedging", "reproduction": bs_dh, "notebook": notebook["deep_hedging"], "absolute_error": abs(bs_dh - notebook["deep_hedging"])},
            {"model": "ProtoHedge", "reproduction": bs_ph, "notebook": notebook["protohedge"], "absolute_error": abs(bs_ph - notebook["protohedge"])},
        ]
    )
    alignment.to_csv(TABLES / "black_scholes_notebook_alignment.csv", index=False)
    write_tabular(
        "black_scholes_notebook_alignment.tex",
        ["Model", "Reproduction", "Saved notebook", "Absolute error"],
        [[r.model, plain(r.reproduction), plain(r.notebook), plain(r.absolute_error)] for r in alignment.itertuples(index=False)],
        "lrrr",
    )

    prior_bs = original[original["environment"] == "black_scholes"]
    prior_dh = prior_bs.loc[prior_bs["model"] == "deep_hedging"].iloc[0]
    prior_ph = prior_bs.loc[prior_bs["model"] == "protohedge"].iloc[0]
    reconciliation = pd.DataFrame(
        [
            {
                "protocol": "Paper-specification attempt",
                "drift": 0.0,
                "prototype_asset": "regenerated K=100",
                "deep_hedging": prior_dh["expected_utility_frozen_training_y"],
                "protohedge": prior_ph["expected_utility_frozen_training_y"],
                "proto_minus_dh": prior_ph["expected_utility_frozen_training_y"]
                - prior_dh["expected_utility_frozen_training_y"],
            },
            {
                "protocol": "Notebook-faithful confirmation",
                "drift": 0.1,
                "prototype_asset": "committed K=100",
                "deep_hedging": bs_dh,
                "protohedge": bs_ph,
                "proto_minus_dh": bs_ph - bs_dh,
            },
            {
                "protocol": "Saved author notebook",
                "drift": 0.1,
                "prototype_asset": "committed K=100",
                "deep_hedging": notebook["deep_hedging"],
                "protohedge": notebook["protohedge"],
                "proto_minus_dh": notebook["protohedge"] - notebook["deep_hedging"],
            },
        ]
    )
    reconciliation.to_csv(TABLES / "black_scholes_protocol_reconciliation.csv", index=False)
    write_tabular(
        "black_scholes_protocol_reconciliation.tex",
        ["Protocol", "Drift", "Prototype input", "DH", "PH", "PH-DH"],
        [
            [
                r.protocol,
                plain(r.drift, 1),
                r.prototype_asset,
                plain(r.deep_hedging),
                plain(r.protohedge),
                signed(r.proto_minus_dh),
            ]
            for r in reconciliation.itertuples(index=False)
        ],
        "lllrrr",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(table))
    width = 0.34
    axes[0].bar(x - width / 2, table["deep_hedging"], width, color=NAVY, label="Deep Hedging")
    axes[0].bar(x + width / 2, table["protohedge"], width, color=TEAL, label="ProtoHedge")
    axes[0].set_xticks(x, ["Black--Scholes", "Stochastic\nvolatility"])
    axes[0].set_ylabel("Frozen-threshold OCE CVaR@50% utility")
    axes[0].set_title("Original-source synthetic confirmation")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color=LIGHT)
    gap_scale = 1000.0
    axes[1].axvline(0, color=GRAY, linewidth=1)
    y = np.arange(len(table))
    axes[1].scatter(gap_scale * table["proto_minus_dh"], y, color=TEAL, s=45, label="Reproduction", zorder=3)
    axes[1].scatter(gap_scale * table["paper_proto_minus_dh"], y, color=ORANGE, marker="x", s=55, label="Paper", zorder=3)
    axes[1].set_yticks(y, ["Black--Scholes", "Stoch. volatility"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"PH-DH utility (units of $10^{-3}$)")
    axes[1].set_title("Small negative gaps favor DH")
    axes[1].grid(axis="x", color=LIGHT)
    axes[1].legend(frameon=False)
    save_figure(fig, "synthetic_reproduction")
    return {
        "black_scholes_gap": float(table.iloc[0]["proto_minus_dh"]),
        "stochastic_volatility_gap": float(table.iloc[1]["proto_minus_dh"]),
        "parameter_ratio": float(table.iloc[0]["dh_parameters"] / table.iloc[0]["ph_parameters"]),
    }


def panel_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame["level"] == "panel"].copy()


def build_primary_panel_assets() -> dict:
    audited = panel_rows(SOURCE_DATA / "checkpoint_sensitivity_bootstrap.csv")
    historical = panel_rows(SOURCE_DATA / "historical_original_cvar50_bootstrap.csv")
    assert len(audited) == 4 and len(historical) == 4

    audited["liability_label"] = audited["liability"].map(liability_label)
    audited["selection_label"] = audited["selection"].map(selection_label)
    audited["supported"] = (audited["ci_low"] > 0) | (audited["ci_high"] < 0)
    order = {("european_call", "best_screened_proto_mean"): 0, ("european_call", "best_screened_proto_cvar05"): 1, ("asian_call", "best_screened_proto_mean"): 2, ("asian_call", "best_screened_proto_cvar05"): 3}
    audited["order"] = [order[(r.liability, r.selection)] for r in audited.itertuples()]
    audited = audited.sort_values("order")
    audited.to_csv(TABLES / "audited_primary_cvar50.csv", index=False)
    write_tabular(
        "audited_primary_cvar50.tex",
        ["Liability", "Selection", "PH-DH", "95\\% block interval", "P(PH>DH)"],
        [
            [r.liability_label, r.selection_label, signed(r.estimate), ci_text(r.ci_low, r.ci_high), plain(r.bootstrap_probability_gt_zero, 3)]
            for r in audited.itertuples(index=False)
        ],
        "llrrr",
    )

    key = ["liability", "selection"]
    old_cols = key + ["estimate", "ci_low", "ci_high"]
    old = historical[old_cols].rename(columns={c: f"historical_{c}" for c in ["estimate", "ci_low", "ci_high"]})
    comparison = audited.merge(old, on=key, validate="one_to_one")
    comparison["magnitude_reduction_pct"] = 100 * (
        1 - comparison["estimate"].abs() / comparison["historical_estimate"].abs()
    )
    comparison.to_csv(TABLES / "checkpoint_sensitivity_comparison.csv", index=False)
    write_tabular(
        "checkpoint_sensitivity_comparison.tex",
        ["Liability", "Selection", "Executed protocol", "Targeted sensitivity", "Absolute-gap reduction"],
        [
            [r.liability_label, r.selection_label, signed(r.historical_estimate), signed(r.estimate), f"{r.magnitude_reduction_pct:+.1f}%"]
            for r in comparison.itertuples(index=False)
        ],
        "llrrr",
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = [f"{r.liability_label}, {r.selection_label}" for r in audited.itertuples(index=False)]
    y = np.arange(len(audited))
    colors = [BLUE, BLUE, TEAL, TEAL]
    ax.axvline(0, color=GRAY, linewidth=1)
    for i, row in enumerate(audited.itertuples(index=False)):
        ax.errorbar(
            row.estimate,
            i,
            xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
            fmt="o",
            color=colors[i],
            capsize=3,
            markersize=6,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("ProtoHedge minus Deep Hedging OCE CVaR@50% utility")
    ax.set_title("Targeted checkpoint-sensitivity panel effects")
    ax.grid(axis="x", color=LIGHT)
    save_figure(fig, "audited_primary_cvar50")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    offsets = [-0.10, 0.10]
    for label, prefix, color, marker, offset in [
        ("Executed penalized selector", "historical", ORANGE, "s", offsets[0]),
        ("Targeted pure-OCE sensitivity", "", TEAL, "o", offsets[1]),
    ]:
        estimates = comparison[f"{prefix + '_' if prefix else ''}estimate"].to_numpy()
        lows = comparison[f"{prefix + '_' if prefix else ''}ci_low"].to_numpy()
        highs = comparison[f"{prefix + '_' if prefix else ''}ci_high"].to_numpy()
        ax.errorbar(
            estimates,
            y + offset,
            xerr=[estimates - lows, highs - estimates],
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
        )
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("ProtoHedge minus Deep Hedging OCE CVaR@50% utility")
    ax.set_title("Why checkpoint semantics change the European conclusion")
    ax.grid(axis="x", color=LIGHT)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "checkpoint_sensitivity_comparison")

    ticker = pd.read_csv(SOURCE_DATA / "checkpoint_sensitivity_bootstrap.csv")
    ticker = ticker[ticker["level"] == "ticker"].copy()
    ticker.to_csv(TABLES / "audited_per_ticker_cvar50.csv", index=False)
    for liability in ["european_call", "asian_call"]:
        subset = ticker[ticker["liability"] == liability]
        mean = subset[subset["selection"] == "best_screened_proto_mean"].set_index("ticker")
        tail = subset[subset["selection"] == "best_screened_proto_cvar05"].set_index("ticker")
        tickers = sorted(set(mean.index) & set(tail.index))
        rows = []
        for ticker_name in tickers:
            m = mean.loc[ticker_name]
            t = tail.loc[ticker_name]
            rows.append([ticker_name, signed(m.estimate), ci_text(m.ci_low, m.ci_high, 4), signed(t.estimate), ci_text(t.ci_low, t.ci_high, 4)])
        write_tabular(
            f"audited_per_ticker_{liability}.tex",
            ["Ticker", "Mean sel.", "95\\% interval", "Tail sel.", "95\\% interval"],
            rows,
            "lrrrr",
        )

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=False)
    for row_idx, liability in enumerate(["european_call", "asian_call"]):
        for col_idx, selection in enumerate(["best_screened_proto_mean", "best_screened_proto_cvar05"]):
            ax = axes[row_idx, col_idx]
            subset = ticker[(ticker["liability"] == liability) & (ticker["selection"] == selection)].sort_values("ticker")
            yy = np.arange(len(subset))
            color = BLUE if liability == "european_call" else TEAL
            ax.axvline(0, color=GRAY, linewidth=1)
            ax.errorbar(
                subset["estimate"],
                yy,
                xerr=[subset["estimate"] - subset["ci_low"], subset["ci_high"] - subset["estimate"]],
                fmt="o",
                color=color,
                capsize=2,
                markersize=4,
            )
            ax.set_yticks(yy, subset["ticker"])
            ax.invert_yaxis()
            ax.grid(axis="x", color=LIGHT)
            ax.set_title(f"{liability_label(liability)}, {selection_label(selection)}")
            ax.set_xlabel("PH-DH utility")
    fig.suptitle("Ticker-level checkpoint-audited OCE CVaR@50% effects", y=1.01)
    fig.tight_layout()
    save_figure(fig, "audited_per_ticker_cvar50")

    # The conference manuscript uses only the validation-tail comparison so
    # all ten assets fit legibly in one compact, two-panel figure.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharey=True)
    tail_selection = "best_screened_proto_cvar05"
    for ax, liability in zip(axes, ["european_call", "asian_call"]):
        subset = ticker[
            (ticker["liability"] == liability)
            & (ticker["selection"] == tail_selection)
        ].sort_values("ticker")
        yy = np.arange(len(subset))
        color = BLUE if liability == "european_call" else TEAL
        ax.axvline(0, color=GRAY, linewidth=1)
        ax.errorbar(
            subset["estimate"],
            yy,
            xerr=[
                subset["estimate"] - subset["ci_low"],
                subset["ci_high"] - subset["estimate"],
            ],
            fmt="o",
            color=color,
            capsize=2,
            markersize=4,
        )
        ax.set_yticks(yy, subset["ticker"])
        ax.invert_yaxis()
        ax.grid(axis="x", color=LIGHT)
        ax.set_title(liability_label(liability))
        ax.set_xlabel("PH-DH OCE utility")
    fig.tight_layout()
    save_figure(fig, "audited_per_ticker_tail_cvar50")
    return {
        f"{r.liability}_{r.selection}": {
            "estimate": float(r.estimate), "low": float(r.ci_low), "high": float(r.ci_high)
        }
        for r in audited.itertuples(index=False)
    }


def build_spot_delta_assets() -> None:
    spot = panel_rows(SOURCE_DATA / "historical_spot_delta_bootstrap.csv")
    keep = spot[spot["metric"].isin(["liability_offset_mean_difference", "liability_offset_cvar50_difference"])].copy()
    keep["liability_label"] = keep["liability"].map(liability_label)
    keep["selection_label"] = keep["selection"].map(selection_label)
    keep["metric_label"] = keep["metric"].map({"liability_offset_mean_difference": "Mean offset", "liability_offset_cvar50_difference": "Empirical CVaR@50%"})
    keep["supported"] = keep["ci_low"] > 0
    keep.to_csv(TABLES / "protohedge_vs_spot_delta.csv", index=False)
    keep = keep.sort_values(["liability_label", "selection_label", "metric_label"])
    write_tabular(
        "protohedge_vs_spot_delta.tex",
        ["Liability", "Selection", "Metric", "PH-Spot Delta", "95\\% block interval"],
        [[r.liability_label, r.selection_label, r.metric_label, signed(r.estimate), ci_text(r.ci_low, r.ci_high)] for r in keep.itertuples(index=False)],
        "lllrr",
    )
    labels = [f"{r.liability_label}\n{r.selection_label}, {r.metric_label}" for r in keep.itertuples(index=False)]
    y = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axvline(0, color=GRAY, linewidth=1)
    colors = [TEAL if value else GRAY for value in keep["supported"]]
    for i, (row, color) in enumerate(zip(keep.itertuples(index=False), colors)):
        ax.errorbar(row.estimate, i, xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]], fmt="o", color=color, capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("ProtoHedge minus Spot-Delta (positive favors ProtoHedge)")
    ax.set_title("Historical comparisons with the transparent Spot-Delta baseline")
    ax.grid(axis="x", color=LIGHT)
    save_figure(fig, "protohedge_vs_spot_delta")


def build_softclip_assets() -> dict:
    metrics = pd.read_csv(SOURCE_DATA / "spy_softclip_metrics.csv")
    bootstrap = pd.read_csv(SOURCE_DATA / "spy_softclip_bootstrap.csv")
    names = {"deep_hedging": "Deep Hedging", "tfp_exact": "ProtoHedge exact TFP", "legacy_approx": "ProtoHedge historical"}
    metrics["paper_label"] = metrics["model"].map(names)
    metrics.to_csv(TABLES / "spy_softclip_model_metrics.csv", index=False)
    write_tabular(
        "spy_softclip_model_metrics.tex",
        [
            "Model",
            "Epoch",
            r"OCE CVaR@50\%",
            "Mean",
            r"Empirical CVaR@50\%",
            r"CVaR@5\%",
            "Bound occ.",
        ],
        [[r.paper_label, int(r.selected_epoch), plain(r.frozen_oce_cvar50_utility), plain(r.liability_offset_mean), plain(r.empirical_cvar50), plain(r.liability_offset_cvar05), plain(r.bound_occupancy, 3)] for r in metrics.itertuples(index=False)],
        "lrrrrrr",
    )
    bootstrap["candidate_label"] = bootstrap["candidate"].map(names)
    bootstrap["benchmark_label"] = bootstrap["benchmark"].map(names)
    bootstrap.to_csv(TABLES / "spy_softclip_contrasts.csv", index=False)
    write_tabular(
        "spy_softclip_contrasts.tex",
        ["Contrast", "Difference", "95\\% block interval", "P(>0)"],
        [[f"{r.candidate_label} vs. {r.benchmark_label}", signed(r.estimate), ci_text(r.ci_low, r.ci_high), plain(r.bootstrap_probability_gt_zero, 3)] for r in bootstrap.itertuples(index=False)],
        "lrrr",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), gridspec_kw={"width_ratios": [1, 1.35]})
    colors = [NAVY, TEAL, ORANGE]
    axes[0].barh(metrics["paper_label"], metrics["frozen_oce_cvar50_utility"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("OCE CVaR@50% utility")
    axes[0].set_title("European SPY, seed 2345")
    axes[0].grid(axis="x", color=LIGHT)
    axes[1].axvline(0, color=GRAY, linewidth=1)
    y = np.arange(len(bootstrap))
    for i, row in enumerate(bootstrap.itertuples(index=False)):
        axes[1].errorbar(row.estimate, i, xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]], fmt="o", color=[TEAL, ORANGE, BLUE][i], capsize=3)
    axes[1].set_yticks(y, [f"{r.candidate_label}\nvs. {r.benchmark_label}" for r in bootstrap.itertuples(index=False)])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Paired utility difference")
    axes[1].set_title("One-seed temporal-block sensitivity")
    axes[1].grid(axis="x", color=LIGHT)
    save_figure(fig, "spy_softclip_sensitivity")
    exact = bootstrap[(bootstrap["candidate"] == "tfp_exact") & (bootstrap["benchmark"] == "deep_hedging")].iloc[0]
    return {"exact_vs_dh": float(exact["estimate"]), "low": float(exact["ci_low"]), "high": float(exact["ci_high"])}


def build_audit_and_scope_assets() -> dict:
    audit = pd.read_csv(SOURCE_DATA / "historical_checkpoint_audit_summary.csv")
    audit["liability_label"] = audit["liability"].map({"ALL": "All", "european_call": "European", "asian_call": "Asian"})
    audit.to_csv(TABLES / "deep_hedging_checkpoint_audit.csv", index=False)
    write_tabular(
        "deep_hedging_checkpoint_audit.tex",
        ["Liability", "Fits", "Epoch -1", "Fraction", "Median trained epoch", "Test bound occ.", "Paths touching bound"],
        [[r.liability_label, int(r.n_seed_fits), int(r.n_initial_untrained_selected), f"{100*r.fraction_initial_untrained_selected:.1f}%", plain(r.median_selected_epoch_trained_only, 1), f"{100*r.mean_test_bound_occupancy:.1f}%", f"{100*r.mean_test_path_bound_touch:.1f}%"] for r in audit.itertuples(index=False)],
        "lrrrrrr",
    )

    inventory = pd.read_csv(SOURCE_DATA / "historical_family_inventory.csv")
    grouped = inventory.groupby(["liability", "paper_faithful_family"], as_index=False).size()
    pivot = grouped.pivot(index="liability", columns="paper_faithful_family", values="size").fillna(0)
    scope_rows = []
    for liability in ["european_call", "asian_call"]:
        faithful = int(pivot.loc[liability].get(True, 0))
        expanded = int(pivot.loc[liability].get(False, 0))
        scope_rows.append({"liability": liability_label(liability), "paper_faithful": faithful, "extension": expanded, "total": faithful + expanded})
    scope = pd.DataFrame(scope_rows)
    scope.loc[len(scope)] = ["All", int(scope["paper_faithful"].sum()), int(scope["extension"].sum()), int(scope["total"].sum())]
    assert int(scope.iloc[-1]["paper_faithful"]) == 15 and int(scope.iloc[-1]["extension"]) == 25
    scope.to_csv(TABLES / "historical_family_scope.csv", index=False)
    write_tabular(
        "historical_family_scope.tex",
        ["Liability", "Released-paper-like", "Uses extension", "Total selections"],
        [[r.liability, int(r.paper_faithful), int(r.extension), int(r.total)] for r in scope.itertuples(index=False)],
        "lrrr",
    )

    plot = scope.iloc[:-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(plot["liability"], plot["paper_faithful"], color=NAVY, label="Released-paper-like")
    ax.bar(plot["liability"], plot["extension"], bottom=plot["paper_faithful"], color=ORANGE, label="Uses extension")
    ax.set_ylabel("Headline mean/tail selections")
    ax.set_title("Scope of the validation-selected historical ProtoHedge family")
    ax.set_ylim(0, 22)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=LIGHT)
    save_figure(fig, "historical_family_scope")
    return {"initial_dh": int(audit.iloc[0]["n_initial_untrained_selected"]), "expanded": 25, "faithful": 15}


def build_descriptive_and_interpretability_assets() -> None:
    descriptive = pd.read_csv(SOURCE_DATA / "historical_descriptive_panel_averages.csv")
    descriptive.to_csv(TABLES / "historical_protocol_descriptive_metrics.csv", index=False)
    rows = []
    for row in descriptive.itertuples(index=False):
        rows.append([row.liability_label, row.paper_label, plain(row.liability_offset_mean_avg), plain(row.liability_offset_cvar05_avg), plain(row.liability_offset_rmse_avg), f"{100*row.pct_at_any_position_bound_avg:.1f}%"])
    write_tabular(
        "historical_protocol_descriptive_metrics.tex",
        ["Liability", "Model", "Mean offset", r"CVaR@5\%", "RMSE", "Bound occ."],
        rows,
        "llrrrr",
    )

    records = []
    for label, filename in [("European", "qqq_european_effective_size.csv"), ("Asian", "qqq_asian_effective_size.csv")]:
        row = pd.read_csv(SOURCE_DATA / filename).iloc[0]
        records.append({"liability": label, "nominal_k": int(row["nominal_k"]), "top1_mass": row["top1_mass_mean"], "top3_mass": row["top3_mass_mean"], "top5_mass": row["top5_mass_mean"], "top10_mass": row["top10_mass_mean"], "entropy_effective_k": row["entropy_effective_k_mean"], "inverse_concentration_effective_k": row["inverse_concentration_effective_k_mean"]})
    effective = pd.DataFrame(records)
    effective.to_csv(TABLES / "qqq_interpretability_summary.csv", index=False)
    write_tabular(
        "qqq_interpretability_summary.tex",
        ["Liability", "Nominal K", "Top-1 mass", "Top-3 mass", "Top-5 mass", "Entropy effective K"],
        [[r.liability, int(r.nominal_k), plain(r.top1_mass, 3), plain(r.top3_mass, 3), plain(r.top5_mass, 3), plain(r.entropy_effective_k, 2)] for r in effective.itertuples(index=False)],
        "lrrrrr",
    )

    regime_rows = []
    regime_type_labels = {
        "return_regime": "Return",
        "vol_regime": "Volatility",
        "drawdown_regime": "Drawdown",
    }
    regime_labels = {
        "down": "Down",
        "middle": "Middle",
        "up": "Up",
        "low_vol": "Low",
        "mid_vol": "Middle",
        "high_vol": "High",
        "mild_dd": "Mild",
        "mid_dd": "Middle",
        "deep_dd": "Deep",
    }
    for liability, filename in [
        ("European", "qqq_european_regime_prototypes.csv"),
        ("Asian", "qqq_asian_regime_prototypes.csv"),
    ]:
        frame = pd.read_csv(SOURCE_DATA / filename)
        for row in frame.itertuples(index=False):
            regime_rows.append(
                [
                    liability,
                    regime_type_labels[row.regime_type],
                    regime_labels[row.regime],
                    str(row.prototype).replace("prototype_", "P"),
                    plain(row.mean_weight, 3),
                ]
            )
    write_tabular(
        "qqq_dominant_prototype_by_regime.tex",
        ["Liability", "Partition", "Regime", "Dominant prototype", "Mean weight"],
        regime_rows,
        "llllr",
    )

    nearest_rows = []
    for liability, filename in [
        ("European", "qqq_european_nearest_states.csv"),
        ("Asian", "qqq_asian_nearest_states.csv"),
    ]:
        frame = pd.read_csv(SOURCE_DATA / filename)
        frame = frame[frame["nearest_rank"] == 1].sort_values("prototype_index")
        for row in frame.itertuples(index=False):
            asian_state = getattr(row, "asian_moneyness", np.nan)
            nearest_rows.append(
                [
                    liability,
                    f"P{int(row.prototype_index)}",
                    f"{row.start_date} to {row.end_date}",
                    int(row.step_index),
                    plain(row.delta_0, 3),
                    plain(row.delta_1, 3),
                    plain(row.price_0, 3),
                    plain(row.price_1, 3),
                    "--" if pd.isna(asian_state) else plain(asian_state, 3),
                ]
            )
    write_tabular(
        "qqq_nearest_historical_states.tex",
        ["Liability", "Proto.", "Episode", "Step", "$h^S$", "$h^C$", "$S/S_0$", "$C/S_0$", "Asian state"],
        nearest_rows,
        "lllrrrrrr",
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for label, filename, color in [("European QQQ", "qqq_european_concentration.csv", BLUE), ("Asian QQQ", "qqq_asian_concentration.csv", TEAL)]:
        concentration = pd.read_csv(SOURCE_DATA / filename).sort_values("k")
        ax.plot(concentration["k"], concentration["cum_usage_mean"], marker="o", markersize=3, linewidth=1.8, label=label, color=color)
    ax.axhline(0.8, color=GRAY, linestyle="--", linewidth=0.9)
    ax.set_xlabel("Number of most-used prototypes")
    ax.set_ylabel("Cumulative assignment mass")
    ax.set_ylim(0, 1.04)
    ax.set_title("Representative QQQ prototype concentration")
    ax.grid(color=LIGHT)
    ax.legend(frameon=False)
    save_figure(fig, "qqq_prototype_concentration")

    # A compact, paper-facing example of the exact state-to-action information
    # exposed by the prototype bottleneck.
    examples = (
        pd.read_csv(SOURCE_DATA / "qqq_european_concentration.csv")
        .sort_values("usage_mean", ascending=False)
        .head(2)
    )
    fig, ax = plt.subplots(figsize=(7.15, 1.70))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.975,
        r"QQQ European PH ($K=10$)",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=NAVY,
    )
    card_colors = [TEAL, BLUE]
    for i, row in enumerate(examples.itertuples(index=False)):
        x = 0.018 + 0.493 * i
        color = card_colors[i]
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.045),
                0.471,
                0.790,
                boxstyle="round,pad=0.008,rounding_size=0.025",
                facecolor="#F7F9FA",
                edgecolor=color,
                linewidth=1.15,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.685),
                0.471,
                0.150,
                boxstyle="round,pad=0.008,rounding_size=0.025",
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
            )
        )
        ax.text(
            x + 0.025,
            0.760,
            f"P{int(row.prototype_index)}",
            color="white",
            fontsize=8.7,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x + 0.448,
            0.760,
            f"avg. weight {100 * row.usage_mean:.1f}%",
            color="white",
            fontsize=6.8,
            ha="right",
            va="center",
        )
        ax.text(x + 0.025, 0.610, "Stored state", fontsize=7.5, fontweight="bold")
        ax.text(
            x + 0.025,
            0.460,
            rf"$S/S_0={row.price_0:.3f},\quad C/S_0={row.price_1:.3f}$",
            fontsize=7.2,
        )
        ax.text(
            x + 0.025,
            0.315,
            rf"$h^S={row.delta_0:+.2f},\quad h^C={row.delta_1:+.2f}$",
            fontsize=7.2,
        )
        ax.text(
            x + 0.025,
            0.165,
            rf"time left $={row.time_left:.2f}$",
            fontsize=7.2,
        )
        ax.plot([x + 0.305, x + 0.305], [0.155, 0.625], color=LIGHT, linewidth=1.0)
        ax.text(x + 0.325, 0.610, "Attached trade", fontsize=7.5, fontweight="bold")
        ax.text(
            x + 0.325,
            0.445,
            rf"$\Delta h^S={row.proto_action_0:+.3f}$",
            fontsize=7.2,
        )
        ax.text(
            x + 0.325,
            0.270,
            rf"$\Delta h^C={row.proto_action_1:+.3f}$",
            fontsize=7.2,
        )
    save_figure(fig, "qqq_prototype_action_example")

    shutil.copy2(SOURCE_DATA / "qqq_european_nearest_episodes.png", FIGURES / "qqq_european_nearest_episodes.png")
    shutil.copy2(SOURCE_DATA / "qqq_asian_nearest_episodes.png", FIGURES / "qqq_asian_nearest_episodes.png")


def build_protocol_table() -> None:
    protocol = json.loads((SOURCE_DATA / "historical_locked_protocol.json").read_text())
    panel = json.loads((SOURCE_DATA / "panel_summary.json").read_text())
    rows = [
        ["Underlyings", ", ".join(protocol["tickers"])],
        ["Liabilities", "European and arithmetic-average Asian ATM short calls"],
        ["Episode timing", f"{protocol['n_decisions']} decisions, {protocol['n_observations']} observations"],
        ["Train / validation / test", "2005--2018 / 2019--2021 / 2022--2024"],
        ["Boundary embargo", f"{panel['embargo_steps']} observations"],
        ["Training seeds", ", ".join(str(v) for v in protocol["seeds"])],
        ["Epochs / learning rate", f"{protocol['epochs']} / {protocol['learning_rate']}"],
        ["DH architecture", "width 20, depth 3, Softplus"],
        ["Prototype counts", ", ".join(str(v) for v in protocol["prototype_counts"])],
        ["Prototype sources", "Deep Hedging and Spot-Delta"],
        ["Similarity variants", "unweighted and hand-weighted"],
        ["Trade / position bounds", "[-1,1] per instrument"],
        ["Bootstrap", f"{protocol['bootstrap_repetitions']} paired circular-block draws; block {protocol['bootstrap_block_length']}"],
    ]
    with (TABLES / "historical_protocol.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "value"])
        writer.writerows(rows)
    write_tabular("historical_protocol.tex", ["Component", "Locked value"], rows, "lp{0.67\\columnwidth}")


def build_study_design_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.set_xlim(0, 10.25)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.10, "Original-source\nsynthetic check", NAVY),
        (2.15, "Contract-consistent\nhistorical panel", BLUE),
        (4.20, "Validation-only\nProtoHedge search", ORANGE),
        (6.25, "Held-out historical\npaired evaluation", TEAL),
        (8.30, "Checkpoint/SoftClip\nsensitivity audits", RED),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch((x, 1.0), 1.65, 1.0, boxstyle="round,pad=0.08,rounding_size=0.08", linewidth=1.2, edgecolor=color, facecolor="white")
        ax.add_patch(patch)
        ax.text(x + 0.825, 1.5, label, ha="center", va="center", fontsize=7.5, color=color, fontweight="bold")
    for i in range(len(boxes) - 1):
        start = boxes[i][0] + 1.67
        end = boxes[i + 1][0] - 0.03
        ax.add_patch(FancyArrowPatch((start, 1.5), (end, 1.5), arrowstyle="-|>", mutation_scale=10, color=GRAY, linewidth=1.1))
    ax.text(5, 2.55, "From implementation fidelity to audited out-of-sample evidence", ha="center", fontsize=11, color=NAVY, fontweight="bold")
    save_figure(fig, "study_design")


def write_results_ledger(dataset: dict, synthetic: dict, primary: dict, softclip: dict, audit: dict) -> None:
    euro_tail = primary["european_call_best_screened_proto_cvar05"]
    euro_mean = primary["european_call_best_screened_proto_mean"]
    asian_tail = primary["asian_call_best_screened_proto_cvar05"]
    asian_mean = primary["asian_call_best_screened_proto_mean"]
    text = f"""# Locked Results Ledger

Generated from checksummed inputs by `generate_assets.py`. Higher utility is better.

## Dataset

- {dataset['total']:,} contract-consistent episodes across ten underlyings.
- {dataset['test']:,} test episodes in 2022--2024.
- 20 trading decisions and 21 observations per episode.

## Original-source synthetic confirmation

- Black--Scholes ProtoHedge minus Deep Hedging: {synthetic['black_scholes_gap']:+.9f}.
- Stochastic-volatility ProtoHedge minus Deep Hedging: {synthetic['stochastic_volatility_gap']:+.9f}.
- Both reproduce the paper's direction: Deep Hedging is slightly better.
- Black--Scholes parameter ratio, DH/PH: {synthetic['parameter_ratio']:.2f}x.

## Targeted checkpoint-sensitivity historical OCE CVaR@50%

- European Validation-Mean: {euro_mean['estimate']:+.6f} [{euro_mean['low']:+.6f}, {euro_mean['high']:+.6f}].
- European Validation-Tail: {euro_tail['estimate']:+.6f} [{euro_tail['low']:+.6f}, {euro_tail['high']:+.6f}].
- Asian Validation-Mean: {asian_mean['estimate']:+.6f} [{asian_mean['low']:+.6f}, {asian_mean['high']:+.6f}].
- Asian Validation-Tail: {asian_tail['estimate']:+.6f} [{asian_tail['low']:+.6f}, {asian_tail['high']:+.6f}].
- Only the Asian Validation-Tail interval excludes zero.

## Robustness facts

- The executed penalized selector selected initialization in {audit['initial_dh']} of 60 Deep Hedging fits.
- Of 40 historical headline mean/tail selections, {audit['faithful']} are released-paper-like and {audit['expanded']} use at least one extension.
- Exact-SoftClip ProtoHedge minus DH on one European SPY seed: {softclip['exact_vs_dh']:+.6f} [{softclip['low']:+.6f}, {softclip['high']:+.6f}].

## Claim boundary

The evidence supports synthetic near-parity, European historical near-parity, and a positive Asian tail-selected result for the validation-selected expanded ProtoHedge family. It does not support universal ProtoHedge superiority, an exact full-panel numerical port, or superiority over an equally tuned Deep Hedging family.
"""
    (HERE / "RESULTS_LEDGER.md").write_text(text)


def write_generated_checksums() -> None:
    candidates = []
    for directory in [TABLES, FIGURES]:
        candidates.extend(path for path in directory.rglob("*") if path.is_file())
    candidates.extend([HERE / "generation_environment.json", HERE / "RESULTS_LEDGER.md"])
    with (HERE / "generated_asset_checksums.sha256").open("w") as handle:
        for path in sorted(candidates):
            handle.write(f"{sha256(path)}  {path.relative_to(HERE)}\n")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    archive_sources()
    dataset = build_dataset_assets()
    synthetic = build_synthetic_assets()
    primary = build_primary_panel_assets()
    build_spot_delta_assets()
    softclip = build_softclip_assets()
    audit = build_audit_and_scope_assets()
    build_descriptive_and_interpretability_assets()
    build_protocol_table()
    build_study_design_figure()
    write_results_ledger(dataset, synthetic, primary, softclip, audit)
    write_generated_checksums()
    print(f"Generated manuscript assets in {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

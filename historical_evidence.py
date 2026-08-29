"""Freeze and audit the completed historical ProtoHedge experiment.

This module is intentionally read-only with respect to the training run.  It
derives paper-facing evidence from frozen paths/checkpoints and writes every
new table to a separate reproducibility directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

from deephedging.outcome_metrics import (  # noqa: E402
    INFERENCE_VERSION,
    OUTCOME_DEFINITION_VERSION,
    circular_block_bootstrap_indices,
    empirical_lower_tail_cvar,
)
from deephedging.real_data_sweep_torch import evaluate_spot_delta  # noqa: E402


ANALYSIS_VERSION = "historical-evidence-v1"
ORIGINAL_UTILITY = "oce-cvar50-lambda1"
DEFAULT_RUN_ROOT = Path(
    ".deephedging_real_runs/submission_rerun_scientific_v4"
)
DEFAULT_OUTPUT_DIR = Path(
    "paper/submission_rerun_scientific_v4/reproducibility"
)
DEFAULT_PANEL_ROOT = Path("Data/NEW_PANEL_DECISION_V2")
SELECTIONS = (
    "best_screened_proto_mean",
    "best_screened_proto_cvar05",
)
AUDITED_SOURCE_PATTERNS = (
    "*.py",
    "tests/*.py",
    "requirements*.txt",
    "cloud_setup.sh",
    "CLOUD_RUN.md",
    "paper/submission_rerun_scientific_v4/locked_protocol.json",
    "paper/submission_rerun_scientific_v4/asset_manifest.csv",
)


def oce_cvar50_values(values, threshold_y):
    """Return the original paper's pathwise OCE CVaR@50% utility."""
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(threshold_y, dtype=np.float64)
    if values.ndim == 1:
        return 2.0 * np.minimum(values + y, 0.0) - y
    return 2.0 * np.minimum(values + y[..., np.newaxis], 0.0) - y[..., np.newaxis]


def frozen_oce_cvar50(values, threshold_y):
    return float(np.mean(oce_cvar50_values(values, threshold_y)))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(label, base=0):
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int(base) + int.from_bytes(digest[:4], "little")


def _slugify(name):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return slug[:180] or "model"


def _json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _git(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _task_directories(run_root):
    run_root = Path(run_root)
    tasks = []
    for liability_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        for task_dir in sorted(liability_dir.glob("*_full_grid_*_epochs")):
            ticker = task_dir.name.split("_full_grid_", 1)[0]
            required = (
                task_dir / "sweep_config.json",
                task_dir / "sweep_metrics.csv",
                task_dir / "paired_test_liability_offsets.npz",
                task_dir / "paired_test_liability_offsets_metadata.csv",
            )
            if all(path.exists() for path in required):
                tasks.append(
                    {
                        "liability": liability_dir.name,
                        "ticker": ticker,
                        "path": task_dir,
                    }
                )
    if not tasks:
        raise FileNotFoundError(f"No completed tasks found under {run_root}")
    return tasks


def _load_task(task):
    path = task["path"]
    with open(path / "sweep_config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    offsets = np.load(path / "paired_test_liability_offsets.npz")
    offset_metadata = pd.read_csv(
        path / "paired_test_liability_offsets_metadata.csv"
    )
    metrics = pd.read_csv(path / "sweep_metrics.csv")
    return config, offsets, offset_metadata, metrics


def _state_path(task_dir, seed, risk_measure, model):
    return (
        Path(task_dir)
        / "model_artifacts"
        / f"seed_{int(seed)}"
        / f"risk_{risk_measure}"
        / _slugify(model)
        / "gym_state.pt"
    )


def _utility_threshold(task_dir, seed, risk_measure, model):
    state_path = _state_path(task_dir, seed, risk_measure, model)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    return float(state["objective.utility.y"].detach().cpu().item())


def _paired_metrics(candidate, benchmark, tail_probability=0.5):
    _, candidate_cvar = empirical_lower_tail_cvar(
        candidate, alpha=tail_probability
    )
    _, benchmark_cvar = empirical_lower_tail_cvar(
        benchmark, alpha=tail_probability
    )
    candidate = np.asarray(candidate, dtype=np.float64)
    benchmark = np.asarray(benchmark, dtype=np.float64)
    return {
        "liability_offset_mean_difference": float(
            candidate.mean() - benchmark.mean()
        ),
        "liability_offset_cvar50_difference": float(
            candidate_cvar - benchmark_cvar
        ),
        "liability_offset_rmse_improvement": float(
            np.sqrt(np.mean(benchmark**2))
            - np.sqrt(np.mean(candidate**2))
        ),
        "liability_offset_downside_deviation_improvement": float(
            np.sqrt(np.mean(np.minimum(benchmark, 0.0) ** 2))
            - np.sqrt(np.mean(np.minimum(candidate, 0.0) ** 2))
        ),
    }


def _seed_average_metrics(candidate, benchmark, tail_probability=0.5):
    candidate = np.asarray(candidate, dtype=np.float64)
    benchmark = np.asarray(benchmark, dtype=np.float64)
    if candidate.shape != benchmark.shape or candidate.ndim != 2:
        raise ValueError(
            "Paired seed matrices must have the same [seed, path] shape"
        )
    rows = [
        _paired_metrics(candidate[i], benchmark[i], tail_probability)
        for i in range(candidate.shape[0])
    ]
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in rows[0]
    }


def _confidence_row(metric, estimate, draws, confidence, **metadata):
    draws = np.asarray(draws, dtype=np.float64)
    alpha = 1.0 - float(confidence)
    return {
        **metadata,
        "metric": metric,
        "estimate": float(estimate),
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "bootstrap_probability_gt_zero": float(np.mean(draws > 0.0)),
        "n_bootstrap": int(draws.size),
        "confidence": float(confidence),
    }


def _bootstrap_spot_comparisons(
    groups,
    n_bootstrap,
    block_length,
    confidence,
    random_state,
):
    """Return ticker-level and equal-ticker panel comparisons to Spot Delta."""
    per_group_draws = {}
    rows = []
    for group_index, (ticker, matrices) in enumerate(sorted(groups.items())):
        candidate, benchmark = matrices
        estimates = _seed_average_metrics(candidate, benchmark)
        samples = circular_block_bootstrap_indices(
            candidate.shape[1],
            block_length,
            n_bootstrap,
            random_state=_stable_seed(ticker, random_state + group_index),
        )
        draws = {
            metric: np.empty(n_bootstrap, dtype=np.float64)
            for metric in estimates
        }
        for draw_index, indices in enumerate(samples):
            stats = _seed_average_metrics(
                candidate[:, indices], benchmark[:, indices]
            )
            for metric, value in stats.items():
                draws[metric][draw_index] = value
        per_group_draws[ticker] = draws
        for metric, estimate in estimates.items():
            rows.append(
                _confidence_row(
                    metric,
                    estimate,
                    draws[metric],
                    confidence,
                    level="ticker",
                    ticker=ticker,
                    n_tickers=1,
                )
            )

    for metric in next(iter(per_group_draws.values())):
        metric_draws = np.stack(
            [per_group_draws[ticker][metric] for ticker in sorted(groups)],
            axis=0,
        ).mean(axis=0)
        estimate = float(
            np.mean(
                [
                    _seed_average_metrics(*groups[ticker])[metric]
                    for ticker in sorted(groups)
                ]
            )
        )
        rows.append(
            _confidence_row(
                metric,
                estimate,
                metric_draws,
                confidence,
                level="panel",
                ticker="ALL_EQUAL_WEIGHT",
                n_tickers=len(groups),
            )
        )
    return rows


def _bootstrap_frozen_utility(
    groups,
    n_bootstrap,
    block_length,
    confidence,
    random_state,
):
    """Bootstrap the trained OCE utility using fixed learned thresholds."""
    rows = []
    ticker_draws = {}
    ticker_estimates = {}
    for group_index, (ticker, values) in enumerate(sorted(groups.items())):
        candidate, benchmark, candidate_y, benchmark_y = values
        difference_path = np.mean(
            oce_cvar50_values(candidate, candidate_y)
            - oce_cvar50_values(benchmark, benchmark_y),
            axis=0,
        )
        estimate = float(difference_path.mean())
        samples = circular_block_bootstrap_indices(
            difference_path.size,
            block_length,
            n_bootstrap,
            random_state=_stable_seed(ticker, random_state + group_index),
        )
        draws = difference_path[samples].mean(axis=1)
        ticker_draws[ticker] = draws
        ticker_estimates[ticker] = estimate
        rows.append(
            _confidence_row(
                "frozen_oce_cvar50_utility_difference",
                estimate,
                draws,
                confidence,
                level="ticker",
                ticker=ticker,
                n_tickers=1,
            )
        )
    panel_draws = np.stack(
        [ticker_draws[ticker] for ticker in sorted(ticker_draws)], axis=0
    ).mean(axis=0)
    rows.append(
        _confidence_row(
            "frozen_oce_cvar50_utility_difference",
            float(np.mean(list(ticker_estimates.values()))),
            panel_draws,
            confidence,
            level="panel",
            ticker="ALL_EQUAL_WEIGHT",
            n_tickers=len(ticker_draws),
        )
    )
    return rows


def _resolve_data_path(config, panel_root, ticker):
    configured = Path(config["data_path"])
    if configured.exists():
        return configured
    fallback = Path(panel_root) / "episodes" / f"{ticker}_training_paths.npy"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"Neither configured path {configured} nor fallback {fallback} exists"
    )


def build_utility_and_spot_tables(
    tasks,
    output_dir,
    panel_root,
    n_bootstrap=2000,
    block_length=20,
    confidence=0.95,
    random_state=20260828,
):
    utility_rows = []
    utility_bootstrap_rows = []
    spot_rows = []
    spot_bootstrap_rows = []
    grouped_utility = {}
    grouped_spot = {}

    for task in tasks:
        liability = task["liability"]
        ticker = task["ticker"]
        task_dir = task["path"]
        config, offsets, metadata, metrics = _load_task(task)
        seeds = [int(seed) for seed in config["seeds"]]

        vanilla_meta_path = (
            task_dir
            / "model_artifacts"
            / f"seed_{seeds[0]}"
            / "risk_cvar"
            / "vanilla"
            / "artifact_metadata.json"
        )
        with open(vanilla_meta_path, encoding="utf-8") as handle:
            vanilla_meta = json.load(handle)
        world_kwargs = dict(vanilla_meta["world_kwargs"])
        data_path = _resolve_data_path(config, panel_root, ticker)
        split_path = task_dir / "fixed_chronological_splits.npz"
        with np.load(split_path) as split_file:
            test_indices = np.asarray(split_file["test"], dtype=np.int64)
        _, spot_result, _ = evaluate_spot_delta(
            data_path, test_indices, world_kwargs
        )
        spot_offset = np.asarray(
            spot_result["liability_offset"], dtype=np.float64
        ).reshape(-1)

        for selection_index, selection in enumerate(SELECTIONS):
            selected = metadata[metadata["selection"] == selection]
            if len(selected) != 1:
                raise RuntimeError(
                    f"Expected one {selection} row in {task_dir}; found {len(selected)}"
                )
            selected = selected.iloc[0]
            candidate = np.asarray(
                offsets[str(selected["candidate_seed_key"])], dtype=np.float64
            )
            benchmark = np.asarray(
                offsets[str(selected["benchmark_seed_key"])], dtype=np.float64
            )
            if candidate.shape[0] != len(seeds):
                raise RuntimeError(
                    f"Seed count mismatch for {liability}/{ticker}/{selection}"
                )
            risk_measure = str(selected["risk_measure"])
            candidate_model = str(selected["candidate_model"])
            benchmark_model = str(selected["benchmark_model"])
            candidate_y = np.asarray(
                [
                    _utility_threshold(
                        task_dir, seed, risk_measure, candidate_model
                    )
                    for seed in seeds
                ]
            )
            benchmark_y = np.asarray(
                [
                    _utility_threshold(
                        task_dir, seed, risk_measure, benchmark_model
                    )
                    for seed in seeds
                ]
            )

            seed_utility_rows = []
            for seed_index, seed in enumerate(seeds):
                candidate_frozen = frozen_oce_cvar50(
                    candidate[seed_index], candidate_y[seed_index]
                )
                benchmark_frozen = frozen_oce_cvar50(
                    benchmark[seed_index], benchmark_y[seed_index]
                )
                _, candidate_empirical = empirical_lower_tail_cvar(
                    candidate[seed_index], alpha=0.5
                )
                _, benchmark_empirical = empirical_lower_tail_cvar(
                    benchmark[seed_index], alpha=0.5
                )
                seed_row = {
                    "liability": liability,
                    "ticker": ticker,
                    "selection": selection,
                    "seed": seed,
                    "candidate_model": candidate_model,
                    "benchmark_model": benchmark_model,
                    "candidate_threshold_y": candidate_y[seed_index],
                    "benchmark_threshold_y": benchmark_y[seed_index],
                    "candidate_frozen_oce_cvar50": candidate_frozen,
                    "benchmark_frozen_oce_cvar50": benchmark_frozen,
                    "frozen_oce_cvar50_difference": (
                        candidate_frozen - benchmark_frozen
                    ),
                    "candidate_empirical_cvar50": candidate_empirical,
                    "benchmark_empirical_cvar50": benchmark_empirical,
                    "empirical_cvar50_difference": (
                        candidate_empirical - benchmark_empirical
                    ),
                    "n_paths": candidate.shape[1],
                    "original_utility": ORIGINAL_UTILITY,
                    "outcome_definition": OUTCOME_DEFINITION_VERSION,
                }
                seed_utility_rows.append(seed_row)
                utility_rows.append(seed_row)

            aggregate = {
                key: float(np.mean([row[key] for row in seed_utility_rows]))
                for key in (
                    "candidate_frozen_oce_cvar50",
                    "benchmark_frozen_oce_cvar50",
                    "frozen_oce_cvar50_difference",
                    "candidate_empirical_cvar50",
                    "benchmark_empirical_cvar50",
                    "empirical_cvar50_difference",
                )
            }
            utility_rows.append(
                {
                    "liability": liability,
                    "ticker": ticker,
                    "selection": selection,
                    "seed": "SEED_AVERAGE",
                    "candidate_model": candidate_model,
                    "benchmark_model": benchmark_model,
                    "candidate_threshold_y": np.nan,
                    "benchmark_threshold_y": np.nan,
                    **aggregate,
                    "n_paths": candidate.shape[1],
                    "original_utility": ORIGINAL_UTILITY,
                    "outcome_definition": OUTCOME_DEFINITION_VERSION,
                }
            )
            grouped_utility.setdefault((liability, selection), {})[ticker] = (
                candidate,
                benchmark,
                candidate_y,
                benchmark_y,
            )

            repeated_spot = np.repeat(
                spot_offset[np.newaxis, :], candidate.shape[0], axis=0
            )
            spot_estimates = _seed_average_metrics(candidate, repeated_spot)
            spot_rows.append(
                {
                    "liability": liability,
                    "ticker": ticker,
                    "selection": selection,
                    "candidate_model": candidate_model,
                    "benchmark_model": "spot_delta",
                    **spot_estimates,
                    "n_paths": candidate.shape[1],
                    "n_seeds": candidate.shape[0],
                    "tail_probability": 0.5,
                    "outcome_definition": OUTCOME_DEFINITION_VERSION,
                }
            )
            grouped_spot.setdefault((liability, selection), {})[ticker] = (
                candidate,
                repeated_spot,
            )

    for group_index, ((liability, selection), groups) in enumerate(
        sorted(grouped_utility.items())
    ):
        rows = _bootstrap_frozen_utility(
            groups,
            n_bootstrap,
            block_length,
            confidence,
            random_state + 10000 * group_index,
        )
        for row in rows:
            utility_bootstrap_rows.append(
                {
                    "liability": liability,
                    "selection": selection,
                    "candidate_model": "validation-selected ProtoHedge",
                    "benchmark_model": "Deep Hedging",
                    "bootstrap_method": "paired_circular_block_seed_average",
                    "block_length": block_length,
                    "aggregation": "seed_average_then_equal_ticker",
                    "original_utility": ORIGINAL_UTILITY,
                    "inference_version": INFERENCE_VERSION,
                    **row,
                }
            )

    for group_index, ((liability, selection), groups) in enumerate(
        sorted(grouped_spot.items())
    ):
        rows = _bootstrap_spot_comparisons(
            groups,
            n_bootstrap,
            block_length,
            confidence,
            random_state + 100000 + 10000 * group_index,
        )
        for row in rows:
            spot_bootstrap_rows.append(
                {
                    "liability": liability,
                    "selection": selection,
                    "candidate_model": "validation-selected ProtoHedge",
                    "benchmark_model": "Spot Delta",
                    "bootstrap_method": "paired_circular_block_seed_average",
                    "block_length": block_length,
                    "aggregation": "seed_average_then_equal_ticker",
                    "tail_probability": 0.5,
                    "inference_version": INFERENCE_VERSION,
                    **row,
                }
            )

    output_dir = Path(output_dir)
    pd.DataFrame(utility_rows).to_csv(
        output_dir / "original_cvar50_utility_per_ticker.csv", index=False
    )
    pd.DataFrame(utility_bootstrap_rows).to_csv(
        output_dir / "original_cvar50_utility_bootstrap.csv", index=False
    )
    pd.DataFrame(spot_rows).to_csv(
        output_dir / "proto_vs_spot_delta_per_ticker.csv", index=False
    )
    pd.DataFrame(spot_bootstrap_rows).to_csv(
        output_dir / "proto_vs_spot_delta_bootstrap.csv", index=False
    )


def build_checkpoint_audit(tasks, output_dir):
    dh_rows = []
    selected_rows = []
    protocol_records = []
    for task in tasks:
        liability = task["liability"]
        ticker = task["ticker"]
        config, _, _, metrics = _load_task(task)
        protocol_records.append(config)
        test = metrics[metrics["split"].astype(str) == "test"].copy()
        dh = test[test["model"].astype(str) == "vanilla"].copy()
        for _, row in dh.iterrows():
            score = float(row["selected_score"])
            penalty = (
                float(config["selection_alpha_action_abs"])
                * float(row["selected_val_action_abs_mean"])
                + float(config["selection_alpha_delta_abs"])
                * float(row["selected_val_delta_abs_mean"])
                + float(config["selection_alpha_bound_occupancy"])
                * float(row["selected_val_pct_at_any_position_bound"])
                + float(config["selection_alpha_path_bound_touch"])
                * float(
                    row[
                        "selected_val_pct_paths_touch_any_position_bound"
                    ]
                )
            )
            epoch = int(row["selected_epoch"])
            dh_rows.append(
                {
                    "liability": liability,
                    "ticker": ticker,
                    "seed": int(row["seed"]),
                    "selected_epoch": epoch,
                    "checkpoint_state": (
                        "initial_untrained" if epoch == -1 else "trained_epoch"
                    ),
                    "selected_penalized_validation_score": score,
                    "selection_penalty_at_checkpoint": penalty,
                    "implied_selected_validation_loss": score - penalty,
                    "selected_val_action_abs_mean": row[
                        "selected_val_action_abs_mean"
                    ],
                    "selected_val_delta_abs_mean": row[
                        "selected_val_delta_abs_mean"
                    ],
                    "selected_val_bound_occupancy": row[
                        "selected_val_pct_at_any_position_bound"
                    ],
                    "selected_val_path_bound_touch": row[
                        "selected_val_pct_paths_touch_any_position_bound"
                    ],
                    "test_bound_occupancy": row[
                        "pct_at_any_position_bound"
                    ],
                    "test_path_bound_touch": row[
                        "pct_paths_touch_any_position_bound"
                    ],
                    "test_liability_offset_mean": row[
                        "liability_offset_mean"
                    ],
                    "test_liability_offset_cvar05": row[
                        "liability_offset_cvar05"
                    ],
                    "selection_metric": row["selection_metric"],
                    "epochs_available": int(config["epochs"]),
                }
            )

        selected_proto = test[
            test["model_family"].astype(str).eq("proto")
            & test["selected_for"].notna()
        ]
        for _, row in selected_proto.iterrows():
            labels = str(row["selected_for"]).split("|")
            for label in labels:
                selected_rows.append(
                    {
                        "liability": liability,
                        "ticker": ticker,
                        "seed": int(row["seed"]),
                        "model_family": "ProtoHedge",
                        "selection": label,
                        "selected_epoch": int(row["selected_epoch"]),
                        "selected_val_bound_occupancy": row[
                            "selected_val_pct_at_any_position_bound"
                        ],
                        "selected_val_path_bound_touch": row[
                            "selected_val_pct_paths_touch_any_position_bound"
                        ],
                        "test_bound_occupancy": row[
                            "pct_at_any_position_bound"
                        ],
                        "test_path_bound_touch": row[
                            "pct_paths_touch_any_position_bound"
                        ],
                    }
                )
        for row in dh_rows[-len(dh) :]:
            selected_rows.append(
                {
                    "liability": liability,
                    "ticker": ticker,
                    "seed": row["seed"],
                    "model_family": "Deep Hedging",
                    "selection": "fixed_original_architecture",
                    "selected_epoch": row["selected_epoch"],
                    "selected_val_bound_occupancy": row[
                        "selected_val_bound_occupancy"
                    ],
                    "selected_val_path_bound_touch": row[
                        "selected_val_path_bound_touch"
                    ],
                    "test_bound_occupancy": row["test_bound_occupancy"],
                    "test_path_bound_touch": row["test_path_bound_touch"],
                }
            )

    dh = pd.DataFrame(dh_rows)
    selected = pd.DataFrame(selected_rows)
    summaries = []
    for liability, frame in [
        ("ALL", dh),
        *[(name, group) for name, group in dh.groupby("liability")],
    ]:
        trained = frame[frame["selected_epoch"] >= 0]
        summaries.append(
            {
                "liability": liability,
                "n_seed_fits": len(frame),
                "n_initial_untrained_selected": int(
                    (frame["selected_epoch"] == -1).sum()
                ),
                "fraction_initial_untrained_selected": float(
                    (frame["selected_epoch"] == -1).mean()
                ),
                "n_epoch_zero_or_initial": int(
                    (frame["selected_epoch"] <= 0).sum()
                ),
                "fraction_epoch_zero_or_initial": float(
                    (frame["selected_epoch"] <= 0).mean()
                ),
                "median_selected_epoch_trained_only": (
                    float(trained["selected_epoch"].median())
                    if len(trained)
                    else np.nan
                ),
                "mean_selected_val_bound_occupancy": float(
                    frame["selected_val_bound_occupancy"].mean()
                ),
                "mean_test_bound_occupancy": float(
                    frame["test_bound_occupancy"].mean()
                ),
                "mean_test_path_bound_touch": float(
                    frame["test_path_bound_touch"].mean()
                ),
                "fraction_test_bound_occupancy_ge_0_49": float(
                    (frame["test_bound_occupancy"] >= 0.49).mean()
                ),
                "fraction_test_paths_all_touch_bound": float(
                    (frame["test_path_bound_touch"] >= 0.999).mean()
                ),
            }
        )

    selected_summary = (
        selected.groupby(["liability", "model_family", "selection"], dropna=False)
        .agg(
            n_seed_fits=("selected_epoch", "size"),
            n_initial_untrained_selected=(
                "selected_epoch",
                lambda values: int((values == -1).sum()),
            ),
            selected_epoch_median=("selected_epoch", "median"),
            selected_val_bound_occupancy_mean=(
                "selected_val_bound_occupancy",
                "mean",
            ),
            test_bound_occupancy_mean=("test_bound_occupancy", "mean"),
            test_path_bound_touch_mean=("test_path_bound_touch", "mean"),
        )
        .reset_index()
    )

    first = protocol_records[0]
    proto_candidates = (
        len(first["prototype_counts"])
        * len(first["prototype_sources"])
        * len(first["weighted_similarity_options"])
        * len(first["learn_distance_feature_weights_options"])
    )
    n_initial_dh = int((dh["selected_epoch"] == -1).sum())
    fairness = {
        "analysis_version": ANALYSIS_VERSION,
        "deep_hedging": {
            "architecture_candidates_per_seed_ticker_liability": 1,
            "architecture": {
                "width": 20,
                "depth": 3,
                "activation": "softplus",
            },
            "checkpoint_candidates": int(first["epochs"]) + 1,
            "initial_state_is_checkpoint_candidate": True,
        },
        "protohedge": {
            "hyperparameter_candidates_per_seed_ticker_liability": proto_candidates,
            "prototype_counts": first["prototype_counts"],
            "prototype_sources": first["prototype_sources"],
            "weighted_similarity_options": first[
                "weighted_similarity_options"
            ],
            "validation_selection_objectives": first[
                "validation_selection_objectives"
            ],
            "test_set_used_for_selection": False,
        },
        "spot_delta_band": {
            "validation_candidates_per_seed_ticker_liability": len(
                first["spot_delta_band_grid"]
            ),
            "band_grid": first["spot_delta_band_grid"],
            "selection_metric": first["tuned_baseline_metric"],
        },
        "shared_training_budget": {
            "epochs": int(first["epochs"]),
            "learning_rate": float(first["lr"]),
            "seeds": first["seeds"],
        },
        "audit_conclusion": (
            "Validation-only ProtoHedge tuning is leakage-safe, but the search "
            f"budget is asymmetric and {n_initial_dh} Deep Hedging fits select the untrained "
            "initial state. A locked checkpoint-selection sensitivity is required "
            "before making a definitive superiority claim."
        ),
    }

    output_dir = Path(output_dir)
    dh.to_csv(output_dir / "dh_checkpoint_audit.csv", index=False)
    pd.DataFrame(summaries).to_csv(
        output_dir / "dh_checkpoint_audit_summary.csv", index=False
    )
    selected_summary.to_csv(
        output_dir / "selected_model_checkpoint_summary.csv", index=False
    )
    _json_dump(fairness, output_dir / "benchmark_tuning_fairness.json")
    _write_audit_report(dh, pd.DataFrame(summaries), fairness, output_dir)


def _write_audit_report(dh, summaries, fairness, output_dir):
    overall = summaries[summaries["liability"] == "ALL"].iloc[0]
    european = summaries[summaries["liability"] == "european_call"].iloc[0]
    asian = summaries[summaries["liability"] == "asian_call"].iloc[0]
    initial = dh[dh["selected_epoch"] == -1][
        ["liability", "ticker", "seed"]
    ]
    initial_text = ", ".join(
        f"{row.liability}/{row.ticker}/seed-{row.seed}"
        for row in initial.itertuples(index=False)
    )
    report = f"""# Historical checkpoint and fairness audit

Analysis version: `{ANALYSIS_VERSION}`.

## Meaning of `selected_epoch = -1`

The trainer evaluates the initialized model before epoch 0 and stores it as the
initial best checkpoint.  A value of `-1` therefore means that none of the 800
trained epochs achieved a lower penalized validation-selection score.  The
artifact is valid and loadable, but its policy weights are untrained random
initialization weights; `-1` is not a missing epoch and not an 801st epoch.

## Findings

- Deep Hedging selected the initialized checkpoint in
  {int(overall.n_initial_untrained_selected)}/{int(overall.n_seed_fits)} fits
  ({overall.fraction_initial_untrained_selected:.1%}).
- The concentration is {int(european.n_initial_untrained_selected)}/
  {int(european.n_seed_fits)} European fits versus
  {int(asian.n_initial_untrained_selected)}/{int(asian.n_seed_fits)} Asian fits.
- Mean DH test position-bound occupancy is
  {european.mean_test_bound_occupancy:.1%} for European and
  {asian.mean_test_bound_occupancy:.1%} for Asian liabilities.  Mean fractions
  of paths touching a bound are {european.mean_test_path_bound_touch:.1%} and
  {asian.mean_test_path_bound_touch:.1%}, respectively.
- Affected fits: {initial_text}.

The implied unpenalized validation loss in `dh_checkpoint_audit.csv` is
reconstructed as selected score minus the four configured diagnostic penalties.
The full epoch histories were not serialized into model artifacts, so the old
run cannot be retrospectively reselected under a new rule.

## Benchmark-tuning fairness

Deep Hedging used one fixed paper architecture per seed.  ProtoHedge evaluated
{fairness['protohedge']['hyperparameter_candidates_per_seed_ticker_liability']}
validation-only configurations per seed and selected separate mean- and
tail-oriented configurations.  This is test-leakage safe, but it is an
asymmetric model-development budget that must be disclosed and stress-tested.

## Decision

Do not discard the completed run.  It remains valid evidence for the protocol
that was actually executed.  Before a definitive claim that ProtoHedge beats
Deep Hedging, run a newly named, locked sensitivity in which checkpoint
selection uses the unpenalized validation OCE loss and saturation is reported as
a diagnostic rather than folded into the checkpoint score.  Preserve the
original architecture, optimizer, seeds, splits, bounds, and test set.  First
run the affected European DH fits plus matched selected ProtoHedge fits; expand
to the complete comparison if the conclusion changes materially.
"""
    with open(Path(output_dir) / "audit_report.md", "w", encoding="utf-8") as handle:
        handle.write(report)


def _write_checksum_manifest(root, destination, relative_to=None):
    root = Path(root)
    destination = Path(destination)
    relative_to = Path(relative_to or root)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != destination.resolve()
    )
    lines = []
    total_bytes = 0
    for path in files:
        total_bytes += path.stat().st_size
        lines.append(f"{_sha256(path)}  {path.relative_to(relative_to)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"file_count": len(files), "total_bytes": total_bytes}


def _source_files(repo_root):
    files = set()
    for pattern in AUDITED_SOURCE_PATTERNS:
        files.update(path for path in repo_root.glob(pattern) if path.is_file())
    return sorted(files)


def _write_source_checksum_manifest(repo_root, destination):
    lines = [
        f"{_sha256(path)}  {path.relative_to(repo_root)}"
        for path in _source_files(repo_root)
    ]
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _package_versions():
    versions = {}
    for package in (
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "cdxbasics",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def freeze_archive(tasks, run_root, panel_root, output_dir, arguments):
    repo_root = Path(__file__).resolve().parent
    output_dir = Path(output_dir)
    raw_stats = _write_checksum_manifest(
        run_root, output_dir / "raw_output_checksums.sha256"
    )
    input_stats = _write_checksum_manifest(
        panel_root, output_dir / "panel_input_checksums.sha256"
    )
    source_count = _write_source_checksum_manifest(
        repo_root, output_dir / "source_snapshot_checksums.sha256"
    )
    status = _git("status", "--porcelain=v1") or ""
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "archive_created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "archive_operation_was_read_only_for_raw_outputs": True,
        "raw_output_checksums_are_integrity_baseline": True,
        "run_root": str(Path(run_root).resolve()),
        "panel_root": str(Path(panel_root).resolve()),
        "task_count": len(tasks),
        "liabilities": sorted({task["liability"] for task in tasks}),
        "tickers": sorted({task["ticker"] for task in tasks}),
        "raw_output_manifest": raw_stats,
        "panel_input_manifest": input_stats,
        "source_file_count": source_count,
        "analysis_arguments": arguments,
        "git": {
            "head_at_archive_time": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "worktree_dirty": bool(status),
            "porcelain_status": status.splitlines(),
            "training_commit_embedded_in_historical_artifacts": False,
            "provenance_note": (
                "The completed artifacts did not embed a Git commit. The HEAD "
                "and source checksums recorded here describe the archive-time "
                "workspace; regenerate this manifest after the planned commit."
            ),
        },
        "archive_environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "training_environment": {
            "embedded_in_artifacts": False,
            "locked_requirements_file": "requirements-cloud.txt",
            "note": (
                "Package versions were not serialized by the historical runner. "
                "Do not relabel archive-time package versions as training versions."
            ),
        },
    }
    _json_dump(manifest, output_dir / "archive_manifest.json")

    readme = f"""# Completed historical-run evidence archive

This directory was generated by `{Path(__file__).name}` using analysis version
`{ANALYSIS_VERSION}`.  It does not modify the completed run.

- `raw_output_checksums.sha256`: every frozen training artifact and result.
- `panel_input_checksums.sha256`: every historical panel input.
- `source_snapshot_checksums.sha256`: archive-time analysis/training source.
- `archive_manifest.json`: configuration, Git context, and environment record.
- `original_cvar50_utility_*.csv`: the original paper's OCE CVaR@50% metric.
- `proto_vs_spot_delta_*.csv`: paired CVaR@50%, mean, RMSE, and downside tests.
- `dh_checkpoint_audit*.csv`: exact checkpoint/saturation diagnostics.
- `benchmark_tuning_fairness.json`: comparative search and selection budgets.
- `audit_report.md`: interpretation and locked sensitivity decision.

The historical runner did not embed its Git commit or complete environment, so
the manifest records that provenance gap rather than inferring false precision.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    _write_checksum_manifest(
        output_dir,
        output_dir / "derived_evidence_checksums.sha256",
        relative_to=output_dir,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-block-length", type=int, default=20)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=20260828)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _task_directories(args.run_root)
    build_utility_and_spot_tables(
        tasks,
        args.output_dir,
        args.panel_root,
        n_bootstrap=args.bootstrap_repetitions,
        block_length=args.bootstrap_block_length,
        confidence=args.bootstrap_confidence,
        random_state=args.random_state,
    )
    build_checkpoint_audit(tasks, args.output_dir)
    freeze_archive(
        tasks,
        args.run_root,
        args.panel_root,
        args.output_dir,
        arguments={
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_block_length": args.bootstrap_block_length,
            "bootstrap_confidence": args.bootstrap_confidence,
            "random_state": args.random_state,
        },
    )
    print(
        f"Archived {len(tasks)} completed tasks to {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

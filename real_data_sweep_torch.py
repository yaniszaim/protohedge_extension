"""
Real-data ProtoHedge experiment sweep.

This module runs the empirical-test grid we need after the single notebook
pilot: unhedged, spot-delta, vanilla Deep Hedging, and multiple ProtoHedge
variants under the same normalized/bounded real-data environment.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if __package__ in [None, ""]:
    import sys

    package_root = Path(__file__).resolve().parent
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

from deephedging.base_torch import torchCast
from deephedging.panel_data_pipeline import (
    EPISODE_TIMING_VERSION,
    OPTION_PATH_VERSION,
    TEMPORAL_SPLIT_VERSION,
    load_chronological_splits,
)
from deephedging.real_data_analysis_torch import (
    load_model_artifact,
    save_model_artifact,
)
from deephedging.prototype_extraction_torch import build_prototype_payload, save_prototype_payload
from deephedging.run_train_torch import run_experiment
from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch
from deephedging.hedge_accounting import HEDGE_ACCOUNTING_VERSION, uses_step_returns
from deephedging.payoff_state import (
    ASIAN_PAYOFF_FEATURE,
    ASIAN_PAYOFF_STATE_VERSION,
    BASE_MODEL_FEATURES,
    model_features_for_liability,
    payoff_state_version_for_liability,
    validate_model_features_for_liability,
)
from deephedging.outcome_metrics import (
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    INFERENCE_VERSION,
    OUTCOME_DEFINITION_VERSION,
    PREMIUM_INCLUDED,
    paired_seed_circular_block_bootstrap,
    summarize_liability_offset,
)


# Backward-compatible alias for the European/synthetic state.
MODEL_FEATURES = list(BASE_MODEL_FEATURES)
DEFAULT_SPOT_DELTA_BAND_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15)
MODEL_SELECTION_VERSION = "validation-only-across-seeds-v1"
PROTO_CONFIG_COLUMNS = [
    "model",
    "risk_measure",
    "prototype_source",
    "n_prototypes",
    "weighted_similarity",
    "learn_distance_feature_weights",
]
VALIDATION_SELECTIONS = {
    "best_screened_proto_mean": "liability_offset_mean_avg",
    "best_screened_proto_cvar05": "liability_offset_cvar05_avg",
}

_TRAINING_HISTORY_FIELDS = {
    "selected_epoch": "best_epoch",
    "selected_score": "best_score",
    "selection_metric": "selection_metric",
    "selected_val_action_abs_mean": "best_val_action_abs_mean",
    "selected_val_delta_abs_mean": "best_val_delta_abs_mean",
    "selected_val_pct_at_any_position_bound": (
        "best_val_pct_at_any_position_bound"
    ),
    "selected_val_pct_paths_touch_any_position_bound": (
        "best_val_pct_paths_touch_any_position_bound"
    ),
}


def _validate_resume_config(previous, expected):
    mismatches = []
    for key, expected_value in expected.items():
        previous_value = previous.get(key, "<missing>")
        if _jsonify(previous_value) != _jsonify(expected_value):
            mismatches.append(
                f"{key}: saved={previous_value!r}, requested={expected_value!r}"
            )
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Refusing to resume a sweep with a different experiment configuration. "
            "Use a new output directory or pass resume=False.\n  - " + details
        )


def _validate_resume_metric_keys(metrics):
    required = {"model", "split", "seed", "risk_measure", "artifact_dir"}
    missing = required.difference(metrics.columns)
    if missing:
        raise RuntimeError(
            "Saved sweep metrics cannot be resumed because columns are missing: "
            + ", ".join(sorted(missing))
        )
    key_columns = ["model", "split", "seed", "risk_measure"]
    duplicates = metrics.duplicated(key_columns, keep=False)
    if duplicates.any():
        duplicate_keys = metrics.loc[duplicates, key_columns].to_dict("records")
        raise RuntimeError(
            "Saved sweep metrics contain duplicate model/split checkpoints: "
            f"{duplicate_keys[:5]}"
        )


def _completed_baseline_checkpoint(metrics, seed):
    if metrics.empty:
        return False
    expected = {
        (model, split)
        for model in ("unhedged", "spot_delta", "spot_delta_band")
        for split in ("train", "val")
    }
    rows = metrics[
        metrics["seed"].astype(int).eq(int(seed))
        & metrics["model"].isin([model for model, _ in expected])
        & metrics["split"].isin(["train", "val"])
    ]
    if rows.empty:
        return False
    actual = set(zip(rows["model"].astype(str), rows["split"].astype(str)))
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError(
            f"Seed {seed} has a partial or inconsistent baseline checkpoint; "
            "restore the pre-run backup before resuming."
        )
    return True


def _training_meta_from_artifact(metadata):
    history = metadata.get("history", {})
    return {
        output_key: history.get(history_key)
        for output_key, history_key in _TRAINING_HISTORY_FIELDS.items()
    }


def _completed_artifact_checkpoint(metrics, model, seed, risk_measure):
    if metrics.empty:
        return None
    rows = metrics[
        metrics["model"].astype(str).eq(str(model))
        & metrics["seed"].astype(int).eq(int(seed))
        & metrics["risk_measure"].astype(str).eq(str(risk_measure))
        & metrics["split"].isin(["train", "val"])
    ]
    if rows.empty:
        return None
    if len(rows) != 2 or set(rows["split"].astype(str)) != {"train", "val"}:
        raise RuntimeError(
            f"Model {model!r}, seed {seed}, risk {risk_measure!r} has a partial "
            "metric checkpoint; restore the pre-run backup before resuming."
        )

    artifact_dirs = rows["artifact_dir"].dropna().astype(str).unique()
    if len(artifact_dirs) != 1:
        raise RuntimeError(
            f"Model {model!r}, seed {seed}, risk {risk_measure!r} does not point "
            "to exactly one saved artifact."
        )
    artifact_dir = Path(artifact_dirs[0])
    metadata_path = artifact_dir / "artifact_metadata.json"
    state_path = artifact_dir / "gym_state.pt"
    if not metadata_path.is_file() or not state_path.is_file():
        raise RuntimeError(
            f"Checkpoint rows exist for {model!r}, seed {seed}, but its model "
            f"artifact is incomplete at {artifact_dir}."
        )
    if metadata_path.stat().st_size == 0 or state_path.stat().st_size == 0:
        raise RuntimeError(
            f"Checkpoint artifact for {model!r}, seed {seed} is empty at "
            f"{artifact_dir}."
        )
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    identity = {
        "model_name": str(model),
        "seed": int(seed),
        "risk_measure": str(risk_measure),
    }
    for key, expected_value in identity.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(
                f"Artifact identity mismatch for {artifact_dir}: expected "
                f"{key}={expected_value!r}, found {metadata.get(key)!r}."
            )
    return rows.copy(), artifact_dir, metadata


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _jsonify(x):
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _normalize_trade_bounds(bounds):
    if bounds is None:
        return {
            "lbnd_as": -5.0,
            "ubnd_as": 5.0,
            "lbnd_av": -5.0,
            "ubnd_av": 5.0,
        }
    if isinstance(bounds, dict):
        return bounds
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        lbnd, ubnd = float(bounds[0]), float(bounds[1])
        return {
            "lbnd_as": lbnd,
            "ubnd_as": ubnd,
            "lbnd_av": lbnd,
            "ubnd_av": ubnd,
        }
    raise TypeError(f"Unsupported trade_bounds format: {type(bounds)!r}")


def _normalize_cumulative_bounds(bounds):
    if bounds is None:
        return {
            "lbnd_delta_s": -2.0,
            "ubnd_delta_s": 2.0,
            "lbnd_delta_v": -2.0,
            "ubnd_delta_v": 2.0,
        }
    if isinstance(bounds, dict):
        return bounds
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        lbnd, ubnd = float(bounds[0]), float(bounds[1])
        return {
            "lbnd_delta_s": lbnd,
            "ubnd_delta_s": ubnd,
            "lbnd_delta_v": lbnd,
            "ubnd_delta_v": ubnd,
        }
    raise TypeError(f"Unsupported cumulative_bounds format: {type(bounds)!r}")


def split_indices(*args, **kwargs):
    raise RuntimeError(
        "Random splitting of historical rolling windows is disabled because it leaks "
        "overlapping observations across train, validation, and test. Load persisted "
        "chronological pre-window splits with load_chronological_splits()."
    )


def world_cfg(
    data_path,
    indices,
    val_indices=None,
    seed=1234,
    normalize=True,
    hedge_mode="step",
    position_bounds=True,
    trade_bounds=None,
    cumulative_bounds=None,
    liability_type="european_call",
    asian_average_type="arithmetic",
    asian_start_step=1,
    asian_end_step=None,
    payoff_state_version=None,
):
    trade_bounds = _normalize_trade_bounds(trade_bounds)
    cumulative_bounds = _normalize_cumulative_bounds(cumulative_bounds)
    expected_payoff_state_version = payoff_state_version_for_liability(liability_type)
    if (
        payoff_state_version not in (None, "")
        and payoff_state_version != expected_payoff_state_version
    ):
        raise ValueError(
            f"Payoff-state version {payoff_state_version!r} does not match "
            f"{expected_payoff_state_version!r} for liability {liability_type!r}"
        )
    cfg = {
        "world_type": "real",
        "data_path": str(data_path),
        "sample_indices": [int(i) for i in indices],
        "shuffle": False,
        "seed": int(seed),
        "normalize": bool(normalize),
        "hedge_mode": str(hedge_mode),
        "position_bounds": bool(position_bounds),
        "liability_type": str(liability_type),
        "asian_average_type": str(asian_average_type),
        "asian_start_step": int(asian_start_step),
        "asian_end_step": asian_end_step,
        "payoff_state_version": expected_payoff_state_version,
        **trade_bounds,
    }
    if position_bounds:
        cfg.update(cumulative_bounds)
    if val_indices is not None:
        cfg["val_sample_indices"] = [int(i) for i in val_indices]
    return cfg


def training_cfg(
    epochs=5,
    lr=1e-3,
    device="auto",
    batch_size=None,
    epoch_refresh=None,
    selection_metric="train_loss",
    selection_alpha_action_abs=0.0,
    selection_alpha_delta_abs=0.0,
    selection_alpha_bound_occupancy=0.0,
    selection_alpha_path_bound_touch=0.0,
    action_penalty_weight=0.0,
    delta_penalty_weight=0.0,
    seed=0,
):
    epoch_refresh = int(epoch_refresh) if epoch_refresh is not None else max(1, int(epochs))
    return {
        "epochs": int(epochs),
        "lr": float(lr),
        "device": str(device),
        "batch_size": batch_size,
        "epoch_refresh": max(1, epoch_refresh),
        "clipvalue": 1.0,
        "global_clipnorm": 1.0,
        "lr_decay_factor": None,
        "lr_decay_patience": None,
        "selection_metric": str(selection_metric),
        "selection_alpha_action_abs": float(selection_alpha_action_abs),
        "selection_alpha_delta_abs": float(selection_alpha_delta_abs),
        "selection_alpha_bound_occupancy": float(selection_alpha_bound_occupancy),
        "selection_alpha_path_bound_touch": float(selection_alpha_path_bound_touch),
        "action_penalty_weight": float(action_penalty_weight),
        "delta_penalty_weight": float(delta_penalty_weight),
        "seed": int(seed),
    }


def objective_cfg(risk_measure="cvar", risk_aversion=1.0):
    return {"risk_measure": str(risk_measure), "risk_aversion": float(risk_aversion)}


def vanilla_model_cfg(feature_list=None):
    return {
        "agent_type": "feed_forward",
        "feature_list": list(feature_list or MODEL_FEATURES),
        "network_width": 20,
        "network_depth": 3,
        "activation": "softplus",
    }


def proto_model_cfg(
    prototype_path,
    weighted_similarity=True,
    learn_distance_feature_weights=False,
    feature_list=None,
):
    return {
        "agent_type": "protopnet",
        "prototype_path": str(prototype_path),
        "feature_list": list(feature_list or MODEL_FEATURES),
        "weighted_similarity": bool(weighted_similarity),
        "learn_distance_feature_weights": bool(learn_distance_feature_weights),
    }


def _assert_finite(label, result):
    keys = [
        "payoff",
        "pnl",
        "cost",
        "liability_offset",
        "gains",
        "actions",
    ]
    for key in keys:
        if key not in result:
            raise AssertionError(f"{label}: missing result['{key}']")
        if not np.isfinite(_as_numpy(result[key])).all():
            raise AssertionError(f"{label}: non-finite values in result['{key}']")


def _position_bound_metrics(deltas, market, tol=1e-4):
    metrics = {
        "pct_at_any_position_bound": 0.0,
        "pct_at_upper_position_bound": 0.0,
        "pct_at_lower_position_bound": 0.0,
        "pct_paths_touch_any_position_bound": 0.0,
    }
    if market is None or "ubnd_delta" not in market or "lbnd_delta" not in market:
        return metrics

    ubnd_delta = _as_numpy(market["ubnd_delta"])
    lbnd_delta = _as_numpy(market["lbnd_delta"])
    near_upper = np.isclose(deltas, ubnd_delta, atol=tol, rtol=0.0)
    near_lower = np.isclose(deltas, lbnd_delta, atol=tol, rtol=0.0)
    near_bound = near_upper | near_lower
    metrics.update(
        {
            "pct_at_any_position_bound": float(np.mean(near_bound)),
            "pct_at_upper_position_bound": float(np.mean(near_upper)),
            "pct_at_lower_position_bound": float(np.mean(near_lower)),
            "pct_paths_touch_any_position_bound": float(np.mean(np.any(near_bound, axis=(1, 2)))),
        }
    )
    return metrics


def summarize_result(result, market=None):
    liability_offset = _as_numpy(
        result.get("liability_offset", result["gains"])
    ).reshape(-1)
    payoff = _as_numpy(result["payoff"]).reshape(-1)
    pnl = _as_numpy(result["pnl"]).reshape(-1)
    cost = _as_numpy(result["cost"]).reshape(-1)
    actions = _as_numpy(result["actions"])
    deltas = _as_numpy(result["deltas"]) if "deltas" in result else np.cumsum(actions, axis=1)

    metrics = summarize_liability_offset(liability_offset)
    metrics.update({
        "payoff_mean": float(payoff.mean()),
        "trading_gain_mean": float(pnl.mean()),
        "cost_mean": float(cost.mean()),
        "action_abs_mean": float(np.mean(np.abs(actions))),
        "action_abs_max": float(np.max(np.abs(actions))),
        "delta_abs_mean": float(np.mean(np.abs(deltas))),
        "delta_abs_max": float(np.max(np.abs(deltas))),
    })
    metrics.update(_position_bound_metrics(deltas=deltas, market=market))
    return metrics


def evaluate_gym(gym, data_path, indices, label, world_kwargs):
    world = RealWorld_Spot_ATM_Torch(
        world_cfg(data_path=data_path, indices=indices, **world_kwargs)
    )
    data = torchCast(world.torch_data, device=getattr(gym, "device", "cpu"))
    gym.eval()
    with torch.no_grad():
        result = gym.forward(data, training=False, return_paths=True)
    result = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    _assert_finite(label, result)
    metrics = summarize_result(result, market=world.data.market)
    metrics.update({"n_paths": int(world.nSamples), "n_steps": int(world.nSteps), "n_inst": int(world.nInst)})
    return world, result, metrics


def evaluate_unhedged(data_path, indices, world_kwargs):
    world = RealWorld_Spot_ATM_Torch(
        world_cfg(data_path=data_path, indices=indices, **world_kwargs)
    )
    payoff = np.asarray(world.data.market.payoff, dtype=np.float32)
    n_paths, n_steps, n_inst = world.data.market.hedges.shape
    result = {
        "payoff": payoff,
        "pnl": np.zeros_like(payoff),
        "cost": np.zeros_like(payoff),
        "liability_offset": payoff.copy(),
        "gains": payoff.copy(),
        "actions": np.zeros((n_paths, n_steps, n_inst), dtype=np.float32),
        "deltas": np.zeros((n_paths, n_steps, n_inst), dtype=np.float32),
    }
    _assert_finite("unhedged", result)
    return world, result, summarize_result(result, market=world.data.market)


def _evaluate_spot_delta_rule(data_path, indices, world_kwargs, band=0.0, label="spot_delta"):
    world = RealWorld_Spot_ATM_Torch(
        world_cfg(data_path=data_path, indices=indices, **world_kwargs)
    )
    market = world.data.market
    hedges = np.asarray(market.hedges, dtype=np.float32)
    trading_cost = np.asarray(market.cost, dtype=np.float32)
    payoff = np.asarray(market.payoff, dtype=np.float32)
    step_return_accounting = uses_step_returns(market)
    ubnd_a = np.asarray(market.ubnd_a, dtype=np.float32)
    lbnd_a = np.asarray(market.lbnd_a, dtype=np.float32)
    ubnd_delta = np.asarray(market.ubnd_delta, dtype=np.float32) if "ubnd_delta" in market else None
    lbnd_delta = np.asarray(market.lbnd_delta, dtype=np.float32) if "lbnd_delta" in market else None
    call_delta = np.asarray(world.data.features.per_step["call_delta"], dtype=np.float32)
    band = float(band)
    if band < 0.0:
        raise ValueError(f"band must be non-negative, got {band}")

    n_paths, n_steps, n_inst = hedges.shape
    delta = np.zeros((n_paths, n_inst), dtype=np.float32)
    actions = np.zeros((n_paths, n_steps, n_inst), dtype=np.float32)
    deltas = np.zeros((n_paths, n_steps, n_inst), dtype=np.float32)
    pnl = np.zeros((n_paths,), dtype=np.float32)
    cost = np.zeros((n_paths,), dtype=np.float32)

    for t in range(n_steps):
        target = np.zeros_like(delta)
        target_low = call_delta[:, t] - band
        target_high = call_delta[:, t] + band
        if ubnd_delta is not None and lbnd_delta is not None:
            target_low = np.minimum(np.maximum(target_low, lbnd_delta[:, t, 0]), ubnd_delta[:, t, 0])
            target_high = np.minimum(np.maximum(target_high, lbnd_delta[:, t, 0]), ubnd_delta[:, t, 0])
        target[:, 0] = np.minimum(np.maximum(delta[:, 0], target_low), target_high)
        action = np.minimum(np.maximum(target - delta, lbnd_a[:, t, :]), ubnd_a[:, t, :])
        if ubnd_delta is not None and lbnd_delta is not None:
            bounded_delta = np.minimum(np.maximum(delta + action, lbnd_delta[:, t, :]), ubnd_delta[:, t, :])
            action = bounded_delta - delta
        next_delta = delta + action
        pnl_position = next_delta if step_return_accounting else action
        pnl += np.sum(pnl_position * hedges[:, t, :], axis=1)
        cost += np.sum(np.abs(action) * trading_cost[:, t, :], axis=1)
        delta = next_delta
        actions[:, t, :] = action
        deltas[:, t, :] = delta

    liability_offset = payoff + pnl - cost
    result = {
        "payoff": payoff,
        "pnl": pnl,
        "cost": cost,
        "liability_offset": liability_offset,
        "gains": liability_offset,
        "actions": actions,
        "deltas": deltas,
    }
    _assert_finite(label, result)
    return world, result, summarize_result(result, market=world.data.market)


def evaluate_spot_delta(data_path, indices, world_kwargs):
    return _evaluate_spot_delta_rule(
        data_path=data_path,
        indices=indices,
        world_kwargs=world_kwargs,
        band=0.0,
        label="spot_delta",
    )


def evaluate_spot_delta_band(data_path, indices, world_kwargs, band=0.05):
    return _evaluate_spot_delta_rule(
        data_path=data_path,
        indices=indices,
        world_kwargs=world_kwargs,
        band=band,
        label="spot_delta_band",
    )


def select_spot_delta_band(
    data_path,
    val_indices,
    world_kwargs,
    band_grid=DEFAULT_SPOT_DELTA_BAND_GRID,
    selection_metric="liability_offset_cvar05",
):
    rows = []
    for band in band_grid:
        _, _, metrics = evaluate_spot_delta_band(
            data_path=data_path,
            indices=val_indices,
            world_kwargs=world_kwargs,
            band=float(band),
        )
        rows.append({"band": float(band), **metrics})

    band_df = pd.DataFrame(rows)
    if band_df.empty:
        raise ValueError("band_grid produced no candidate rows")
    if selection_metric not in band_df.columns:
        raise ValueError(f"selection_metric '{selection_metric}' not in {band_df.columns.tolist()}")

    ascending = selection_metric in {
        "liability_offset_variance",
        "liability_offset_mae",
        "liability_offset_rmse",
        "liability_offset_downside_deviation",
        "negative_offset_rate",
        "cost_mean",
        "action_abs_mean",
        "delta_abs_mean",
    }
    sort_cols = [
        selection_metric,
        "liability_offset_mean",
        "pct_at_any_position_bound",
        "band",
    ]
    sort_ascending = [ascending, False, True, True]
    best = band_df.sort_values(sort_cols, ascending=sort_ascending).iloc[0]
    return {
        "best_band": float(best["band"]),
        "selection_metric": str(selection_metric),
        "candidates": band_df.sort_values(sort_cols, ascending=sort_ascending).reset_index(drop=True),
    }


def metric_row(model, split, metrics, **kwargs):
    row = {"model": model, "split": split, **kwargs}
    row.update(metrics)
    return row


def validation_proto_summary(
    metrics_df,
    expected_seeds=None,
    max_bound_occupancy=None,
    max_path_touch_rate=None,
):
    """Aggregate candidate performance using validation observations only."""
    validation = metrics_df[
        (metrics_df["split"] == "val")
        & (metrics_df.get("model_family", "") == "proto")
    ].copy()
    if validation.empty:
        return pd.DataFrame()

    group_cols = [col for col in PROTO_CONFIG_COLUMNS if col in validation]
    summary = (
        validation.groupby(group_cols, dropna=False)
        .agg(
            n_seed_runs=("seed", "nunique"),
            liability_offset_mean_avg=("liability_offset_mean", "mean"),
            liability_offset_mean_std=("liability_offset_mean", "std"),
            liability_offset_cvar05_avg=("liability_offset_cvar05", "mean"),
            liability_offset_cvar05_std=("liability_offset_cvar05", "std"),
            liability_offset_rmse_avg=("liability_offset_rmse", "mean"),
            liability_offset_downside_deviation_avg=(
                "liability_offset_downside_deviation",
                "mean",
            ),
            validation_bound_occupancy_avg=(
                "pct_at_any_position_bound",
                "mean",
            ),
            validation_path_bound_touch_avg=(
                "pct_paths_touch_any_position_bound",
                "mean",
            ),
        )
        .reset_index()
    )
    if expected_seeds is None:
        expected_seeds = int(validation["seed"].nunique())
    summary["has_all_seed_runs"] = (
        summary["n_seed_runs"] == int(expected_seeds)
    )
    if max_bound_occupancy is None:
        summary["passes_validation_bound_occupancy"] = True
    else:
        summary["passes_validation_bound_occupancy"] = (
            summary["validation_bound_occupancy_avg"]
            <= float(max_bound_occupancy)
        )
    if max_path_touch_rate is None:
        summary["passes_validation_path_touch"] = True
    else:
        summary["passes_validation_path_touch"] = (
            summary["validation_path_bound_touch_avg"]
            <= float(max_path_touch_rate)
        )
    summary["passes_validation_robust_screen"] = (
        summary["has_all_seed_runs"]
        & summary["passes_validation_bound_occupancy"]
        & summary["passes_validation_path_touch"]
    )
    summary["selection_split"] = "validation"
    summary["model_selection_version"] = MODEL_SELECTION_VERSION
    summary["outcome_definition"] = OUTCOME_DEFINITION_VERSION
    summary["premium_included"] = PREMIUM_INCLUDED
    return summary.sort_values(
        [
            "passes_validation_robust_screen",
            "liability_offset_mean_avg",
            "liability_offset_cvar05_avg",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def select_proto_configs_from_validation(
    metrics_df,
    expected_seeds=None,
    max_bound_occupancy=None,
    max_path_touch_rate=None,
):
    """Select prespecified mean and tail configurations without using test data."""
    summary = validation_proto_summary(
        metrics_df,
        expected_seeds=expected_seeds,
        max_bound_occupancy=max_bound_occupancy,
        max_path_touch_rate=max_path_touch_rate,
    )
    if summary.empty:
        return summary, pd.DataFrame()

    eligible = summary[summary["passes_validation_robust_screen"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "No ProtoHedge configuration passed the prespecified validation "
            "robustness screen. The test period remains untouched; revise the "
            "protocol before rerunning rather than selecting from test outcomes."
        )

    selections = []
    for selection_name, objective in VALIDATION_SELECTIONS.items():
        if objective == "liability_offset_mean_avg":
            sort_columns = [
                "liability_offset_mean_avg",
                "liability_offset_cvar05_avg",
                "validation_bound_occupancy_avg",
                "n_prototypes",
                "model",
            ]
        else:
            sort_columns = [
                "liability_offset_cvar05_avg",
                "liability_offset_mean_avg",
                "validation_bound_occupancy_avg",
                "n_prototypes",
                "model",
            ]
        selected = eligible.sort_values(
            sort_columns,
            ascending=[False, False, True, True, True],
        ).iloc[0]
        selections.append(
            {
                "selection": selection_name,
                "selection_objective": objective,
                "selection_split": "validation",
                "model_selection_version": MODEL_SELECTION_VERSION,
                **selected.to_dict(),
            }
        )
    return summary, pd.DataFrame(selections)


def _proto_config_key(record):
    return tuple(record.get(column) for column in PROTO_CONFIG_COLUMNS)


def _training_meta(result):
    history = result.get("history", {})
    return {
        "selected_epoch": history.get("best_epoch"),
        "selected_score": history.get("best_score"),
        "selection_metric": history.get("selection_metric"),
        "selected_val_action_abs_mean": history.get("best_val_action_abs_mean"),
        "selected_val_delta_abs_mean": history.get("best_val_delta_abs_mean"),
        "selected_val_pct_at_any_position_bound": history.get("best_val_pct_at_any_position_bound"),
        "selected_val_pct_paths_touch_any_position_bound": history.get("best_val_pct_paths_touch_any_position_bound"),
    }


def run_real_data_sweep(
    data_path="Data/NEW_PANEL_DECISION_V2/episodes/SPY_training_paths.npy",
    split_path=None,
    episode_metadata_path=None,
    output_dir=".deephedging_real_runs/real_data_sweep",
    samples=512,
    seeds=(1234,),
    epochs=5,
    prototype_counts=(25,),
    prototype_sources=("vanilla", "spot_delta", "zero"),
    weighted_similarity_options=(True, False),
    learn_distance_feature_weights_options=(False,),
    risk_measures=("cvar",),
    normalize=True,
    hedge_mode="step",
    position_bounds=True,
    trade_bounds=None,
    cumulative_bounds=None,
    liability_type="european_call",
    asian_average_type="arithmetic",
    asian_start_step=1,
    asian_end_step=None,
    lr=1e-3,
    device="auto",
    batch_size=None,
    epoch_refresh=None,
    selection_metric="train_loss",
    selection_alpha_action_abs=0.0,
    selection_alpha_delta_abs=0.0,
    selection_alpha_bound_occupancy=0.0,
    selection_alpha_path_bound_touch=0.0,
    action_penalty_weight=0.0,
    delta_penalty_weight=0.0,
    max_bound_occupancy=None,
    max_path_touch_rate=None,
    max_points=None,
    spot_delta_band_grid=DEFAULT_SPOT_DELTA_BAND_GRID,
    tuned_baseline_metric="liability_offset_mean",
    bootstrap_repetitions=2000,
    bootstrap_confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
    bootstrap_block_length=None,
    resume=True,
):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    seeds = tuple(int(seed) for seed in seeds)
    risk_measures = tuple(str(risk) for risk in risk_measures)
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_dir = output_dir / "prototypes"
    prototype_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir / "model_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    model_features = validate_model_features_for_liability(
        liability_type,
        model_features_for_liability(liability_type),
    )
    payoff_state_version = payoff_state_version_for_liability(liability_type)
    if (
        payoff_state_version == ASIAN_PAYOFF_STATE_VERSION
        and ASIAN_PAYOFF_FEATURE not in model_features
    ):
        raise RuntimeError(
            "Asian liabilities require the observable running-average payoff state"
        )

    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 3 or data.shape[-1] < 5:
        raise ValueError(f"Expected data shape [N,T,>=5], got {data.shape}")
    if int(bootstrap_repetitions) <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    splits, split_info, episode_metadata = load_chronological_splits(
        data_path=data_path,
        split_path=split_path,
        episode_metadata_path=episode_metadata_path,
        samples=samples,
    )
    if split_info.get("episode_timing_version") != EPISODE_TIMING_VERSION:
        raise RuntimeError("Real-data sweep received an unsupported episode timing convention")
    if int(data.shape[1]) != int(split_info["n_steps"]) + 1:
        raise RuntimeError("Real-data episodes must have one terminal observation after all decisions")
    bootstrap_block_length = (
        int(split_info["n_steps"])
        if bootstrap_block_length is None
        else int(bootstrap_block_length)
    )
    if bootstrap_block_length <= 0:
        raise ValueError("bootstrap_block_length must be positive")

    resume_expected_config = {
        "data_path": str(data_path.resolve()),
        "samples": samples,
        "split_path": str(Path(split_info["split_path"]).resolve()),
        "episode_metadata_path": str(
            Path(split_info["episode_metadata_path"]).resolve()
        ),
        "temporal_split": split_info,
        "episode_timing_version": EPISODE_TIMING_VERSION,
        "seeds": list(seeds),
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": batch_size,
        "epoch_refresh": epoch_refresh,
        "selection_metric": selection_metric,
        "selection_alpha_action_abs": selection_alpha_action_abs,
        "selection_alpha_delta_abs": selection_alpha_delta_abs,
        "selection_alpha_bound_occupancy": selection_alpha_bound_occupancy,
        "selection_alpha_path_bound_touch": selection_alpha_path_bound_touch,
        "action_penalty_weight": action_penalty_weight,
        "delta_penalty_weight": delta_penalty_weight,
        "max_bound_occupancy": max_bound_occupancy,
        "max_path_touch_rate": max_path_touch_rate,
        "max_points": max_points,
        "spot_delta_band_grid": list(spot_delta_band_grid),
        "tuned_baseline_metric": tuned_baseline_metric,
        "outcome_definition": OUTCOME_DEFINITION_VERSION,
        "premium_included": PREMIUM_INCLUDED,
        "model_selection_version": MODEL_SELECTION_VERSION,
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "bootstrap_confidence": float(bootstrap_confidence),
        "bootstrap_block_length": int(bootstrap_block_length),
        "inference_version": INFERENCE_VERSION,
        "prototype_counts": list(prototype_counts),
        "prototype_sources": list(prototype_sources),
        "weighted_similarity_options": list(weighted_similarity_options),
        "learn_distance_feature_weights_options": list(
            learn_distance_feature_weights_options
        ),
        "risk_measures": list(risk_measures),
        "normalize": normalize,
        "hedge_mode": hedge_mode,
        "hedge_accounting_version": HEDGE_ACCOUNTING_VERSION,
        "position_bounds": position_bounds,
        "trade_bounds": trade_bounds,
        "cumulative_bounds": cumulative_bounds,
        "liability_type": liability_type,
        "payoff_state_version": payoff_state_version,
        "model_features": model_features,
        "asian_average_type": asian_average_type,
        "asian_start_step": asian_start_step,
        "asian_end_step": asian_end_step,
    }
    resume_metrics = pd.DataFrame()
    metrics_path = output_dir / "sweep_metrics.csv"
    config_path = output_dir / "sweep_config.json"
    if bool(resume) and metrics_path.is_file():
        if not config_path.is_file():
            raise RuntimeError(
                f"Cannot safely resume {metrics_path} without {config_path}."
            )
        with open(config_path, "r") as f:
            previous_config = json.load(f)
        _validate_resume_config(previous_config, resume_expected_config)
        resume_metrics = pd.read_csv(metrics_path)
        _validate_resume_metric_keys(resume_metrics)
        if resume_metrics["split"].astype(str).eq("test").any():
            if previous_config.get("checkpoint_stage") == "complete":
                print(f"Complete sweep checkpoint found; reusing {metrics_path}")
                return resume_metrics
            raise RuntimeError(
                "The saved sweep contains test rows but is not marked complete. "
                "Restore the pre-run checkpoint before resuming to keep the locked "
                "test evaluation exactly-once."
            )
        print(
            f"Resume checkpoint found | rows={len(resume_metrics)} | "
            f"stage={previous_config.get('checkpoint_stage', 'unknown')}"
        )

    np.savez(
        output_dir / "fixed_chronological_splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        split_version=split_info["split_version"],
        option_path_version=split_info["option_path_version"],
        episode_timing_version=split_info["episode_timing_version"],
        n_steps=int(split_info["n_steps"]),
        n_observations=int(split_info["n_observations"]),
        payoff_state_version=(
            "" if payoff_state_version is None else payoff_state_version
        ),
        model_features=np.asarray(model_features, dtype=str),
    )
    selected_indices = np.concatenate([splits[name] for name in ("train", "val", "test")])
    selected_metadata = episode_metadata[
        episode_metadata["episode_index"].isin(selected_indices)
    ].copy()
    selected_metadata.to_csv(output_dir / "fixed_episode_metadata.csv", index=False)
    print(
        "fixed chronological split | "
        f"train={len(splits['train'])} | val={len(splits['val'])} | "
        f"test={len(splits['test'])} | version={split_info['split_version']} | "
        f"option_path={split_info['option_path_version']} | "
        f"payoff_state={payoff_state_version or 'terminal-only'} | "
        f"features={model_features}"
    )

    world_kwargs = {
        "seed": None,  # filled per run
        "normalize": normalize,
        "hedge_mode": hedge_mode,
        "position_bounds": position_bounds,
        "trade_bounds": trade_bounds,
        "cumulative_bounds": cumulative_bounds,
        "liability_type": liability_type,
        "asian_average_type": asian_average_type,
        "asian_start_step": asian_start_step,
        "asian_end_step": asian_end_step,
        "payoff_state_version": payoff_state_version,
    }

    all_rows = resume_metrics.to_dict(orient="records")
    run_records = []
    proto_candidate_records = []
    selected_band_by_seed = {}
    vanilla_artifact_records = []
    vanilla_test_offsets = {}

    def checkpoint(stage):
        if not all_rows:
            return pd.DataFrame()

        metrics_df = pd.DataFrame(all_rows)
        metrics_df["liability_type"] = str(liability_type)
        metrics_df["payoff_state_version"] = (
            "" if payoff_state_version is None else payoff_state_version
        )
        metrics_df["model_features"] = "|".join(model_features)
        metrics_df["outcome_definition"] = OUTCOME_DEFINITION_VERSION
        metrics_df["premium_included"] = PREMIUM_INCLUDED
        metrics_df["model_selection_version"] = MODEL_SELECTION_VERSION
        metrics_path = output_dir / "sweep_metrics.csv"
        metrics_tmp_path = output_dir / "sweep_metrics.csv.tmp"
        metrics_df.to_csv(metrics_tmp_path, index=False)
        metrics_tmp_path.replace(metrics_path)
        config_path = output_dir / "sweep_config.json"
        config_tmp_path = output_dir / "sweep_config.json.tmp"
        with open(config_tmp_path, "w") as f:
            json.dump(
                _jsonify(
                    {
                        "data_path": data_path,
                        "output_dir": output_dir,
                        "samples": samples,
                        "split_path": split_info["split_path"],
                        "episode_metadata_path": split_info["episode_metadata_path"],
                        "temporal_split": split_info,
                        "episode_timing_version": EPISODE_TIMING_VERSION,
                        "seeds": list(seeds),
                        "epochs": epochs,
                        "lr": lr,
                        "device": str(device),
                        "batch_size": batch_size,
                        "epoch_refresh": epoch_refresh,
                        "selection_metric": selection_metric,
                        "selection_alpha_action_abs": selection_alpha_action_abs,
                        "selection_alpha_delta_abs": selection_alpha_delta_abs,
                        "selection_alpha_bound_occupancy": selection_alpha_bound_occupancy,
                        "selection_alpha_path_bound_touch": selection_alpha_path_bound_touch,
                        "action_penalty_weight": action_penalty_weight,
                        "delta_penalty_weight": delta_penalty_weight,
                        "max_bound_occupancy": max_bound_occupancy,
                        "max_path_touch_rate": max_path_touch_rate,
                        "max_points": max_points,
                        "spot_delta_band_grid": list(spot_delta_band_grid),
                        "tuned_baseline_metric": tuned_baseline_metric,
                        "outcome_definition": OUTCOME_DEFINITION_VERSION,
                        "premium_included": PREMIUM_INCLUDED,
                        "model_selection_version": MODEL_SELECTION_VERSION,
                        "validation_selection_objectives": VALIDATION_SELECTIONS,
                        "bootstrap_repetitions": int(bootstrap_repetitions),
                        "bootstrap_confidence": float(bootstrap_confidence),
                        "bootstrap_block_length": int(bootstrap_block_length),
                        "inference_version": INFERENCE_VERSION,
                        "prototype_counts": list(prototype_counts),
                        "prototype_sources": list(prototype_sources),
                        "weighted_similarity_options": list(weighted_similarity_options),
                        "learn_distance_feature_weights_options": list(learn_distance_feature_weights_options),
                        "risk_measures": list(risk_measures),
                        "normalize": normalize,
                        "hedge_mode": hedge_mode,
                        "hedge_accounting_version": HEDGE_ACCOUNTING_VERSION,
                        "position_bounds": position_bounds,
                        "trade_bounds": trade_bounds,
                        "cumulative_bounds": cumulative_bounds,
                        "liability_type": liability_type,
                        "payoff_state_version": payoff_state_version,
                        "model_features": model_features,
                        "asian_average_type": asian_average_type,
                        "asian_start_step": asian_start_step,
                        "asian_end_step": asian_end_step,
                        "checkpoint_stage": stage,
                        "runs": run_records,
                    }
                ),
                f,
                indent=2,
            )
        config_tmp_path.replace(config_path)

        if stage == "complete":
            summarize_sweep(
                metrics_df,
                output_dir,
                max_bound_occupancy=max_bound_occupancy,
                max_path_touch_rate=max_path_touch_rate,
            )
            plot_sweep_results(
                metrics_df,
                output_dir,
                max_bound_occupancy=max_bound_occupancy,
                max_path_touch_rate=max_path_touch_rate,
            )
        print(f"\n[checkpoint:{stage}] saved {len(metrics_df)} metric rows to {metrics_path}")
        return metrics_df

    for seed in seeds:
        split_dir = output_dir / f"seed_{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez(split_dir / "splits.npz", **splits)
        seed_world_kwargs = {**world_kwargs, "seed": int(seed)}

        band_selection = select_spot_delta_band(
            data_path=data_path,
            val_indices=splits["val"],
            world_kwargs=seed_world_kwargs,
            band_grid=spot_delta_band_grid,
            selection_metric=tuned_baseline_metric,
        )
        band_selection["candidates"].to_csv(split_dir / "spot_delta_band_selection.csv", index=False)
        selected_band = float(band_selection["best_band"])
        selected_band_by_seed[int(seed)] = selected_band
        print(
            f"\n=== seed={seed} tuned spot-delta band ===\n"
            f"selected band={selected_band:.4f} by {tuned_baseline_metric}"
        )

        # Data-only baselines.
        baselines_complete = _completed_baseline_checkpoint(resume_metrics, seed)
        baseline_artifacts = {}
        for split_name in ("train", "val"):
            idx = splits[split_name]
            _, unhedged_result, unhedged_metrics = evaluate_unhedged(data_path, idx, seed_world_kwargs)
            if not baselines_complete:
                all_rows.append(metric_row("unhedged", split_name, unhedged_metrics, seed=seed, model_family="baseline"))

            _, spot_delta_result, spot_delta_metrics = evaluate_spot_delta(data_path, idx, seed_world_kwargs)
            _, spot_delta_band_result, spot_delta_band_metrics = evaluate_spot_delta_band(
                data_path,
                idx,
                seed_world_kwargs,
                band=selected_band,
            )
            baseline_artifacts[split_name] = {
                "spot_delta": spot_delta_result,
                "spot_delta_band": spot_delta_band_result,
            }
            if not baselines_complete:
                all_rows.append(metric_row("spot_delta", split_name, spot_delta_metrics, seed=seed, model_family="baseline"))
                all_rows.append(
                    metric_row(
                        "spot_delta_band",
                        split_name,
                        spot_delta_band_metrics,
                        seed=seed,
                        model_family="baseline",
                        selected_band=selected_band,
                        baseline_selection_metric=tuned_baseline_metric,
                    )
                )

        run_records.append(
            {
                "seed": int(seed),
                "model": "data_baselines",
                "spot_delta_band": selected_band,
                "baseline_selection_metric": tuned_baseline_metric,
                "split_version": TEMPORAL_SPLIT_VERSION,
                "option_path_version": OPTION_PATH_VERSION,
                "payoff_state_version": payoff_state_version,
                "model_features": model_features,
            }
        )
        if baselines_complete:
            print(f"[resume] seed={seed} baselines already complete; reusing rows")
        else:
            checkpoint(f"seed_{seed}_baselines")

        for risk_measure in risk_measures:
            train_world_cfg = world_cfg(
                data_path=data_path,
                indices=splits["train"],
                val_indices=splits["val"],
                **seed_world_kwargs,
            )
            vanilla_checkpoint = _completed_artifact_checkpoint(
                resume_metrics,
                model="vanilla",
                seed=seed,
                risk_measure=risk_measure,
            )
            if vanilla_checkpoint is not None:
                _, vanilla_artifact_dir, vanilla_artifact_meta = vanilla_checkpoint
                vanilla_meta = _training_meta_from_artifact(vanilla_artifact_meta)
                vanilla = None
                print(
                    f"[resume] seed={seed} risk={risk_measure} vanilla already "
                    "complete; reusing artifact"
                )
            else:
                print(f"\n=== seed={seed} risk={risk_measure} vanilla ===")
                vanilla = run_experiment(
                    override_world=train_world_cfg,
                    override_model=vanilla_model_cfg(feature_list=model_features),
                    override_objective=objective_cfg(risk_measure=risk_measure),
                    override_training=training_cfg(
                        epochs=epochs,
                        lr=lr,
                        device=device,
                        batch_size=batch_size,
                        epoch_refresh=epoch_refresh,
                        selection_metric=selection_metric,
                        selection_alpha_action_abs=selection_alpha_action_abs,
                        selection_alpha_delta_abs=selection_alpha_delta_abs,
                        selection_alpha_bound_occupancy=selection_alpha_bound_occupancy,
                        selection_alpha_path_bound_touch=selection_alpha_path_bound_touch,
                        action_penalty_weight=action_penalty_weight,
                        delta_penalty_weight=delta_penalty_weight,
                        seed=seed,
                    ),
                )
                vanilla_meta = _training_meta(vanilla)
                vanilla_artifact_dir = save_model_artifact(
                    result=vanilla,
                    artifact_root=artifact_root,
                    model_name="vanilla",
                    seed=seed,
                    risk_measure=risk_measure,
                    split_indices=splits,
                    data_path=data_path,
                    world_kwargs=seed_world_kwargs,
                    model_family="vanilla",
                    split_info=split_info,
                )

                for split_name in ("train", "val"):
                    idx = splits[split_name]
                    _, _, metrics = evaluate_gym(
                        vanilla["gym"],
                        data_path,
                        idx,
                        f"vanilla {split_name}",
                        seed_world_kwargs,
                    )
                    all_rows.append(
                        metric_row(
                            "vanilla",
                            split_name,
                            metrics,
                            seed=seed,
                            risk_measure=risk_measure,
                            model_family="vanilla",
                            artifact_dir=str(vanilla_artifact_dir),
                            **vanilla_meta,
                        )
                    )
            vanilla_artifact_records.append(
                {
                    "seed": int(seed),
                    "risk_measure": str(risk_measure),
                    "artifact_dir": str(vanilla_artifact_dir),
                    **vanilla_meta,
                }
            )

            run_records.append(
                {
                    "seed": int(seed),
                    "risk_measure": risk_measure,
                    "model": "vanilla",
                    "artifact_dir": str(vanilla_artifact_dir),
                    "payoff_state_version": payoff_state_version,
                    "model_features": model_features,
                }
            )
            if vanilla_checkpoint is None:
                checkpoint(f"seed_{seed}_risk_{risk_measure}_vanilla")

            source_results = {
                "vanilla": None if vanilla is None else vanilla["training_result"],
                "spot_delta": baseline_artifacts["train"]["spot_delta"],
                "spot_delta_band": baseline_artifacts["train"]["spot_delta_band"],
                "zero": None,
            }
            prototype_world = None if vanilla is None else vanilla["world"]

            for source in prototype_sources:
                if source not in source_results:
                    raise ValueError(f"Unknown prototype source '{source}'")
                for n_prototypes in prototype_counts:
                    prototype_path = prototype_dir / (
                        f"seed_{seed}_risk_{risk_measure}_source_{source}_k_{n_prototypes}.pkl"
                    )
                    prototype_ready = (
                        prototype_path.is_file()
                        and prototype_path.stat().st_size > 0
                    )
                    if not prototype_ready:
                        if prototype_world is None:
                            vanilla_bundle = load_model_artifact(
                                vanilla_artifact_dir
                            )
                            prototype_world, vanilla_train_result, _ = evaluate_gym(
                                vanilla_bundle["gym"],
                                data_path,
                                splits["train"],
                                f"vanilla resume source seed={seed}",
                                seed_world_kwargs,
                            )
                            source_results["vanilla"] = vanilla_train_result
                        payload = build_prototype_payload(
                            world=prototype_world,
                            result=source_results[source],
                            n_prototypes=int(n_prototypes),
                            feature_names=model_features,
                            random_state=int(seed),
                            max_points=max_points,
                        )
                        save_prototype_payload(payload, prototype_path)

                    for weighted in weighted_similarity_options:
                        for learn_weights in learn_distance_feature_weights_options:
                            model_name = (
                                f"proto_source={source}_k={n_prototypes}_"
                                f"weighted={int(bool(weighted))}_learnw={int(bool(learn_weights))}"
                            )
                            proto_checkpoint = _completed_artifact_checkpoint(
                                resume_metrics,
                                model=model_name,
                                seed=seed,
                                risk_measure=risk_measure,
                            )
                            if proto_checkpoint is not None:
                                checkpoint_rows, proto_artifact_dir, artifact_meta = (
                                    proto_checkpoint
                                )
                                expected_meta = {
                                    "model_family": "proto",
                                    "prototype_source": str(source),
                                    "n_prototypes": int(n_prototypes),
                                    "weighted_similarity": bool(weighted),
                                    "learn_distance_feature_weights": bool(
                                        learn_weights
                                    ),
                                }
                                for key, expected_value in expected_meta.items():
                                    if artifact_meta.get(key) != expected_value:
                                        raise RuntimeError(
                                            f"Artifact {proto_artifact_dir} has "
                                            f"{key}={artifact_meta.get(key)!r}; expected "
                                            f"{expected_value!r}."
                                        )
                                proto_meta = _training_meta_from_artifact(
                                    artifact_meta
                                )
                                checkpoint_row = checkpoint_rows.iloc[0]
                                proto_action_norm_mean = float(
                                    checkpoint_row["proto_action_norm_mean"]
                                )
                                proto_action_norm_max = float(
                                    checkpoint_row["proto_action_norm_max"]
                                )
                                if not np.isfinite(
                                    [proto_action_norm_mean, proto_action_norm_max]
                                ).all():
                                    raise RuntimeError(
                                        f"Saved prototype-action norms are invalid for "
                                        f"{model_name!r}, seed {seed}."
                                    )
                                print(
                                    f"[resume] {model_name} seed={seed} "
                                    "already complete; reusing artifact"
                                )
                            else:
                                print(
                                    f"\n=== {model_name} seed={seed} "
                                    f"risk={risk_measure} ==="
                                )
                                proto = run_experiment(
                                    override_world=train_world_cfg,
                                    override_model=proto_model_cfg(
                                        prototype_path,
                                        weighted_similarity=weighted,
                                        learn_distance_feature_weights=learn_weights,
                                        feature_list=model_features,
                                    ),
                                    override_objective=objective_cfg(
                                        risk_measure=risk_measure
                                    ),
                                    override_training=training_cfg(
                                        epochs=epochs,
                                        lr=lr,
                                        device=device,
                                        batch_size=batch_size,
                                        epoch_refresh=epoch_refresh,
                                        selection_metric=selection_metric,
                                        selection_alpha_action_abs=selection_alpha_action_abs,
                                        selection_alpha_delta_abs=selection_alpha_delta_abs,
                                        selection_alpha_bound_occupancy=selection_alpha_bound_occupancy,
                                        selection_alpha_path_bound_touch=selection_alpha_path_bound_touch,
                                        action_penalty_weight=action_penalty_weight,
                                        delta_penalty_weight=delta_penalty_weight,
                                        seed=seed,
                                    ),
                                )
                                proto_meta = _training_meta(proto)
                                proto_artifact_dir = save_model_artifact(
                                    result=proto,
                                    artifact_root=artifact_root,
                                    model_name=model_name,
                                    seed=seed,
                                    risk_measure=risk_measure,
                                    split_indices=splits,
                                    data_path=data_path,
                                    world_kwargs=seed_world_kwargs,
                                    model_family="proto",
                                    prototype_source=source,
                                    n_prototypes=n_prototypes,
                                    weighted_similarity=weighted,
                                    learn_distance_feature_weights=learn_weights,
                                    split_info=split_info,
                                )

                                proto_actions = (
                                    proto["model"]
                                    .prototype_actions.detach()
                                    .cpu()
                                    .numpy()
                                )
                                proto_action_norm_mean = float(
                                    np.linalg.norm(proto_actions, axis=1).mean()
                                )
                                proto_action_norm_max = float(
                                    np.linalg.norm(proto_actions, axis=1).max()
                                )

                                # Candidate configurations remain blind to the test
                                # period until selection is frozen across seeds.
                                for split_name in ("train", "val"):
                                    idx = splits[split_name]
                                    _, _, metrics = evaluate_gym(
                                        proto["gym"],
                                        data_path,
                                        idx,
                                        model_name,
                                        seed_world_kwargs,
                                    )
                                    all_rows.append(
                                        metric_row(
                                            model_name,
                                            split_name,
                                            metrics,
                                            seed=seed,
                                            risk_measure=risk_measure,
                                            model_family="proto",
                                            prototype_source=source,
                                            n_prototypes=int(n_prototypes),
                                            weighted_similarity=bool(weighted),
                                            learn_distance_feature_weights=bool(
                                                learn_weights
                                            ),
                                            proto_action_norm_mean=proto_action_norm_mean,
                                            proto_action_norm_max=proto_action_norm_max,
                                            artifact_dir=str(proto_artifact_dir),
                                            **proto_meta,
                                        )
                                    )

                            candidate_record = {
                                "seed": int(seed),
                                "risk_measure": str(risk_measure),
                                "model": model_name,
                                "model_family": "proto",
                                "prototype_source": source,
                                "n_prototypes": int(n_prototypes),
                                "weighted_similarity": bool(weighted),
                                "learn_distance_feature_weights": bool(learn_weights),
                                "prototype_path": str(prototype_path),
                                "artifact_dir": str(proto_artifact_dir),
                                "payoff_state_version": payoff_state_version,
                                "model_features": model_features,
                                "proto_action_norm_mean": proto_action_norm_mean,
                                "proto_action_norm_max": proto_action_norm_max,
                                **proto_meta,
                            }
                            proto_candidate_records.append(candidate_record)
                            run_records.append(
                                candidate_record
                            )

                            if proto_checkpoint is None:
                                checkpoint(
                                    f"seed_{seed}_risk_{risk_measure}_source_{source}_"
                                    f"k_{n_prototypes}_weighted_{int(bool(weighted))}_"
                                    f"learnw_{int(bool(learn_weights))}"
                                )

    preselection_metrics = pd.DataFrame(all_rows)
    validation_summary, validation_selections = (
        select_proto_configs_from_validation(
            preselection_metrics,
            expected_seeds=len(seeds),
            max_bound_occupancy=max_bound_occupancy,
            max_path_touch_rate=max_path_touch_rate,
        )
    )
    validation_summary.to_csv(
        output_dir / "validation_model_selection.csv",
        index=False,
    )
    validation_selections.to_csv(
        output_dir / "validation_selected_models.csv",
        index=False,
    )

    # Nothing above this line evaluates the locked test period. Baseline
    # hyperparameters, neural checkpoints, and ProtoHedge configurations are
    # now frozen, so every reported model can be evaluated exactly once.
    for seed in seeds:
        seed_world_kwargs = {**world_kwargs, "seed": int(seed)}
        selected_band = selected_band_by_seed[int(seed)]
        _, _, unhedged_metrics = evaluate_unhedged(
            data_path,
            splits["test"],
            seed_world_kwargs,
        )
        all_rows.append(
            metric_row(
                "unhedged",
                "test",
                unhedged_metrics,
                seed=seed,
                model_family="baseline",
            )
        )
        _, _, spot_delta_metrics = evaluate_spot_delta(
            data_path,
            splits["test"],
            seed_world_kwargs,
        )
        all_rows.append(
            metric_row(
                "spot_delta",
                "test",
                spot_delta_metrics,
                seed=seed,
                model_family="baseline",
            )
        )
        _, _, spot_delta_band_metrics = evaluate_spot_delta_band(
            data_path,
            splits["test"],
            seed_world_kwargs,
            band=selected_band,
        )
        all_rows.append(
            metric_row(
                "spot_delta_band",
                "test",
                spot_delta_band_metrics,
                seed=seed,
                model_family="baseline",
                selected_band=selected_band,
                baseline_selection_metric=tuned_baseline_metric,
            )
        )

    for record in vanilla_artifact_records:
        seed = int(record["seed"])
        risk_measure = str(record["risk_measure"])
        seed_world_kwargs = {**world_kwargs, "seed": seed}
        bundle = load_model_artifact(record["artifact_dir"])
        _, test_result, test_metrics = evaluate_gym(
            bundle["gym"],
            data_path,
            splits["test"],
            f"vanilla frozen validation checkpoint test seed={seed}",
            seed_world_kwargs,
        )
        vanilla_test_offsets[(risk_measure, seed)] = _as_numpy(
            test_result.get("liability_offset", test_result["gains"])
        ).reshape(-1)
        all_rows.append(
            metric_row(
                "vanilla",
                "test",
                test_metrics,
                seed=seed,
                risk_measure=risk_measure,
                model_family="vanilla",
                artifact_dir=record["artifact_dir"],
                **{
                    key: record.get(key)
                    for key in [
                        "selected_epoch",
                        "selected_score",
                        "selection_metric",
                        "selected_val_action_abs_mean",
                        "selected_val_delta_abs_mean",
                        "selected_val_pct_at_any_position_bound",
                        "selected_val_pct_paths_touch_any_position_bound",
                    ]
                },
            )
        )

    records_by_key = {}
    for record in proto_candidate_records:
        records_by_key.setdefault(_proto_config_key(record), []).append(record)
    selections_by_key = {}
    for _, selection_row in validation_selections.iterrows():
        key = _proto_config_key(selection_row)
        selections_by_key.setdefault(key, []).append(
            str(selection_row["selection"])
        )

    selected_test_offsets = {}
    for key, selection_labels in selections_by_key.items():
        records = sorted(
            records_by_key.get(key, []),
            key=lambda record: int(record["seed"]),
        )
        if len(records) != len(seeds):
            raise RuntimeError(
                f"Selected configuration {key} has {len(records)} seed artifacts; "
                f"expected {len(seeds)}"
            )
        validation_row = validation_summary[
            validation_summary.apply(
                lambda row: _proto_config_key(row) == key,
                axis=1,
            )
        ].iloc[0]
        selected_for = "|".join(sorted(selection_labels))

        for record in records:
            seed = int(record["seed"])
            bundle = load_model_artifact(record["artifact_dir"])
            seed_world_kwargs = {**world_kwargs, "seed": seed}
            _, test_result, test_metrics = evaluate_gym(
                bundle["gym"],
                data_path,
                splits["test"],
                f"{record['model']} frozen validation selection test",
                seed_world_kwargs,
            )
            selected_test_offsets[(key, seed)] = _as_numpy(
                test_result.get(
                    "liability_offset",
                    test_result["gains"],
                )
            ).reshape(-1)
            all_rows.append(
                metric_row(
                    record["model"],
                    "test",
                    test_metrics,
                    seed=seed,
                    risk_measure=record["risk_measure"],
                    model_family="proto",
                    prototype_source=record["prototype_source"],
                    n_prototypes=int(record["n_prototypes"]),
                    weighted_similarity=bool(
                        record["weighted_similarity"]
                    ),
                    learn_distance_feature_weights=bool(
                        record["learn_distance_feature_weights"]
                    ),
                    proto_action_norm_mean=record[
                        "proto_action_norm_mean"
                    ],
                    proto_action_norm_max=record[
                        "proto_action_norm_max"
                    ],
                    artifact_dir=record["artifact_dir"],
                    selected_for=selected_for,
                    selection_split="validation",
                    selection_objectives="|".join(
                        VALIDATION_SELECTIONS[label]
                        for label in sorted(selection_labels)
                    ),
                    validation_liability_offset_mean_avg=float(
                        validation_row["liability_offset_mean_avg"]
                    ),
                    validation_liability_offset_cvar05_avg=float(
                        validation_row["liability_offset_cvar05_avg"]
                    ),
                    validation_bound_occupancy_avg=float(
                        validation_row["validation_bound_occupancy_avg"]
                    ),
                    validation_path_bound_touch_avg=float(
                        validation_row["validation_path_bound_touch_avg"]
                    ),
                    passes_validation_robust_screen=bool(
                        validation_row[
                            "passes_validation_robust_screen"
                        ]
                    ),
                    **{
                        key_name: record.get(key_name)
                        for key_name in [
                            "selected_epoch",
                            "selected_score",
                            "selection_metric",
                            "selected_val_action_abs_mean",
                            "selected_val_delta_abs_mean",
                            "selected_val_pct_at_any_position_bound",
                            "selected_val_pct_paths_touch_any_position_bound",
                        ]
                    },
                )
            )

    bootstrap_rows = []
    paired_offset_arrays = {}
    paired_offset_metadata = []
    for selection_idx, selection_row in validation_selections.iterrows():
        key = _proto_config_key(selection_row)
        records = sorted(
            records_by_key[key],
            key=lambda record: int(record["seed"]),
        )
        candidate_by_seed = []
        benchmark_by_seed = []
        for record in records:
            seed = int(record["seed"])
            risk_measure = str(record["risk_measure"])
            candidate_by_seed.append(selected_test_offsets[(key, seed)])
            benchmark_by_seed.append(
                vanilla_test_offsets[(risk_measure, seed)]
            )
        candidate_matrix = np.stack(candidate_by_seed, axis=0)
        benchmark_matrix = np.stack(benchmark_by_seed, axis=0)
        candidate_average = candidate_matrix.mean(axis=0)
        benchmark_average = benchmark_matrix.mean(axis=0)
        selection_name = str(selection_row["selection"])
        candidate_key = f"{selection_name}__candidate"
        benchmark_key = f"{selection_name}__benchmark"
        candidate_seed_key = f"{candidate_key}__by_seed"
        benchmark_seed_key = f"{benchmark_key}__by_seed"
        paired_offset_arrays[candidate_key] = candidate_average
        paired_offset_arrays[benchmark_key] = benchmark_average
        paired_offset_arrays[candidate_seed_key] = candidate_matrix
        paired_offset_arrays[benchmark_seed_key] = benchmark_matrix
        paired_offset_metadata.append(
            {
                "selection": selection_name,
                "candidate_key": candidate_key,
                "benchmark_key": benchmark_key,
                "candidate_seed_key": candidate_seed_key,
                "benchmark_seed_key": benchmark_seed_key,
                "candidate_model": selection_row["model"],
                "benchmark_model": "vanilla",
                "risk_measure": selection_row["risk_measure"],
                "n_paths": int(candidate_average.shape[0]),
                "n_seeds": int(candidate_matrix.shape[0]),
                "seed_aggregation": "metric_then_average_across_seeds",
                "selection_split": "validation",
                "model_selection_version": MODEL_SELECTION_VERSION,
                "outcome_definition": OUTCOME_DEFINITION_VERSION,
                "inference_version": INFERENCE_VERSION,
            }
        )
        comparison_rows = paired_seed_circular_block_bootstrap(
            candidate_matrix,
            benchmark_matrix,
            block_length=bootstrap_block_length,
            n_bootstrap=int(bootstrap_repetitions),
            confidence=float(bootstrap_confidence),
            random_state=982451653 + int(selection_idx),
        )
        for row in comparison_rows:
            bootstrap_rows.append(
                {
                    "selection": selection_row["selection"],
                    "selection_split": "validation",
                    "model_selection_version": MODEL_SELECTION_VERSION,
                    "candidate_model": selection_row["model"],
                    "benchmark_model": "vanilla",
                    "risk_measure": selection_row["risk_measure"],
                    "prototype_source": selection_row[
                        "prototype_source"
                    ],
                    "n_prototypes": int(
                        selection_row["n_prototypes"]
                    ),
                    "weighted_similarity": bool(
                        selection_row["weighted_similarity"]
                    ),
                    "learn_distance_feature_weights": bool(
                        selection_row[
                            "learn_distance_feature_weights"
                        ]
                    ),
                    "seed_aggregation": "metric_then_average_across_seeds",
                    "outcome_definition": OUTCOME_DEFINITION_VERSION,
                    "inference_version": INFERENCE_VERSION,
                    **row,
                }
            )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(
        output_dir / "paired_block_bootstrap.csv",
        index=False,
    )
    np.savez_compressed(
        output_dir / "paired_test_liability_offsets.npz",
        **paired_offset_arrays,
    )
    pd.DataFrame(paired_offset_metadata).to_csv(
        output_dir / "paired_test_liability_offsets_metadata.csv",
        index=False,
    )
    run_records.append(
        {
            "model": "validation_selection_manifest",
            "model_selection_version": MODEL_SELECTION_VERSION,
            "selected_configurations": validation_selections.to_dict(
                orient="records"
            ),
            "paired_block_bootstrap_path": str(
                output_dir / "paired_block_bootstrap.csv"
            ),
            "paired_test_liability_offsets_path": str(
                output_dir / "paired_test_liability_offsets.npz"
            ),
        }
    )

    metrics_df = checkpoint("complete")
    print(f"\nsaved sweep metrics: {output_dir / 'sweep_metrics.csv'}")
    return metrics_df


def summarize_sweep(metrics_df, output_dir, max_bound_occupancy=None, max_path_touch_rate=None):
    output_dir = Path(output_dir)
    test = metrics_df[metrics_df["split"] == "test"].copy()
    numeric_defaults = [
        "selected_epoch",
        "selected_score",
        "selected_val_pct_at_any_position_bound",
        "selected_val_pct_paths_touch_any_position_bound",
        "selected_band",
        "validation_liability_offset_mean_avg",
        "validation_liability_offset_cvar05_avg",
        "validation_bound_occupancy_avg",
        "validation_path_bound_touch_avg",
    ]
    for col in numeric_defaults:
        if col not in test.columns:
            test[col] = np.nan
    if "passes_validation_robust_screen" not in test.columns:
        test["passes_validation_robust_screen"] = (
            test.get("model_family", "") != "proto"
        )
    else:
        baseline_mask = test.get("model_family", "") != "proto"
        test.loc[baseline_mask, "passes_validation_robust_screen"] = True
    test["passes_validation_robust_screen"] = test[
        "passes_validation_robust_screen"
    ].map(lambda value: bool(value) if pd.notna(value) else False)
    if "selected_for" not in test.columns:
        test["selected_for"] = ""
    if "selection_split" not in test.columns:
        test["selection_split"] = ""
    group_cols = [
        "model",
        "model_family",
        "prototype_source",
        "n_prototypes",
        "weighted_similarity",
        "learn_distance_feature_weights",
        "risk_measure",
        "liability_type",
        "payoff_state_version",
        "model_features",
        "outcome_definition",
        "premium_included",
    ]
    group_cols = [c for c in group_cols if c in test.columns]
    summary = (
        test.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("liability_offset_mean", "size"),
            liability_offset_mean_avg=("liability_offset_mean", "mean"),
            liability_offset_mean_std=("liability_offset_mean", "std"),
            liability_offset_variance_avg=(
                "liability_offset_variance",
                "mean",
            ),
            liability_offset_p05_avg=("liability_offset_p05", "mean"),
            liability_offset_cvar05_avg=(
                "liability_offset_cvar05",
                "mean",
            ),
            liability_offset_mae_avg=("liability_offset_mae", "mean"),
            liability_offset_rmse_avg=("liability_offset_rmse", "mean"),
            liability_offset_downside_deviation_avg=(
                "liability_offset_downside_deviation",
                "mean",
            ),
            negative_offset_rate_avg=("negative_offset_rate", "mean"),
            delta_abs_max_avg=("delta_abs_max", "mean"),
            action_abs_mean_avg=("action_abs_mean", "mean"),
            pct_at_any_position_bound_avg=("pct_at_any_position_bound", "mean"),
            pct_at_upper_position_bound_avg=("pct_at_upper_position_bound", "mean"),
            pct_at_lower_position_bound_avg=("pct_at_lower_position_bound", "mean"),
            pct_paths_touch_any_position_bound_avg=("pct_paths_touch_any_position_bound", "mean"),
            selected_epoch_avg=("selected_epoch", "mean"),
            selected_score_avg=("selected_score", "mean"),
            selected_val_pct_at_any_position_bound_avg=("selected_val_pct_at_any_position_bound", "mean"),
            selected_val_pct_paths_touch_any_position_bound_avg=("selected_val_pct_paths_touch_any_position_bound", "mean"),
            selected_band_avg=("selected_band", "mean"),
            validation_liability_offset_mean_avg=(
                "validation_liability_offset_mean_avg",
                "mean",
            ),
            validation_liability_offset_cvar05_avg=(
                "validation_liability_offset_cvar05_avg",
                "mean",
            ),
            validation_bound_occupancy_avg=(
                "validation_bound_occupancy_avg",
                "mean",
            ),
            validation_path_bound_touch_avg=(
                "validation_path_bound_touch_avg",
                "mean",
            ),
            passes_validation_robust_screen=(
                "passes_validation_robust_screen",
                "all",
            ),
            selected_for=(
                "selected_for",
                lambda values: "|".join(
                    sorted(
                        {
                            token
                            for value in values.astype(str)
                            for token in value.split("|")
                            if token and token != "nan"
                        }
                    )
                ),
            ),
            selection_split=(
                "selection_split",
                lambda values: "validation"
                if "validation" in set(values.astype(str))
                else "",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "liability_offset_mean_avg",
                "liability_offset_cvar05_avg",
            ],
            ascending=False,
        )
    )
    if max_bound_occupancy is not None:
        summary["passes_test_bound_occupancy_diagnostic"] = (
            summary["pct_at_any_position_bound_avg"] <= float(max_bound_occupancy)
        )
    else:
        summary["passes_test_bound_occupancy_diagnostic"] = True
    if max_path_touch_rate is not None:
        summary["passes_test_path_touch_diagnostic"] = (
            summary["pct_paths_touch_any_position_bound_avg"] <= float(max_path_touch_rate)
        )
    else:
        summary["passes_test_path_touch_diagnostic"] = True
    summary["passes_test_robustness_diagnostic"] = (
        summary["passes_test_bound_occupancy_diagnostic"]
        & summary["passes_test_path_touch_diagnostic"]
    )
    summary["passes_robust_screen"] = summary[
        "passes_validation_robust_screen"
    ]
    summary.to_csv(output_dir / "sweep_test_summary.csv", index=False)
    return summary


def build_paper_report(metrics_df, output_dir, max_bound_occupancy=None, max_path_touch_rate=None):
    """Build paper tables from frozen validation selections and test outcomes."""
    output_dir = Path(output_dir)
    summary = summarize_sweep(
        metrics_df,
        output_dir,
        max_bound_occupancy=max_bound_occupancy,
        max_path_touch_rate=max_path_touch_rate,
    ).copy()
    validation_summary, validation_selections = (
        select_proto_configs_from_validation(
            metrics_df,
            expected_seeds=int(metrics_df["seed"].nunique()),
            max_bound_occupancy=max_bound_occupancy,
            max_path_touch_rate=max_path_touch_rate,
        )
    )
    validation_summary.to_csv(
        output_dir / "validation_model_selection.csv",
        index=False,
    )
    validation_selections.to_csv(
        output_dir / "validation_selected_models.csv",
        index=False,
    )

    def _baseline_value(model_name, metric, risk_measure=None):
        rows = summary[summary["model"] == model_name]
        if risk_measure is not None and "risk_measure" in rows.columns:
            same_risk = rows[rows["risk_measure"] == risk_measure]
            if not same_risk.empty:
                rows = same_risk
        if rows.empty:
            return np.nan
        return float(rows.iloc[0][metric])

    higher_is_better = [
        "liability_offset_mean_avg",
        "liability_offset_cvar05_avg",
    ]
    lower_is_better = [
        "liability_offset_rmse_avg",
        "liability_offset_downside_deviation_avg",
    ]
    for metric in higher_is_better + lower_is_better:
        for baseline in ["unhedged", "spot_delta", "spot_delta_band", "vanilla"]:
            col = (
                f"{metric}_minus_{baseline}"
                if metric in higher_is_better
                else f"{metric}_improvement_vs_{baseline}"
            )
            values = []
            for _, row in summary.iterrows():
                baseline_value = _baseline_value(baseline, metric, row.get("risk_measure"))
                if metric in higher_is_better:
                    values.append(float(row[metric]) - baseline_value)
                else:
                    values.append(baseline_value - float(row[metric]))
            summary[col] = values

    summary.to_csv(output_dir / "paper_model_comparison.csv", index=False)

    best_rows = []
    for _, selected in validation_selections.iterrows():
        key = _proto_config_key(selected)
        matching_test = summary[
            summary.apply(
                lambda row: _proto_config_key(row) == key,
                axis=1,
            )
        ]
        if len(matching_test) != 1:
            raise RuntimeError(
                "Each frozen validation selection must have exactly one "
                f"aggregated test row; found {len(matching_test)} for {key}"
            )
        test_row = matching_test.iloc[0]
        objective = str(selected["selection_objective"])
        best_rows.append(
            {
                "selection": selected["selection"],
                "selection_objective": objective,
                "selection_split": "validation",
                "model_selection_version": MODEL_SELECTION_VERSION,
                "validation_selection_score": float(selected[objective]),
                "validation_liability_offset_mean_avg": float(
                    selected["liability_offset_mean_avg"]
                ),
                "validation_liability_offset_cvar05_avg": float(
                    selected["liability_offset_cvar05_avg"]
                ),
                "validation_bound_occupancy_avg": float(
                    selected["validation_bound_occupancy_avg"]
                ),
                "validation_path_bound_touch_avg": float(
                    selected["validation_path_bound_touch_avg"]
                ),
                **test_row.to_dict(),
            }
        )

    if best_rows:
        best_df = pd.DataFrame(best_rows)
    else:
        best_df = pd.DataFrame()
    best_df.to_csv(output_dir / "paper_best_models.csv", index=False)

    selected_models = ["unhedged", "spot_delta", "spot_delta_band", "vanilla"]
    if not best_df.empty:
        selected_models.extend(best_df["model"].astype(str).tolist())
    paper_table = summary[summary["model"].astype(str).isin(dict.fromkeys(selected_models).keys())].copy()
    paper_table = paper_table.sort_values(
        [
            "liability_offset_mean_avg",
            "liability_offset_cvar05_avg",
        ],
        ascending=False,
    )
    paper_table.to_csv(output_dir / "paper_selected_table.csv", index=False)

    _write_interpretation_md(summary=summary, best_df=best_df, output_dir=output_dir)
    return {
        "summary": summary,
        "best": best_df,
        "selected": paper_table,
    }


def _fmt(x):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.6f}"


def _write_interpretation_md(summary, best_df, output_dir):
    output_dir = Path(output_dir)
    lines = [
        "# Real-Data ProtoHedge Sweep Interpretation",
        "",
        "This report is generated from `sweep_metrics.csv`. The reported "
        "outcome is premium-excluded liability offset, not economic profit "
        "and loss. Higher mean and lower-tail CVaR are better; lower RMSE "
        "and downside deviation are better.",
        "",
        "ProtoHedge configurations are selected from validation observations "
        "averaged across seeds. Only the two prespecified frozen selections "
        "are evaluated on the held-out test period.",
        "",
    ]

    if summary.empty:
        lines.append("No test summary rows were available.")
        (output_dir / "interpretation.md").write_text("\n".join(lines))
        return

    baselines = summary[summary["model"].isin(["unhedged", "spot_delta", "spot_delta_band", "vanilla"])].copy()
    if not baselines.empty:
        lines.extend(["## Baselines", ""])
        for _, row in baselines.iterrows():
            line = (
                f"- `{row['model']}`: mean offset "
                f"`{_fmt(row['liability_offset_mean_avg'])}`, "
                f"offset CVaR5 "
                f"`{_fmt(row['liability_offset_cvar05_avg'])}`, "
                f"offset RMSE "
                f"`{_fmt(row['liability_offset_rmse_avg'])}`"
            )
            if "selected_band_avg" in row.index and not pd.isna(row["selected_band_avg"]):
                line += f", selected band `{_fmt(row['selected_band_avg'])}`"
            if "pct_at_any_position_bound_avg" in row.index:
                line += f", bound occupancy `{_fmt(row['pct_at_any_position_bound_avg'])}`"
            lines.append(line)
        lines.append("")

    if not best_df.empty:
        lines.extend(["## Frozen Validation Selections", ""])
        vanilla = summary[summary["model"] == "vanilla"]
        for _, row in best_df.iterrows():
            line = (
                f"- `{row['selection']}` chose `{row['model']}` on validation "
                f"using `{row['selection_objective']}`. Its test mean offset is "
                f"`{_fmt(row['liability_offset_mean_avg'])}`, test offset CVaR5 "
                f"is `{_fmt(row['liability_offset_cvar05_avg'])}`, and test "
                f"offset RMSE is `{_fmt(row['liability_offset_rmse_avg'])}`."
            )
            lines.append(line)
            if not vanilla.empty:
                same_risk = vanilla[
                    vanilla["risk_measure"] == row["risk_measure"]
                ]
                benchmark = same_risk.iloc[0] if not same_risk.empty else vanilla.iloc[0]
                lines.append(
                    f"- Test mean-offset difference versus Deep Hedging: "
                    f"`{_fmt(row['liability_offset_mean_avg'] - benchmark['liability_offset_mean_avg'])}`; "
                    f"test CVaR5 difference: "
                    f"`{_fmt(row['liability_offset_cvar05_avg'] - benchmark['liability_offset_cvar05_avg'])}`."
                )
        lines.append("")

    bootstrap_path = output_dir / "paired_block_bootstrap.csv"
    if bootstrap_path.exists():
        bootstrap = pd.read_csv(bootstrap_path)
        if not bootstrap.empty:
            lines.extend(["## Dependence-Aware Uncertainty", ""])
            for _, row in bootstrap.iterrows():
                lines.append(
                    f"- `{row['selection']}` / `{row['metric']}`: estimate "
                    f"`{_fmt(row['estimate'])}`, "
                    f"{float(row['confidence']):.0%} circular-block interval "
                    f"`[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]`."
                )
            lines.append("")

    lines.extend(
        [
            "## Larger Picture",
            "",
            "This sweep is the empirical bridge between the implementation and the paper extension. "
            "A positive result means more than training code running: it means a compact prototype "
            "policy can compete with neural Deep Hedging and simple classical hedges on once-used held-out "
            "historical paths while remaining inspectable.",
            "",
            "Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data "
            "runs, multiple seeds, and robustness checks across prototype counts and market regimes.",
        ]
    )

    (output_dir / "interpretation.md").write_text("\n".join(lines))


def plot_sweep_results(metrics_df, output_dir, max_bound_occupancy=None, max_path_touch_rate=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = metrics_df[metrics_df["split"] == "test"].copy()
    if test.empty:
        return

    summary = summarize_sweep(
        metrics_df,
        output_dir,
        max_bound_occupancy=max_bound_occupancy,
        max_path_touch_rate=max_path_touch_rate,
    )
    build_paper_report(
        metrics_df,
        output_dir,
        max_bound_occupancy=max_bound_occupancy,
        max_path_touch_rate=max_path_touch_rate,
    )
    top = summary.head(25).copy()
    labels = top["model"].astype(str)

    def _bar(metric, filename, title, xlabel):
        fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(top))))
        ax.barh(labels[::-1], top[metric].to_numpy()[::-1])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    _bar(
        "liability_offset_mean_avg",
        "sweep_test_liability_offset_mean.png",
        "Test Mean Liability Offset By Model",
        "mean liability offset",
    )
    _bar(
        "liability_offset_cvar05_avg",
        "sweep_test_liability_offset_cvar05.png",
        "Test 5% CVaR of Liability Offset By Model",
        "5% CVaR of liability offset",
    )
    _bar(
        "liability_offset_rmse_avg",
        "sweep_test_liability_offset_rmse.png",
        "Test Liability-Offset RMSE By Model",
        "liability-offset RMSE",
    )
    _bar(
        "liability_offset_downside_deviation_avg",
        "sweep_test_liability_offset_downside_deviation.png",
        "Test Liability-Offset Downside Deviation By Model",
        "liability-offset downside deviation",
    )
    if "pct_at_any_position_bound_avg" in top.columns:
        _bar(
            "pct_at_any_position_bound_avg",
            "sweep_test_bound_saturation.png",
            "Test Position-Bound Occupancy By Model",
            "fraction of delta states at a bound",
        )

    proto = metrics_df[
        (metrics_df["split"] == "val")
        & (metrics_df.get("model_family", "") == "proto")
    ].copy()
    if not proto.empty and {"n_prototypes", "prototype_source", "weighted_similarity"}.issubset(proto.columns):
        fig, ax = plt.subplots(figsize=(10, 5))
        for (source, weighted), grp in proto.groupby(["prototype_source", "weighted_similarity"], dropna=False):
            curve = (
                grp.groupby("n_prototypes")
                .agg(
                    liability_offset_mean=(
                        "liability_offset_mean",
                        "mean",
                    ),
                    liability_offset_cvar05=(
                        "liability_offset_cvar05",
                        "mean",
                    ),
                )
                .reset_index()
                .sort_values("n_prototypes")
            )
            label = f"{source}, weighted={int(bool(weighted))}"
            ax.plot(
                curve["n_prototypes"],
                curve["liability_offset_mean"],
                marker="o",
                label=label,
            )
        ax.set_title("ProtoHedge Validation Mean Offset vs Prototype Count")
        ax.set_xlabel("number of prototypes")
        ax.set_ylabel("validation mean liability offset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "proto_validation_offset_vs_n_prototypes.png",
            dpi=160,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for (source, weighted), grp in proto.groupby(["prototype_source", "weighted_similarity"], dropna=False):
            curve = (
                grp.groupby("n_prototypes")
                .agg(
                    liability_offset_cvar05=(
                        "liability_offset_cvar05",
                        "mean",
                    )
                )
                .reset_index()
                .sort_values("n_prototypes")
            )
            label = f"{source}, weighted={int(bool(weighted))}"
            ax.plot(
                curve["n_prototypes"],
                curve["liability_offset_cvar05"],
                marker="o",
                label=label,
            )
        ax.set_title("ProtoHedge Validation Offset CVaR5 vs Prototype Count")
        ax.set_xlabel("number of prototypes")
        ax.set_ylabel("validation 5% CVaR of liability offset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "proto_validation_offset_cvar05_vs_n_prototypes.png",
            dpi=160,
        )
        plt.close(fig)


def _parse_bool_list(values):
    if values is None:
        return None
    out = []
    for value in values:
        if isinstance(value, bool):
            out.append(value)
        else:
            out.append(str(value).lower() in ["1", "true", "yes", "y"])
    return tuple(out)


def _parse_optional_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = str(value).strip().lower()
    if value in ["none", "null", "all", "full"]:
        return None
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="Data/NEW_PANEL_DECISION_V2/episodes/SPY_training_paths.npy")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--episode-metadata-path", default=None)
    parser.add_argument("--output-dir", default=".deephedging_real_runs/real_data_sweep")
    parser.add_argument(
        "--samples",
        type=_parse_optional_int,
        default=512,
        help="Number of paths to use, or 'none'/'all' for the full dataset.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--n-prototypes", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--prototype-sources", nargs="+", default=["vanilla", "spot_delta", "zero"])
    parser.add_argument("--weighted", nargs="+", default=["true", "false"])
    parser.add_argument("--learn-weights", nargs="+", default=["false"])
    parser.add_argument("--risk-measures", nargs="+", default=["cvar"])
    parser.add_argument("--liability-type", default="european_call")
    parser.add_argument("--asian-average-type", default="arithmetic")
    parser.add_argument("--asian-start-step", type=int, default=0)
    parser.add_argument("--asian-end-step", type=_parse_optional_int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument("--epoch-refresh", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--selection-metric", default="train_loss")
    parser.add_argument("--selection-alpha-action-abs", type=float, default=0.0)
    parser.add_argument("--selection-alpha-delta-abs", type=float, default=0.0)
    parser.add_argument("--selection-alpha-bound-occupancy", type=float, default=0.0)
    parser.add_argument("--selection-alpha-path-bound-touch", type=float, default=0.0)
    parser.add_argument("--action-penalty-weight", type=float, default=0.0)
    parser.add_argument("--delta-penalty-weight", type=float, default=0.0)
    parser.add_argument("--max-bound-occupancy", type=float, default=None)
    parser.add_argument("--max-path-touch-rate", type=float, default=None)
    parser.add_argument(
        "--spot-delta-band-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPOT_DELTA_BAND_GRID),
    )
    parser.add_argument(
        "--tuned-baseline-metric",
        default="liability_offset_mean",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=DEFAULT_BOOTSTRAP_CONFIDENCE,
    )
    parser.add_argument(
        "--bootstrap-block-length",
        type=_parse_optional_int,
        default=None,
    )
    args = parser.parse_args()

    run_real_data_sweep(
        data_path=args.data_path,
        split_path=args.split_path,
        episode_metadata_path=args.episode_metadata_path,
        output_dir=args.output_dir,
        samples=args.samples,
        epochs=args.epochs,
        seeds=tuple(args.seeds),
        prototype_counts=tuple(args.n_prototypes),
        prototype_sources=tuple(args.prototype_sources),
        weighted_similarity_options=_parse_bool_list(args.weighted),
        learn_distance_feature_weights_options=_parse_bool_list(args.learn_weights),
        risk_measures=tuple(args.risk_measures),
        liability_type=args.liability_type,
        asian_average_type=args.asian_average_type,
        asian_start_step=args.asian_start_step,
        asian_end_step=args.asian_end_step,
        lr=args.lr,
        device=args.device,
        epoch_refresh=args.epoch_refresh,
        selection_metric=args.selection_metric,
        selection_alpha_action_abs=args.selection_alpha_action_abs,
        selection_alpha_delta_abs=args.selection_alpha_delta_abs,
        selection_alpha_bound_occupancy=args.selection_alpha_bound_occupancy,
        selection_alpha_path_bound_touch=args.selection_alpha_path_bound_touch,
        action_penalty_weight=args.action_penalty_weight,
        delta_penalty_weight=args.delta_penalty_weight,
        max_bound_occupancy=args.max_bound_occupancy,
        max_path_touch_rate=args.max_path_touch_rate,
        max_points=args.max_points,
        spot_delta_band_grid=tuple(args.spot_delta_band_grid),
        tuned_baseline_metric=args.tuned_baseline_metric,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_block_length=args.bootstrap_block_length,
    )


if __name__ == "__main__":
    main()

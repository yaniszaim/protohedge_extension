"""Helpers for post-sweep real-data ProtoHedge analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deephedging.proto_analysis_torch import load_prototype_payload
from deephedging.run_train_torch import build_experiment_components
from deephedging.hedge_accounting import HEDGE_ACCOUNTING_VERSION
from deephedging.panel_data_pipeline import (
    EPISODE_TIMING_VERSION,
    OPTION_PATH_VERSION,
    TEMPORAL_SPLIT_VERSION,
)
from deephedging.payoff_state import (
    ASIAN_PAYOFF_FEATURE,
    ASIAN_PAYOFF_STATE_VERSION,
    is_asian_liability,
    payoff_state_version_for_liability,
)
from deephedging.outcome_metrics import (
    OUTCOME_DEFINITION_VERSION,
    PREMIUM_INCLUDED,
    summarize_liability_offset,
)


ARTIFACT_META = "artifact_metadata.json"
ARTIFACT_STATE = "gym_state.pt"
ARTIFACT_HISTORY = "training_history.json"


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _jsonify(x):
    if isinstance(x, dict):
        return {str(key): _jsonify(value) for key, value in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(value) for value in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return slug[:180] or "model"


def save_model_artifact(
    result,
    artifact_root,
    model_name,
    seed,
    risk_measure,
    split_indices,
    data_path,
    world_kwargs,
    model_family,
    prototype_source=None,
    n_prototypes=None,
    weighted_similarity=None,
    learn_distance_feature_weights=None,
    split_info=None,
):
    artifact_root = Path(artifact_root)
    artifact_dir = artifact_root / f"seed_{int(seed)}" / f"risk_{risk_measure}" / slugify_name(model_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {
        k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
        for k, v in result["gym"].state_dict().items()
    }
    torch.save(state_dict, artifact_dir / ARTIFACT_STATE)

    model = result.get("model")
    if hasattr(model, "prototype_actions"):
        np.save(artifact_dir / "prototype_actions.npy", _np(model.prototype_actions))

    full_history = _jsonify(result.get("history", {}))
    with open(artifact_dir / ARTIFACT_HISTORY, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2, sort_keys=True)
        f.write("\n")

    result_config = result.get("config", {})
    result_world = result.get("world")
    liability_type = str(
        getattr(
            result_world,
            "liability_type",
            result_config.get("world", {}).get(
                "liability_type",
                world_kwargs.get("liability_type", "european_call"),
            ),
        )
    )
    model_features = sorted(
        result_config.get("model", {}).get(
            "feature_list",
            ["price", "delta", "time_left"],
        )
    )
    payoff_state_version = getattr(
        result_world,
        "payoff_state_version",
        payoff_state_version_for_liability(liability_type),
    )
    meta = {
        "model_name": str(model_name),
        "model_family": str(model_family),
        "seed": int(seed),
        "risk_measure": str(risk_measure),
        "prototype_source": prototype_source,
        "n_prototypes": None if n_prototypes is None else int(n_prototypes),
        "weighted_similarity": None if weighted_similarity is None else bool(weighted_similarity),
        "learn_distance_feature_weights": (
            None if learn_distance_feature_weights is None else bool(learn_distance_feature_weights)
        ),
        "prototype_path": result.get("config", {}).get("model", {}).get("prototype_path"),
        "data_path": str(data_path),
        "world_kwargs": world_kwargs,
        "liability_type": liability_type,
        "model_features": model_features,
        "payoff_state_version": payoff_state_version,
        "outcome_definition": OUTCOME_DEFINITION_VERSION,
        "premium_included": PREMIUM_INCLUDED,
        "hedge_accounting": getattr(result.get("world"), "pnl_accounting", "trade_to_terminal"),
        "hedge_accounting_version": HEDGE_ACCOUNTING_VERSION,
        "split_indices": {k: [int(i) for i in v] for k, v in split_indices.items()},
        "temporal_split_version": (
            None if split_info is None else split_info.get("split_version")
        ),
        "option_path_version": (
            None if split_info is None else split_info.get("option_path_version")
        ),
        "episode_timing_version": (
            None if split_info is None else split_info.get("episode_timing_version")
        ),
        "temporal_split": split_info,
        "config": result_config,
        "history": {
            "history_file": ARTIFACT_HISTORY,
            "best_epoch": result.get("history", {}).get("best_epoch"),
            "best_score": result.get("history", {}).get("best_score"),
            "best_val_loss": result.get("history", {}).get("best_val_loss"),
            "init_loss": result.get("history", {}).get("init_loss"),
            "init_val_loss": result.get("history", {}).get("init_val_loss"),
            "selection_metric": result.get("history", {}).get("selection_metric"),
            "best_val_action_abs_mean": result.get("history", {}).get("best_val_action_abs_mean"),
            "best_val_delta_abs_mean": result.get("history", {}).get("best_val_delta_abs_mean"),
            "best_val_pct_at_any_position_bound": result.get("history", {}).get("best_val_pct_at_any_position_bound"),
            "best_val_pct_paths_touch_any_position_bound": result.get("history", {}).get("best_val_pct_paths_touch_any_position_bound"),
        },
    }
    with open(artifact_dir / ARTIFACT_META, "w") as f:
        json.dump(meta, f, indent=2)
    return artifact_dir


def load_model_artifact(artifact_dir, map_location="cpu"):
    artifact_dir = Path(artifact_dir)
    with open(artifact_dir / ARTIFACT_META, "r") as f:
        meta = json.load(f)

    world_cfg = meta.get("config", {}).get("world", {})
    world_type = str(world_cfg.get("world_type", "synthetic")).lower()
    hedge_mode = str(world_cfg.get("hedge_mode", "terminal")).lower()
    liability_type = str(
        meta.get(
            "liability_type",
            world_cfg.get("liability_type", "european_call"),
        )
    )
    is_legacy_step_artifact = (
        world_type in {"real", "real_data", "world_real"}
        and hedge_mode in {"step", "period", "one_step"}
        and meta.get("hedge_accounting_version") != HEDGE_ACCOUNTING_VERSION
    )
    if is_legacy_step_artifact:
        raise RuntimeError(
            f"Artifact '{artifact_dir}' was trained before corrected cumulative-inventory "
            "P&L accounting was versioned. Retrain this real-data model; evaluating its "
            "old weights under the corrected equation would not produce a valid result."
        )

    if (
        world_type in {"real", "real_data", "world_real"}
        and meta.get("temporal_split_version") != TEMPORAL_SPLIT_VERSION
    ):
        raise RuntimeError(
            f"Artifact '{artifact_dir}' was not trained with the required fixed "
            "chronological pre-window split. Retrain it before using it for empirical claims."
        )
    if (
        world_type in {"real", "real_data", "world_real"}
        and meta.get("option_path_version") != OPTION_PATH_VERSION
    ):
        raise RuntimeError(
            f"Artifact '{artifact_dir}' was not trained on contract-consistent listed-option "
            "paths. Rebuild the panel and retrain before using it for empirical claims."
        )
    if (
        world_type in {"real", "real_data", "world_real"}
        and meta.get("episode_timing_version") != EPISODE_TIMING_VERSION
    ):
        raise RuntimeError(
            f"Artifact '{artifact_dir}' was not trained with one terminal observation "
            "after the final hedge decision. Rebuild the panel and retrain it."
        )
    if (
        world_type in {"real", "real_data", "world_real"}
        and is_asian_liability(liability_type)
    ):
        model_features = sorted(
            meta.get(
                "model_features",
                meta.get("config", {}).get("model", {}).get("feature_list", []),
            )
        )
        if meta.get("payoff_state_version") != ASIAN_PAYOFF_STATE_VERSION:
            raise RuntimeError(
                f"Artifact '{artifact_dir}' predates the required observable Asian "
                "payoff state. Retrain it with running-average moneyness before "
                "using it for empirical claims."
            )
        if ASIAN_PAYOFF_FEATURE not in model_features:
            raise RuntimeError(
                f"Artifact '{artifact_dir}' does not include "
                f"'{ASIAN_PAYOFF_FEATURE}' in the policy state. Retrain it before "
                "using it for empirical claims."
            )

    components = build_experiment_components(meta["config"])
    state_dict = torch.load(
        artifact_dir / ARTIFACT_STATE,
        map_location=map_location,
        weights_only=True,
    )
    components["gym"].load_state_dict(state_dict)
    components["gym"].eval()
    components["artifact_dir"] = artifact_dir
    components["metadata"] = meta
    return components


def list_saved_artifacts(output_dir):
    output_dir = Path(output_dir)
    rows = []
    for meta_path in sorted(output_dir.glob(f"model_artifacts/**/{ARTIFACT_META}")):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        rows.append({
            "artifact_dir": str(meta_path.parent),
            "model_name": meta.get("model_name"),
            "model_family": meta.get("model_family"),
            "seed": meta.get("seed"),
            "risk_measure": meta.get("risk_measure"),
            "prototype_source": meta.get("prototype_source"),
            "n_prototypes": meta.get("n_prototypes"),
            "weighted_similarity": meta.get("weighted_similarity"),
            "learn_distance_feature_weights": meta.get("learn_distance_feature_weights"),
            "best_epoch": meta.get("history", {}).get("best_epoch"),
            "best_score": meta.get("history", {}).get("best_score"),
            "temporal_split_version": meta.get("temporal_split_version"),
            "option_path_version": meta.get("option_path_version"),
            "episode_timing_version": meta.get("episode_timing_version"),
            "liability_type": meta.get("liability_type"),
            "payoff_state_version": meta.get("payoff_state_version"),
            "model_features": "|".join(meta.get("model_features", [])),
            "outcome_definition": meta.get("outcome_definition"),
            "premium_included": meta.get("premium_included"),
        })
    return pd.DataFrame(rows)


def evaluate_saved_artifact(artifact_dir, split="test"):
    from deephedging.real_data_sweep_torch import evaluate_gym

    bundle = load_model_artifact(artifact_dir)
    meta = bundle["metadata"]
    split_indices = meta["split_indices"][split]
    world_kwargs = dict(meta["world_kwargs"])
    eval_world, result, metrics = evaluate_gym(
        bundle["gym"],
        data_path=meta["data_path"],
        indices=split_indices,
        label=f"{meta['model_name']} {split}",
        world_kwargs=world_kwargs,
    )
    bundle["eval_world"] = eval_world
    bundle["eval_split"] = split
    return bundle, result, metrics


def evaluate_saved_baselines(artifact_dir, split="test"):
    from deephedging.real_data_sweep_torch import (
        DEFAULT_SPOT_DELTA_BAND_GRID,
        evaluate_spot_delta,
        evaluate_spot_delta_band,
        evaluate_unhedged,
        select_spot_delta_band,
    )

    artifact_dir = Path(artifact_dir)
    with open(artifact_dir / ARTIFACT_META, "r") as f:
        meta = json.load(f)
    split_indices = meta["split_indices"][split]
    world_kwargs = dict(meta["world_kwargs"])
    band_selection = select_spot_delta_band(
        meta["data_path"],
        meta["split_indices"]["val"],
        world_kwargs,
        band_grid=DEFAULT_SPOT_DELTA_BAND_GRID,
        selection_metric="liability_offset_mean",
    )
    selected_band = float(band_selection["best_band"])
    test_world, unhedged_result, unhedged_metrics = evaluate_unhedged(meta["data_path"], split_indices, world_kwargs)
    _, spot_delta_result, spot_delta_metrics = evaluate_spot_delta(meta["data_path"], split_indices, world_kwargs)
    _, spot_delta_band_result, spot_delta_band_metrics = evaluate_spot_delta_band(
        meta["data_path"],
        split_indices,
        world_kwargs,
        band=selected_band,
    )
    return {
        "world": test_world,
        "unhedged": {"result": unhedged_result, "metrics": unhedged_metrics},
        "spot_delta": {"result": spot_delta_result, "metrics": spot_delta_metrics},
        "spot_delta_band": {
            "result": spot_delta_band_result,
            "metrics": {**spot_delta_band_metrics, "selected_band": selected_band},
            "selected_band": selected_band,
            "band_selection": band_selection["candidates"],
        },
    }


def build_regime_frame(world, result_dict):
    spot = _np(world.data.features.per_step["spot"]).astype(np.float64)
    ivol = _np(world.data.features.per_step["ivol"]).astype(np.float64)
    price0 = np.maximum(spot[:, 0], 1e-8)
    terminal_return = spot[:, -1] / price0 - 1.0
    log_returns = np.diff(np.log(np.maximum(spot, 1e-8)), axis=1)
    realized_vol = log_returns.std(axis=1)
    avg_ivol = ivol.mean(axis=1)
    running_peak = np.maximum.accumulate(spot, axis=1)
    drawdowns = spot / np.maximum(running_peak, 1e-8) - 1.0
    max_drawdown = drawdowns.min(axis=1)

    frame = pd.DataFrame({
        "terminal_return": terminal_return,
        "realized_vol": realized_vol,
        "avg_ivol": avg_ivol,
        "max_drawdown": max_drawdown,
    })
    frame["return_regime"] = pd.qcut(frame["terminal_return"], q=3, labels=["down", "middle", "up"], duplicates="drop")
    frame["vol_regime"] = pd.qcut(frame["avg_ivol"], q=3, labels=["low_vol", "mid_vol", "high_vol"], duplicates="drop")
    frame["drawdown_regime"] = pd.qcut(frame["max_drawdown"], q=3, labels=["mild_dd", "mid_dd", "deep_dd"], duplicates="drop")

    for name, result in result_dict.items():
        liability_offset = _np(
            result.get("liability_offset", result["gains"])
        ).reshape(-1)
        frame[f"{name}_liability_offset"] = liability_offset
        frame[f"{name}_trading_gain"] = _np(result["pnl"]).reshape(-1)
        frame[f"{name}_cost"] = _np(result["cost"]).reshape(-1)
    return frame


def summarize_by_regime(frame, regime_col, series_cols=None):
    if series_cols is None:
        series_cols = [
            c
            for c in frame.columns
            if c.endswith("_liability_offset") or c.endswith("_payoff")
        ]
    rows = []
    for regime, grp in frame.groupby(regime_col, observed=True):
        for col in series_cols:
            values = grp[col].to_numpy(dtype=float)
            offset_metrics = summarize_liability_offset(values)
            rows.append({
                "regime_type": regime_col,
                "regime": str(regime),
                "series": col,
                "n": int(len(values)),
                "mean": offset_metrics["liability_offset_mean"],
                "p05": offset_metrics["liability_offset_p05"],
                "cvar05": offset_metrics["liability_offset_cvar05"],
                "mae": offset_metrics["liability_offset_mae"],
                "rmse": offset_metrics["liability_offset_rmse"],
                "downside_deviation": offset_metrics[
                    "liability_offset_downside_deviation"
                ],
                "negative_offset_rate": offset_metrics[
                    "negative_offset_rate"
                ],
            })
    return pd.DataFrame(rows)


def _expanded_feature_columns(feature_names, world, input_dim):
    per_step = world.data.features.per_step
    n_inst = int(world.nInst)
    cols = []
    count = 0
    for name in sorted(feature_names or ["price", "delta", "time_left"]):
        if name in per_step:
            arr = _np(per_step[name])
            width = 1 if arr.ndim == 2 else int(arr.shape[-1])
        elif name in ["delta", "action"]:
            width = n_inst
        elif name in ["pnl", "cost"]:
            width = 1
        else:
            width = 1
        if width == 1:
            cols.append(name)
            count += 1
        else:
            for j in range(width):
                cols.append(f"{name}_{j}")
                count += 1
    if count != int(input_dim):
        cols = [f"feature_{i}" for i in range(int(input_dim))]
    return cols


def prototype_usage_table(bundle, result, top_n=15):
    meta = bundle["metadata"]
    proto_path = meta.get("prototype_path")
    if not proto_path:
        raise ValueError("Saved artifact does not have a prototype_path; this is only valid for ProtoHedge runs.")
    if "prototype_weights" not in result:
        raise ValueError("Result does not include prototype_weights. Re-evaluate the ProtoHedge artifact on a split.")

    payload = load_prototype_payload(proto_path)
    prototypes = np.asarray(payload["prototypes"], dtype=np.float32)
    scaler = payload.get("scaler")
    weights = _np(result["prototype_weights"])
    usage_mean = weights.mean(axis=(0, 1))
    top1 = weights.argmax(axis=2).reshape(-1)
    top1_freq = np.bincount(top1, minlength=prototypes.shape[0]).astype(np.float64)
    top1_freq = top1_freq / max(1.0, top1_freq.sum())

    raw_prototypes = scaler.inverse_transform(prototypes) if scaler is not None else prototypes
    feature_names = payload.get("feature_names", meta.get("config", {}).get("model", {}).get("feature_list"))
    feature_cols = _expanded_feature_columns(feature_names, bundle["world"], raw_prototypes.shape[1])

    proto_actions = _np(bundle["gym"].agent.prototype_actions)
    rows = []
    for idx in range(raw_prototypes.shape[0]):
        row = {
            "prototype_index": int(idx),
            "usage_mean": float(usage_mean[idx]),
            "top1_frequency": float(top1_freq[idx]),
        }
        for j, col in enumerate(feature_cols):
            row[col] = float(raw_prototypes[idx, j])
        for j in range(proto_actions.shape[1]):
            row[f"proto_action_{j}"] = float(proto_actions[idx, j])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["usage_mean", "top1_frequency"], ascending=False).reset_index(drop=True)
    return df.head(int(top_n)).copy(), df


def prototype_usage_by_regime(result, regime_frame, regime_cols=("return_regime", "vol_regime", "drawdown_regime")):
    if "prototype_weights" not in result:
        raise ValueError("Result does not include prototype_weights")
    weights = _np(result["prototype_weights"]).mean(axis=1)
    usage = pd.DataFrame(weights, columns=[f"prototype_{i}" for i in range(weights.shape[1])])
    usage = pd.concat([regime_frame[list(regime_cols)].reset_index(drop=True), usage], axis=1)
    rows = []
    proto_cols = [c for c in usage.columns if c.startswith("prototype_")]
    for regime_col in regime_cols:
        for regime, grp in usage.groupby(regime_col, observed=True):
            means = grp[proto_cols].mean(axis=0)
            for proto_name, val in means.items():
                rows.append({
                    "regime_type": regime_col,
                    "regime": str(regime),
                    "prototype": proto_name,
                    "mean_weight": float(val),
                })
    return pd.DataFrame(rows)


def build_frontier_candidates(summary_df, vanilla_tol_pct=0.02):
    """Return only configurations already frozen by validation selection."""
    summary_df = summary_df.copy()
    del vanilla_tol_pct
    if "selected_for" not in summary_df.columns:
        raise RuntimeError(
            "Test-set frontier construction is disabled. Use a sweep produced "
            "by the validation-only selection protocol."
        )
    proto = summary_df[
        (summary_df.get("model_family", "") == "proto")
        & summary_df["selected_for"].fillna("").ne("")
    ].copy()
    rows = []
    reason_map = {
        "best_screened_proto_mean": "best_screened_mean",
        "best_screened_proto_cvar05": "best_screened_cvar05",
    }
    for _, row in proto.iterrows():
        for selection in str(row["selected_for"]).split("|"):
            if not selection:
                continue
            rows.append(
                {
                    "reason": reason_map.get(selection, selection),
                    "selection_split": "validation",
                    **row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def build_paper_model_table(summary_df, frontier_df):
    summary_df = summary_df.copy()
    frontier_df = frontier_df.copy()
    vanilla = summary_df[summary_df["model"] == "vanilla"]
    if vanilla.empty:
        raise ValueError("summary_df does not contain a vanilla row")
    vanilla_row = vanilla.iloc[0]
    vanilla_mean = float(vanilla_row["liability_offset_mean_avg"])
    vanilla_cvar = float(vanilla_row["liability_offset_cvar05_avg"])
    vanilla_rmse = float(vanilla_row["liability_offset_rmse_avg"])
    vanilla_downside = float(
        vanilla_row["liability_offset_downside_deviation_avg"]
    )
    vanilla_occupancy = float(vanilla_row["pct_at_any_position_bound_avg"])

    label_map = {
        "unhedged": ("baseline_unhedged", "Unhedged"),
        "spot_delta": ("baseline_spot_delta", "Spot-Delta"),
        "spot_delta_band": ("baseline_spot_delta_band", "Spot-Delta Band"),
        "vanilla": ("vanilla", "Vanilla DH"),
        "best_screened_mean": (
            "proto_mean",
            "ProtoHedge (Validation-Mean)",
        ),
        "best_screened_cvar05": (
            "proto_tail",
            "ProtoHedge (Validation-Tail)",
        ),
    }

    picks = []
    for model_name in ["unhedged", "spot_delta", "spot_delta_band", "vanilla"]:
        row = summary_df[summary_df["model"] == model_name]
        if row.empty:
            continue
        analysis_label, paper_label = label_map[model_name]
        picks.append({
            "analysis_label": analysis_label,
            "paper_label": paper_label,
            **row.iloc[0].to_dict(),
        })

    if not frontier_df.empty:
        for _, row in frontier_df.iterrows():
            reason = str(row.get("reason"))
            analysis_label, paper_label = label_map.get(reason, ("proto_candidate", f"ProtoHedge ({reason})"))
            picks.append({
                "analysis_label": analysis_label,
                "paper_label": paper_label,
                **row.to_dict(),
            })

    paper_df = pd.DataFrame(picks)
    if paper_df.empty:
        return paper_df

    paper_df = paper_df.drop_duplicates(subset=["model"], keep="first").copy()
    paper_df["mean_gap_vs_vanilla"] = (
        paper_df["liability_offset_mean_avg"] - vanilla_mean
    )
    paper_df["mean_gap_vs_vanilla_pct"] = 100.0 * paper_df["mean_gap_vs_vanilla"] / max(abs(vanilla_mean), 1e-12)
    paper_df["cvar_gap_vs_vanilla"] = (
        paper_df["liability_offset_cvar05_avg"] - vanilla_cvar
    )
    paper_df["rmse_improvement_vs_vanilla"] = (
        vanilla_rmse - paper_df["liability_offset_rmse_avg"]
    )
    paper_df["downside_deviation_improvement_vs_vanilla"] = (
        vanilla_downside
        - paper_df["liability_offset_downside_deviation_avg"]
    )
    paper_df["bound_occupancy_gap_vs_vanilla"] = vanilla_occupancy - paper_df["pct_at_any_position_bound_avg"]

    order = {
        "baseline_unhedged": 0,
        "baseline_spot_delta": 1,
        "baseline_spot_delta_band": 2,
        "vanilla": 3,
        "proto_mean": 4,
        "proto_tail": 5,
        "proto_candidate": 6,
    }
    paper_df["paper_order"] = paper_df["analysis_label"].map(order).fillna(99).astype(int)
    return paper_df.sort_values(
        ["paper_order", "liability_offset_mean_avg"],
        ascending=[True, False],
    ).reset_index(drop=True)


def select_frontier_artifacts_for_all_seeds(artifact_index_df, frontier_df):
    artifact_index_df = artifact_index_df.copy()
    frontier_df = frontier_df.copy()
    if artifact_index_df.empty:
        raise ValueError("artifact_index_df is empty")

    label_map = {
        "vanilla": ("vanilla", "Vanilla DH"),
        "best_screened_mean": (
            "proto_mean",
            "ProtoHedge (Validation-Mean)",
        ),
        "best_screened_cvar05": (
            "proto_tail",
            "ProtoHedge (Validation-Tail)",
        ),
    }

    rows = []
    vanilla_rows = artifact_index_df[artifact_index_df["model_family"] == "vanilla"].copy()
    for _, row in vanilla_rows.iterrows():
        analysis_label, paper_label = label_map["vanilla"]
        rows.append({
            "analysis_label": analysis_label,
            "paper_label": paper_label,
            "reason": "vanilla",
            **row.to_dict(),
        })

    for _, frontier_row in frontier_df.iterrows():
        model_name = frontier_row.get("model")
        if not model_name:
            continue
        matched = artifact_index_df[artifact_index_df["model_name"] == model_name].copy()
        if matched.empty:
            continue
        analysis_label, paper_label = label_map.get(
            str(frontier_row.get("reason")),
            ("proto_candidate", f"ProtoHedge ({frontier_row.get('reason')})"),
        )
        for _, row in matched.iterrows():
            rows.append({
                "analysis_label": analysis_label,
                "paper_label": paper_label,
                "reason": frontier_row.get("reason"),
                **row.to_dict(),
            })

    artifact_df = pd.DataFrame(rows)
    if artifact_df.empty:
        return artifact_df
    artifact_df = artifact_df.drop_duplicates(subset=["analysis_label", "seed", "artifact_dir"]).copy()
    order = {
        "vanilla": 0,
        "proto_mean": 1,
        "proto_tail": 2,
        "proto_candidate": 3,
    }
    artifact_df["paper_order"] = artifact_df["analysis_label"].map(order).fillna(99).astype(int)
    return artifact_df.sort_values(["seed", "paper_order", "model_name"]).reset_index(drop=True)


def paired_result_stats(result_a, result_b, label_a, label_b):
    offset_a = _np(
        result_a.get("liability_offset", result_a["gains"])
    ).reshape(-1)
    offset_b = _np(
        result_b.get("liability_offset", result_b["gains"])
    ).reshape(-1)
    if offset_a.shape != offset_b.shape:
        raise ValueError(
            "Result shapes do not match for paired comparison: "
            f"{offset_a.shape} vs {offset_b.shape}"
        )

    diff = offset_a - offset_b
    metrics_a = summarize_liability_offset(offset_a)
    metrics_b = summarize_liability_offset(offset_b)

    return {
        "comparison": f"{label_a}_minus_{label_b}",
        "lhs": str(label_a),
        "rhs": str(label_b),
        "outcome_definition": OUTCOME_DEFINITION_VERSION,
        "n_paths": int(diff.shape[0]),
        "liability_offset_mean_difference": float(diff.mean()),
        "liability_offset_median_difference": float(np.median(diff)),
        "liability_offset_p05_difference": float(np.quantile(diff, 0.05)),
        "higher_offset_rate": float(np.mean(diff > 0.0)),
        "tie_rate": float(np.mean(np.isclose(diff, 0.0, atol=1e-8, rtol=0.0))),
        "lhs_liability_offset_mean": metrics_a["liability_offset_mean"],
        "rhs_liability_offset_mean": metrics_b["liability_offset_mean"],
        "lhs_liability_offset_cvar05": metrics_a[
            "liability_offset_cvar05"
        ],
        "rhs_liability_offset_cvar05": metrics_b[
            "liability_offset_cvar05"
        ],
        "liability_offset_cvar05_difference": (
            metrics_a["liability_offset_cvar05"]
            - metrics_b["liability_offset_cvar05"]
        ),
        "liability_offset_rmse_improvement": (
            metrics_b["liability_offset_rmse"]
            - metrics_a["liability_offset_rmse"]
        ),
        "liability_offset_downside_deviation_improvement": (
            metrics_b["liability_offset_downside_deviation"]
            - metrics_a["liability_offset_downside_deviation"]
        ),
    }


def aggregate_regime_summaries(regime_summary_df):
    regime_summary_df = regime_summary_df.copy()
    required = {"analysis_label", "paper_label", "regime_type", "regime", "series", "seed"}
    missing = required.difference(regime_summary_df.columns)
    if missing:
        raise ValueError(f"regime_summary_df is missing required columns: {sorted(missing)}")

    grouped = regime_summary_df.groupby(
        ["analysis_label", "paper_label", "regime_type", "regime", "series"],
        observed=True,
    )
    agg = grouped.agg(
        n_seeds=("seed", "nunique"),
        mean_avg=("mean", "mean"),
        mean_std=("mean", "std"),
        p05_avg=("p05", "mean"),
        p05_std=("p05", "std"),
        cvar05_avg=("cvar05", "mean"),
        cvar05_std=("cvar05", "std"),
        mae_avg=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_avg=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        downside_deviation_avg=("downside_deviation", "mean"),
        downside_deviation_std=("downside_deviation", "std"),
        negative_offset_rate_avg=("negative_offset_rate", "mean"),
        negative_offset_rate_std=("negative_offset_rate", "std"),
    ).reset_index()
    return agg.sort_values(["regime_type", "regime", "paper_label"]).reset_index(drop=True)

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


ARTIFACT_META = "artifact_metadata.json"
ARTIFACT_STATE = "gym_state.pt"


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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
        "split_indices": {k: [int(i) for i in v] for k, v in split_indices.items()},
        "config": result.get("config", {}),
        "history": {
            "best_epoch": result.get("history", {}).get("best_epoch"),
            "best_score": result.get("history", {}).get("best_score"),
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

    components = build_experiment_components(meta["config"])
    state_dict = torch.load(artifact_dir / ARTIFACT_STATE, map_location=map_location)
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
        selection_metric="gains_mean",
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
        frame[f"{name}_gains"] = _np(result["gains"]).reshape(-1)
        frame[f"{name}_pnl"] = _np(result["pnl"]).reshape(-1)
        frame[f"{name}_cost"] = _np(result["cost"]).reshape(-1)
    return frame


def summarize_by_regime(frame, regime_col, series_cols=None):
    if series_cols is None:
        series_cols = [c for c in frame.columns if c.endswith("_gains") or c.endswith("_payoff")]
    rows = []
    for regime, grp in frame.groupby(regime_col, observed=True):
        for col in series_cols:
            values = grp[col].to_numpy(dtype=float)
            p05 = float(np.quantile(values, 0.05))
            cvar05 = float(values[values <= p05].mean()) if np.any(values <= p05) else p05
            rows.append({
                "regime_type": regime_col,
                "regime": str(regime),
                "series": col,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "p05": p05,
                "cvar05": cvar05,
                "shortfall_prob": float(np.mean(values < 0.0)),
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
    summary_df = summary_df.copy()
    vanilla = summary_df[summary_df["model"] == "vanilla"]
    if vanilla.empty:
        raise ValueError("summary_df does not contain a vanilla row")
    vanilla_mean = float(vanilla.iloc[0]["gains_mean_avg"])
    proto = summary_df[summary_df["model_family"] == "proto"].copy()
    screened = proto[proto.get("passes_robust_screen", True) == True].copy()
    near_vanilla = screened[screened["gains_mean_avg"] >= vanilla_mean * (1.0 - float(vanilla_tol_pct))].copy()

    picks = []
    if not screened.empty:
        picks.append(("best_screened_mean", screened.sort_values("gains_mean_avg", ascending=False).iloc[0]))
        picks.append(("best_screened_cvar05", screened.sort_values("gains_cvar05_avg", ascending=False).iloc[0]))
        picks.append(("best_screened_shortfall", screened.sort_values("shortfall_prob_avg", ascending=True).iloc[0]))
    for _, row in near_vanilla.sort_values("gains_mean_avg", ascending=False).iterrows():
        picks.append(("near_vanilla", row))

    seen = set()
    rows = []
    for reason, row in picks:
        key = str(row["model"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"reason": reason, **row.to_dict()})
    return pd.DataFrame(rows)


def build_paper_model_table(summary_df, frontier_df):
    summary_df = summary_df.copy()
    frontier_df = frontier_df.copy()
    vanilla = summary_df[summary_df["model"] == "vanilla"]
    if vanilla.empty:
        raise ValueError("summary_df does not contain a vanilla row")
    vanilla_row = vanilla.iloc[0]
    vanilla_mean = float(vanilla_row["gains_mean_avg"])
    vanilla_cvar = float(vanilla_row["gains_cvar05_avg"])
    vanilla_shortfall = float(vanilla_row["shortfall_prob_avg"])
    vanilla_occupancy = float(vanilla_row["pct_at_any_position_bound_avg"])

    label_map = {
        "unhedged": ("baseline_unhedged", "Unhedged"),
        "spot_delta": ("baseline_spot_delta", "Spot-Delta"),
        "spot_delta_band": ("baseline_spot_delta_band", "Spot-Delta Band"),
        "vanilla": ("vanilla", "Vanilla DH"),
        "best_screened_mean": ("proto_mean", "ProtoHedge (Mean frontier)"),
        "best_screened_cvar05": ("proto_tail", "ProtoHedge (Tail frontier)"),
        "best_screened_shortfall": ("proto_shortfall", "ProtoHedge (Shortfall frontier)"),
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
    paper_df["mean_gap_vs_vanilla"] = paper_df["gains_mean_avg"] - vanilla_mean
    paper_df["mean_gap_vs_vanilla_pct"] = 100.0 * paper_df["mean_gap_vs_vanilla"] / max(abs(vanilla_mean), 1e-12)
    paper_df["cvar_gap_vs_vanilla"] = paper_df["gains_cvar05_avg"] - vanilla_cvar
    paper_df["shortfall_gap_vs_vanilla"] = vanilla_shortfall - paper_df["shortfall_prob_avg"]
    paper_df["bound_occupancy_gap_vs_vanilla"] = vanilla_occupancy - paper_df["pct_at_any_position_bound_avg"]

    order = {
        "baseline_unhedged": 0,
        "baseline_spot_delta": 1,
        "baseline_spot_delta_band": 2,
        "vanilla": 3,
        "proto_mean": 4,
        "proto_tail": 5,
        "proto_shortfall": 6,
        "proto_candidate": 7,
    }
    paper_df["paper_order"] = paper_df["analysis_label"].map(order).fillna(99).astype(int)
    return paper_df.sort_values(["paper_order", "gains_mean_avg"], ascending=[True, False]).reset_index(drop=True)


def select_frontier_artifacts_for_all_seeds(artifact_index_df, frontier_df):
    artifact_index_df = artifact_index_df.copy()
    frontier_df = frontier_df.copy()
    if artifact_index_df.empty:
        raise ValueError("artifact_index_df is empty")

    label_map = {
        "vanilla": ("vanilla", "Vanilla DH"),
        "best_screened_mean": ("proto_mean", "ProtoHedge (Mean frontier)"),
        "best_screened_cvar05": ("proto_tail", "ProtoHedge (Tail frontier)"),
        "best_screened_shortfall": ("proto_shortfall", "ProtoHedge (Shortfall frontier)"),
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
    order = {"vanilla": 0, "proto_mean": 1, "proto_tail": 2, "proto_shortfall": 3, "proto_candidate": 4}
    artifact_df["paper_order"] = artifact_df["analysis_label"].map(order).fillna(99).astype(int)
    return artifact_df.sort_values(["seed", "paper_order", "model_name"]).reset_index(drop=True)


def paired_result_stats(result_a, result_b, label_a, label_b):
    gains_a = _np(result_a["gains"]).reshape(-1)
    gains_b = _np(result_b["gains"]).reshape(-1)
    if gains_a.shape != gains_b.shape:
        raise ValueError(f"Result shapes do not match for paired comparison: {gains_a.shape} vs {gains_b.shape}")

    diff = gains_a - gains_b

    def _cvar05(x):
        p05 = float(np.quantile(x, 0.05))
        return float(x[x <= p05].mean()) if np.any(x <= p05) else p05

    return {
        "comparison": f"{label_a}_minus_{label_b}",
        "lhs": str(label_a),
        "rhs": str(label_b),
        "n_paths": int(diff.shape[0]),
        "mean_gap": float(diff.mean()),
        "median_gap": float(np.median(diff)),
        "p05_gap": float(np.quantile(diff, 0.05)),
        "win_rate": float(np.mean(diff > 0.0)),
        "tie_rate": float(np.mean(np.isclose(diff, 0.0, atol=1e-8, rtol=0.0))),
        "lhs_mean": float(gains_a.mean()),
        "rhs_mean": float(gains_b.mean()),
        "lhs_cvar05": _cvar05(gains_a),
        "rhs_cvar05": _cvar05(gains_b),
        "lhs_shortfall": float(np.mean(gains_a < 0.0)),
        "rhs_shortfall": float(np.mean(gains_b < 0.0)),
        "cvar05_gap": _cvar05(gains_a) - _cvar05(gains_b),
        "shortfall_gap": float(np.mean(gains_b < 0.0) - np.mean(gains_a < 0.0)),
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
        shortfall_prob_avg=("shortfall_prob", "mean"),
        shortfall_prob_std=("shortfall_prob", "std"),
    ).reset_index()
    return agg.sort_values(["regime_type", "regime", "paper_label"]).reset_index(drop=True)

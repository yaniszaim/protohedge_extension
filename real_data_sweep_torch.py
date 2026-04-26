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
from deephedging.prototype_extraction_torch import build_prototype_payload, save_prototype_payload
from deephedging.run_train_torch import run_experiment
from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch


MODEL_FEATURES = ["price", "delta", "time_left"]


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


def split_indices(n_paths, train_frac=0.70, val_frac=0.15, seed=1234, samples=None):
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(int(n_paths))
    if samples is not None:
        samples = int(samples)
        if samples <= 2:
            raise ValueError("samples must leave room for train/val/test splits")
        if samples > n_paths:
            raise ValueError(f"samples={samples} exceeds available paths={n_paths}")
        indices = indices[:samples]

    n = len(indices)
    n_train = max(1, int(np.floor(n * float(train_frac))))
    n_val = max(1, int(np.floor(n * float(val_frac))))
    if n_train + n_val >= n:
        raise ValueError("split fractions leave no test paths")

    return {
        "train": indices[:n_train].astype(np.int64),
        "val": indices[n_train:n_train + n_val].astype(np.int64),
        "test": indices[n_train + n_val:].astype(np.int64),
    }


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
):
    trade_bounds = trade_bounds or {
        "lbnd_as": -5.0,
        "ubnd_as": 5.0,
        "lbnd_av": -5.0,
        "ubnd_av": 5.0,
    }
    cumulative_bounds = cumulative_bounds or {
        "lbnd_delta_s": -2.0,
        "ubnd_delta_s": 2.0,
        "lbnd_delta_v": -2.0,
        "ubnd_delta_v": 2.0,
    }
    cfg = {
        "world_type": "real",
        "data_path": str(data_path),
        "sample_indices": [int(i) for i in indices],
        "shuffle": False,
        "seed": int(seed),
        "normalize": bool(normalize),
        "hedge_mode": str(hedge_mode),
        "position_bounds": bool(position_bounds),
        **trade_bounds,
    }
    if position_bounds:
        cfg.update(cumulative_bounds)
    if val_indices is not None:
        cfg["val_sample_indices"] = [int(i) for i in val_indices]
    return cfg


def training_cfg(epochs=5, lr=1e-3, batch_size=None, epoch_refresh=None):
    epoch_refresh = int(epoch_refresh) if epoch_refresh is not None else max(1, int(epochs))
    return {
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": batch_size,
        "epoch_refresh": max(1, epoch_refresh),
        "clipvalue": 1.0,
        "global_clipnorm": 1.0,
        "lr_decay_factor": None,
        "lr_decay_patience": None,
    }


def objective_cfg(risk_measure="cvar", risk_aversion=1.0):
    return {"risk_measure": str(risk_measure), "risk_aversion": float(risk_aversion)}


def vanilla_model_cfg():
    return {
        "agent_type": "feed_forward",
        "feature_list": MODEL_FEATURES,
        "network_width": 20,
        "network_depth": 3,
        "activation": "softplus",
    }


def proto_model_cfg(prototype_path, weighted_similarity=True, learn_distance_feature_weights=False):
    return {
        "agent_type": "protopnet",
        "prototype_path": str(prototype_path),
        "feature_list": MODEL_FEATURES,
        "weighted_similarity": bool(weighted_similarity),
        "learn_distance_feature_weights": bool(learn_distance_feature_weights),
    }


def _assert_finite(label, result):
    for key in ["payoff", "pnl", "cost", "gains", "actions"]:
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
    gains = _as_numpy(result["gains"]).reshape(-1)
    payoff = _as_numpy(result["payoff"]).reshape(-1)
    pnl = _as_numpy(result["pnl"]).reshape(-1)
    cost = _as_numpy(result["cost"]).reshape(-1)
    actions = _as_numpy(result["actions"])
    deltas = _as_numpy(result["deltas"]) if "deltas" in result else np.cumsum(actions, axis=1)

    var05 = float(np.quantile(gains, 0.05))
    cvar05 = float(gains[gains <= var05].mean()) if np.any(gains <= var05) else var05
    metrics = {
        "gains_mean": float(gains.mean()),
        "gains_std": float(gains.std()),
        "gains_p01": float(np.quantile(gains, 0.01)),
        "gains_p05": var05,
        "gains_cvar05": cvar05,
        "gains_p50": float(np.quantile(gains, 0.50)),
        "gains_p95": float(np.quantile(gains, 0.95)),
        "shortfall_prob": float(np.mean(gains < 0.0)),
        "payoff_mean": float(payoff.mean()),
        "pnl_mean": float(pnl.mean()),
        "cost_mean": float(cost.mean()),
        "action_abs_mean": float(np.mean(np.abs(actions))),
        "action_abs_max": float(np.max(np.abs(actions))),
        "delta_abs_mean": float(np.mean(np.abs(deltas))),
        "delta_abs_max": float(np.max(np.abs(deltas))),
    }
    metrics.update(_position_bound_metrics(deltas=deltas, market=market))
    return metrics


def evaluate_gym(gym, data_path, indices, label, world_kwargs):
    world = RealWorld_Spot_ATM_Torch(
        world_cfg(data_path=data_path, indices=indices, **world_kwargs)
    )
    data = torchCast(world.torch_data)
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
        "gains": payoff.copy(),
        "actions": np.zeros((n_paths, n_steps, n_inst), dtype=np.float32),
        "deltas": np.zeros((n_paths, n_steps, n_inst), dtype=np.float32),
    }
    _assert_finite("unhedged", result)
    return world, result, summarize_result(result, market=world.data.market)


def evaluate_spot_delta(data_path, indices, world_kwargs):
    world = RealWorld_Spot_ATM_Torch(
        world_cfg(data_path=data_path, indices=indices, **world_kwargs)
    )
    market = world.data.market
    hedges = np.asarray(market.hedges, dtype=np.float32)
    trading_cost = np.asarray(market.cost, dtype=np.float32)
    payoff = np.asarray(market.payoff, dtype=np.float32)
    ubnd_a = np.asarray(market.ubnd_a, dtype=np.float32)
    lbnd_a = np.asarray(market.lbnd_a, dtype=np.float32)
    ubnd_delta = np.asarray(market.ubnd_delta, dtype=np.float32) if "ubnd_delta" in market else None
    lbnd_delta = np.asarray(market.lbnd_delta, dtype=np.float32) if "lbnd_delta" in market else None
    call_delta = np.asarray(world.data.features.per_step["call_delta"], dtype=np.float32)

    n_paths, n_steps, n_inst = hedges.shape
    delta = np.zeros((n_paths, n_inst), dtype=np.float32)
    actions = np.zeros((n_paths, n_steps, n_inst), dtype=np.float32)
    deltas = np.zeros((n_paths, n_steps, n_inst), dtype=np.float32)
    pnl = np.zeros((n_paths,), dtype=np.float32)
    cost = np.zeros((n_paths,), dtype=np.float32)

    for t in range(n_steps):
        target = np.zeros_like(delta)
        target[:, 0] = call_delta[:, t]
        if ubnd_delta is not None and lbnd_delta is not None:
            target = np.minimum(np.maximum(target, lbnd_delta[:, t, :]), ubnd_delta[:, t, :])
        action = np.minimum(np.maximum(target - delta, lbnd_a[:, t, :]), ubnd_a[:, t, :])
        if ubnd_delta is not None and lbnd_delta is not None:
            bounded_delta = np.minimum(np.maximum(delta + action, lbnd_delta[:, t, :]), ubnd_delta[:, t, :])
            action = bounded_delta - delta
        pnl += np.sum(action * hedges[:, t, :], axis=1)
        cost += np.sum(np.abs(action) * trading_cost[:, t, :], axis=1)
        delta += action
        actions[:, t, :] = action
        deltas[:, t, :] = delta

    result = {"payoff": payoff, "pnl": pnl, "cost": cost, "gains": payoff + pnl - cost, "actions": actions, "deltas": deltas}
    _assert_finite("spot_delta", result)
    return world, result, summarize_result(result, market=world.data.market)


def metric_row(model, split, metrics, **kwargs):
    row = {"model": model, "split": split, **kwargs}
    row.update(metrics)
    return row


def run_real_data_sweep(
    data_path="Data/training_paths.npy",
    output_dir=".deephedging_real_runs/real_data_sweep",
    samples=512,
    train_frac=0.70,
    val_frac=0.15,
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
    lr=1e-3,
    batch_size=None,
    epoch_refresh=None,
    max_points=None,
):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_dir = output_dir / "prototypes"
    prototype_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 3 or data.shape[-1] < 5:
        raise ValueError(f"Expected data shape [N,T,>=5], got {data.shape}")

    world_kwargs = {
        "seed": None,  # filled per run
        "normalize": normalize,
        "hedge_mode": hedge_mode,
        "position_bounds": position_bounds,
        "trade_bounds": trade_bounds,
        "cumulative_bounds": cumulative_bounds,
    }

    all_rows = []
    run_records = []

    def checkpoint(stage):
        if not all_rows:
            return pd.DataFrame()

        metrics_df = pd.DataFrame(all_rows)
        metrics_path = output_dir / "sweep_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        with open(output_dir / "sweep_config.json", "w") as f:
            json.dump(
                _jsonify(
                    {
                        "data_path": data_path,
                        "output_dir": output_dir,
                        "samples": samples,
                        "train_frac": train_frac,
                        "val_frac": val_frac,
                        "seeds": list(seeds),
                        "epochs": epochs,
                        "lr": lr,
                        "batch_size": batch_size,
                        "epoch_refresh": epoch_refresh,
                        "max_points": max_points,
                        "prototype_counts": list(prototype_counts),
                        "prototype_sources": list(prototype_sources),
                        "weighted_similarity_options": list(weighted_similarity_options),
                        "learn_distance_feature_weights_options": list(learn_distance_feature_weights_options),
                        "risk_measures": list(risk_measures),
                        "normalize": normalize,
                        "hedge_mode": hedge_mode,
                        "position_bounds": position_bounds,
                        "trade_bounds": trade_bounds,
                        "cumulative_bounds": cumulative_bounds,
                        "checkpoint_stage": stage,
                        "runs": run_records,
                    }
                ),
                f,
                indent=2,
            )

        summarize_sweep(metrics_df, output_dir)
        plot_sweep_results(metrics_df, output_dir)
        print(f"\n[checkpoint:{stage}] saved {len(metrics_df)} metric rows to {metrics_path}")
        return metrics_df

    for seed in seeds:
        splits = split_indices(data.shape[0], train_frac=train_frac, val_frac=val_frac, seed=seed, samples=samples)
        split_dir = output_dir / f"seed_{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez(split_dir / "splits.npz", **splits)
        seed_world_kwargs = {**world_kwargs, "seed": int(seed)}

        # Data-only baselines.
        baseline_artifacts = {}
        for split_name, idx in splits.items():
            _, unhedged_result, unhedged_metrics = evaluate_unhedged(data_path, idx, seed_world_kwargs)
            all_rows.append(metric_row("unhedged", split_name, unhedged_metrics, seed=seed, model_family="baseline"))

            _, spot_delta_result, spot_delta_metrics = evaluate_spot_delta(data_path, idx, seed_world_kwargs)
            baseline_artifacts[split_name] = {"spot_delta": spot_delta_result}
            all_rows.append(metric_row("spot_delta", split_name, spot_delta_metrics, seed=seed, model_family="baseline"))

        run_records.append({"seed": int(seed), "model": "data_baselines"})
        checkpoint(f"seed_{seed}_baselines")

        for risk_measure in risk_measures:
            print(f"\n=== seed={seed} risk={risk_measure} vanilla ===")
            train_world_cfg = world_cfg(
                data_path=data_path,
                indices=splits["train"],
                val_indices=splits["val"],
                **seed_world_kwargs,
            )
            vanilla = run_experiment(
                override_world=train_world_cfg,
                override_model=vanilla_model_cfg(),
                override_objective=objective_cfg(risk_measure=risk_measure),
                override_training=training_cfg(
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    epoch_refresh=epoch_refresh,
                ),
            )

            for split_name, idx in splits.items():
                _, _, metrics = evaluate_gym(vanilla["gym"], data_path, idx, f"vanilla {split_name}", seed_world_kwargs)
                all_rows.append(
                    metric_row(
                        "vanilla",
                        split_name,
                        metrics,
                        seed=seed,
                        risk_measure=risk_measure,
                        model_family="vanilla",
                    )
                )

            run_records.append({"seed": int(seed), "risk_measure": risk_measure, "model": "vanilla"})
            checkpoint(f"seed_{seed}_risk_{risk_measure}_vanilla")

            source_results = {
                "vanilla": vanilla["training_result"],
                "spot_delta": baseline_artifacts["train"]["spot_delta"],
                "zero": None,
            }

            for source in prototype_sources:
                if source not in source_results:
                    raise ValueError(f"Unknown prototype source '{source}'")
                for n_prototypes in prototype_counts:
                    payload = build_prototype_payload(
                        world=vanilla["world"],
                        result=source_results[source],
                        n_prototypes=int(n_prototypes),
                        feature_names=MODEL_FEATURES,
                        random_state=int(seed),
                        max_points=max_points,
                    )
                    prototype_path = prototype_dir / (
                        f"seed_{seed}_risk_{risk_measure}_source_{source}_k_{n_prototypes}.pkl"
                    )
                    save_prototype_payload(payload, prototype_path)

                    for weighted in weighted_similarity_options:
                        for learn_weights in learn_distance_feature_weights_options:
                            model_name = (
                                f"proto_source={source}_k={n_prototypes}_"
                                f"weighted={int(bool(weighted))}_learnw={int(bool(learn_weights))}"
                            )
                            print(f"\n=== {model_name} seed={seed} risk={risk_measure} ===")
                            proto = run_experiment(
                                override_world=train_world_cfg,
                                override_model=proto_model_cfg(
                                    prototype_path,
                                    weighted_similarity=weighted,
                                    learn_distance_feature_weights=learn_weights,
                                ),
                                override_objective=objective_cfg(risk_measure=risk_measure),
                                override_training=training_cfg(
                                    epochs=epochs,
                                    lr=lr,
                                    batch_size=batch_size,
                                    epoch_refresh=epoch_refresh,
                                ),
                            )

                            proto_actions = proto["model"].prototype_actions.detach().cpu().numpy()
                            proto_action_norm_mean = float(np.linalg.norm(proto_actions, axis=1).mean())
                            proto_action_norm_max = float(np.linalg.norm(proto_actions, axis=1).max())

                            for split_name, idx in splits.items():
                                _, _, metrics = evaluate_gym(proto["gym"], data_path, idx, model_name, seed_world_kwargs)
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
                                        learn_distance_feature_weights=bool(learn_weights),
                                        proto_action_norm_mean=proto_action_norm_mean,
                                        proto_action_norm_max=proto_action_norm_max,
                                    )
                                )

                            run_records.append(
                                {
                                    "seed": int(seed),
                                    "risk_measure": risk_measure,
                                    "prototype_source": source,
                                    "n_prototypes": int(n_prototypes),
                                    "weighted_similarity": bool(weighted),
                                    "learn_distance_feature_weights": bool(learn_weights),
                                    "prototype_path": str(prototype_path),
                                }
                            )

                            checkpoint(
                                f"seed_{seed}_risk_{risk_measure}_source_{source}_"
                                f"k_{n_prototypes}_weighted_{int(bool(weighted))}_"
                                f"learnw_{int(bool(learn_weights))}"
                            )

    metrics_df = checkpoint("complete")
    print(f"\nsaved sweep metrics: {output_dir / 'sweep_metrics.csv'}")
    return metrics_df


def summarize_sweep(metrics_df, output_dir):
    output_dir = Path(output_dir)
    test = metrics_df[metrics_df["split"] == "test"].copy()
    group_cols = [
        "model",
        "model_family",
        "prototype_source",
        "n_prototypes",
        "weighted_similarity",
        "learn_distance_feature_weights",
        "risk_measure",
    ]
    group_cols = [c for c in group_cols if c in test.columns]
    summary = (
        test.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("gains_mean", "size"),
            gains_mean_avg=("gains_mean", "mean"),
            gains_mean_std=("gains_mean", "std"),
            gains_p05_avg=("gains_p05", "mean"),
            gains_cvar05_avg=("gains_cvar05", "mean"),
            shortfall_prob_avg=("shortfall_prob", "mean"),
            delta_abs_max_avg=("delta_abs_max", "mean"),
            action_abs_mean_avg=("action_abs_mean", "mean"),
            pct_at_any_position_bound_avg=("pct_at_any_position_bound", "mean"),
            pct_at_upper_position_bound_avg=("pct_at_upper_position_bound", "mean"),
            pct_at_lower_position_bound_avg=("pct_at_lower_position_bound", "mean"),
            pct_paths_touch_any_position_bound_avg=("pct_paths_touch_any_position_bound", "mean"),
        )
        .reset_index()
        .sort_values(["gains_mean_avg", "gains_cvar05_avg"], ascending=False)
    )
    summary.to_csv(output_dir / "sweep_test_summary.csv", index=False)
    return summary


def build_paper_report(metrics_df, output_dir):
    """
    Build compact tables for writeups and decision-making.

    The raw sweep CSV is intentionally long. These artifacts answer the actual
    research questions: which model wins on mean, which wins on downside risk,
    and how far each candidate is from the relevant baselines.
    """
    output_dir = Path(output_dir)
    summary = summarize_sweep(metrics_df, output_dir).copy()

    def _baseline_value(model_name, metric, risk_measure=None):
        rows = summary[summary["model"] == model_name]
        if risk_measure is not None and "risk_measure" in rows.columns:
            same_risk = rows[rows["risk_measure"] == risk_measure]
            if not same_risk.empty:
                rows = same_risk
        if rows.empty:
            return np.nan
        return float(rows.iloc[0][metric])

    for metric in ["gains_mean_avg", "gains_cvar05_avg", "shortfall_prob_avg"]:
        for baseline in ["unhedged", "spot_delta", "vanilla"]:
            col = f"{metric}_minus_{baseline}"
            values = []
            for _, row in summary.iterrows():
                baseline_value = _baseline_value(baseline, metric, row.get("risk_measure"))
                if metric == "shortfall_prob_avg":
                    # Lower shortfall is better, so positive means improvement.
                    values.append(baseline_value - float(row[metric]))
                else:
                    values.append(float(row[metric]) - baseline_value)
            summary[col] = values

    summary.to_csv(output_dir / "paper_model_comparison.csv", index=False)

    best_rows = []
    if not summary.empty:
        best_rows.append(("best_mean", summary.sort_values("gains_mean_avg", ascending=False).iloc[0]))
        best_rows.append(("best_cvar05", summary.sort_values("gains_cvar05_avg", ascending=False).iloc[0]))
        best_rows.append(("best_shortfall", summary.sort_values("shortfall_prob_avg", ascending=True).iloc[0]))

        proto = summary[summary["model_family"] == "proto"]
        if not proto.empty:
            best_rows.append(("best_proto_mean", proto.sort_values("gains_mean_avg", ascending=False).iloc[0]))
            best_rows.append(("best_proto_cvar05", proto.sort_values("gains_cvar05_avg", ascending=False).iloc[0]))
            best_rows.append(("best_proto_shortfall", proto.sort_values("shortfall_prob_avg", ascending=True).iloc[0]))

    if best_rows:
        best_df = pd.DataFrame([{"selection": name, **row.to_dict()} for name, row in best_rows])
    else:
        best_df = pd.DataFrame()
    best_df.to_csv(output_dir / "paper_best_models.csv", index=False)

    selected_models = ["unhedged", "spot_delta", "vanilla"]
    if not best_df.empty:
        selected_models.extend(best_df["model"].astype(str).tolist())
    paper_table = summary[summary["model"].astype(str).isin(dict.fromkeys(selected_models).keys())].copy()
    paper_table = paper_table.sort_values(["gains_mean_avg", "gains_cvar05_avg"], ascending=False)
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
        "This report is generated from `sweep_metrics.csv`.",
        "",
    ]

    if summary.empty:
        lines.append("No test summary rows were available.")
        (output_dir / "interpretation.md").write_text("\n".join(lines))
        return

    baselines = summary[summary["model"].isin(["unhedged", "spot_delta", "vanilla"])].copy()
    if not baselines.empty:
        lines.extend(["## Baselines", ""])
        for _, row in baselines.iterrows():
            line = (
                f"- `{row['model']}`: mean `{_fmt(row['gains_mean_avg'])}`, "
                f"CVaR5 `{_fmt(row['gains_cvar05_avg'])}`, "
                f"shortfall `{_fmt(row['shortfall_prob_avg'])}`"
            )
            if "pct_at_any_position_bound_avg" in row.index:
                line += f", bound occupancy `{_fmt(row['pct_at_any_position_bound_avg'])}`"
            lines.append(line)
        lines.append("")

    proto = summary[summary["model_family"] == "proto"].copy()
    if not proto.empty:
        best_mean = proto.sort_values("gains_mean_avg", ascending=False).iloc[0]
        best_tail = proto.sort_values("gains_cvar05_avg", ascending=False).iloc[0]
        best_shortfall = proto.sort_values("shortfall_prob_avg", ascending=True).iloc[0]
        lines.extend(["## Best ProtoHedge Configurations", ""])
        lines.append(
            f"- Best mean: `{best_mean['model']}` with mean `{_fmt(best_mean['gains_mean_avg'])}` "
            f"and CVaR5 `{_fmt(best_mean['gains_cvar05_avg'])}`."
        )
        lines.append(
            f"- Best downside tail: `{best_tail['model']}` with CVaR5 `{_fmt(best_tail['gains_cvar05_avg'])}` "
            f"and mean `{_fmt(best_tail['gains_mean_avg'])}`."
        )
        lines.append(
            f"- Lowest shortfall probability: `{best_shortfall['model']}` with shortfall "
            f"`{_fmt(best_shortfall['shortfall_prob_avg'])}`."
        )
        lines.append("")

        vanilla_mean = summary.loc[summary["model"] == "vanilla", "gains_mean_avg"]
        unhedged_mean = summary.loc[summary["model"] == "unhedged", "gains_mean_avg"]
        spot_mean = summary.loc[summary["model"] == "spot_delta", "gains_mean_avg"]
        if not vanilla_mean.empty:
            lines.append(
                f"Best ProtoHedge mean minus vanilla mean: "
                f"`{_fmt(best_mean['gains_mean_avg'] - float(vanilla_mean.iloc[0]))}`."
            )
        if not unhedged_mean.empty:
            lines.append(
                f"Best ProtoHedge mean minus unhedged mean: "
                f"`{_fmt(best_mean['gains_mean_avg'] - float(unhedged_mean.iloc[0]))}`."
            )
        if not spot_mean.empty:
            lines.append(
                f"Best ProtoHedge mean minus spot-delta mean: "
                f"`{_fmt(best_mean['gains_mean_avg'] - float(spot_mean.iloc[0]))}`."
            )
        if "pct_at_any_position_bound_avg" in best_mean.index:
            lines.append(
                f"Best ProtoHedge bound occupancy: "
                f"`{_fmt(best_mean['pct_at_any_position_bound_avg'])}`."
            )
        if "pct_paths_touch_any_position_bound_avg" in best_mean.index:
            lines.append(
                f"Best ProtoHedge path touch rate: "
                f"`{_fmt(best_mean['pct_paths_touch_any_position_bound_avg'])}`."
            )
        if "pct_at_any_position_bound_avg" in summary.columns:
            vanilla_bounds = summary.loc[summary["model"] == "vanilla", "pct_at_any_position_bound_avg"]
            if not vanilla_bounds.empty:
                lines.append(
                    f"Vanilla bound occupancy: "
                    f"`{_fmt(float(vanilla_bounds.iloc[0]))}`."
                )
        lines.append("")

    lines.extend(
        [
            "## Larger Picture",
            "",
            "This sweep is the empirical bridge between the PyTorch port and the paper extension. "
            "A positive result means more than training code running: it means a compact prototype "
            "policy can compete with neural Deep Hedging and simple classical hedges on held-out "
            "historical paths while remaining inspectable.",
            "",
            "Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data "
            "runs, multiple seeds, and robustness checks across prototype counts and market regimes.",
        ]
    )

    (output_dir / "interpretation.md").write_text("\n".join(lines))


def plot_sweep_results(metrics_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = metrics_df[metrics_df["split"] == "test"].copy()
    if test.empty:
        return

    summary = summarize_sweep(metrics_df, output_dir)
    build_paper_report(metrics_df, output_dir)
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

    _bar("gains_mean_avg", "sweep_test_gains_mean.png", "Test Mean Gains By Model", "mean gains")
    _bar("gains_cvar05_avg", "sweep_test_cvar05.png", "Test 5% CVaR By Model", "5% CVaR gains")
    _bar("shortfall_prob_avg", "sweep_test_shortfall_prob.png", "Test Shortfall Probability", "P(gains < 0)")
    if "pct_at_any_position_bound_avg" in top.columns:
        _bar(
            "pct_at_any_position_bound_avg",
            "sweep_test_bound_saturation.png",
            "Test Position-Bound Occupancy By Model",
            "fraction of delta states at a bound",
        )

    proto = test[test.get("model_family", "") == "proto"].copy()
    if not proto.empty and {"n_prototypes", "prototype_source", "weighted_similarity"}.issubset(proto.columns):
        fig, ax = plt.subplots(figsize=(10, 5))
        for (source, weighted), grp in proto.groupby(["prototype_source", "weighted_similarity"], dropna=False):
            curve = (
                grp.groupby("n_prototypes")
                .agg(gains_mean=("gains_mean", "mean"), gains_cvar05=("gains_cvar05", "mean"))
                .reset_index()
                .sort_values("n_prototypes")
            )
            label = f"{source}, weighted={int(bool(weighted))}"
            ax.plot(curve["n_prototypes"], curve["gains_mean"], marker="o", label=label)
        ax.set_title("ProtoHedge Test Mean Gains vs Prototype Count")
        ax.set_xlabel("number of prototypes")
        ax.set_ylabel("test mean gains")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "proto_gains_vs_n_prototypes.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for (source, weighted), grp in proto.groupby(["prototype_source", "weighted_similarity"], dropna=False):
            curve = (
                grp.groupby("n_prototypes")
                .agg(gains_cvar05=("gains_cvar05", "mean"))
                .reset_index()
                .sort_values("n_prototypes")
            )
            label = f"{source}, weighted={int(bool(weighted))}"
            ax.plot(curve["n_prototypes"], curve["gains_cvar05"], marker="o", label=label)
        ax.set_title("ProtoHedge Test 5% CVaR vs Prototype Count")
        ax.set_xlabel("number of prototypes")
        ax.set_ylabel("test 5% CVaR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "proto_cvar05_vs_n_prototypes.png", dpi=160)
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
    parser.add_argument("--data-path", default="Data/training_paths.npy")
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch-refresh", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    run_real_data_sweep(
        data_path=args.data_path,
        output_dir=args.output_dir,
        samples=args.samples,
        epochs=args.epochs,
        seeds=tuple(args.seeds),
        prototype_counts=tuple(args.n_prototypes),
        prototype_sources=tuple(args.prototype_sources),
        weighted_similarity_options=_parse_bool_list(args.weighted),
        learn_distance_feature_weights_options=_parse_bool_list(args.learn_weights),
        risk_measures=tuple(args.risk_measures),
        lr=args.lr,
        epoch_refresh=args.epoch_refresh,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()

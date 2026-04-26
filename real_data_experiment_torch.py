"""
First controlled real-data pilot for the PyTorch ProtoHedge pipeline.

This is not the final research experiment. It is the scaffold we want before
that: explicit train/validation/test splits, prototype extraction from the
training split only, vanilla-vs-prototype training, and finite holdout metrics.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


if __package__ in [None, ""]:
    package_root = Path(__file__).resolve().parent
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

from deephedging.base_torch import torchCast
from deephedging.prototype_extraction_torch import build_prototype_payload, save_prototype_payload
from deephedging.run_train_torch import run_experiment
from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch


REQUIRED_RESULT_KEYS = ["utility", "utility0", "pnl", "gains", "actions", "payoff"]
SUMMARY_KEYS = ["utility", "utility0", "gains", "payoff", "pnl", "cost"]


def _resolve_existing_path(path):
    path = Path(path)
    package_root = Path(__file__).resolve().parent
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([package_root / path, package_root / "Data" / path, package_root / "Data" / path.name])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find path '{path}'")


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _validate_data_file(data_path):
    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 3 or data.shape[-1] < 5:
        raise AssertionError(f"Expected real data shape [N, T, >=5], got {data.shape}")
    if not np.isfinite(data).all():
        raise AssertionError(f"Real data contains NaN or infinite values: {data_path}")
    return data.shape


def _split_indices(n_paths, train_frac, val_frac, seed, samples=None):
    n_paths = int(n_paths)
    train_frac = float(train_frac)
    val_frac = float(val_frac)

    if train_frac <= 0.0 or val_frac <= 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError("Require train_frac > 0, val_frac > 0, and train_frac + val_frac < 1")

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(n_paths)
    if samples is not None:
        samples = int(samples)
        if samples <= 2:
            raise ValueError("--samples must leave room for train, validation, and test splits")
        if samples > n_paths:
            raise ValueError(f"--samples={samples} exceeds available paths {n_paths}")
        indices = indices[:samples]

    n_selected = int(indices.shape[0])
    n_train = max(1, int(np.floor(n_selected * train_frac)))
    n_val = max(1, int(np.floor(n_selected * val_frac)))
    if n_train + n_val >= n_selected:
        raise ValueError("Split fractions leave no test paths; increase --samples or reduce fractions")

    return {
        "train": indices[:n_train].astype(np.int64),
        "val": indices[n_train:n_train + n_val].astype(np.int64),
        "test": indices[n_train + n_val:].astype(np.int64),
    }


def _world_cfg(data_path, indices, val_indices=None, seed=1234):
    cfg = {
        "world_type": "real",
        "data_path": str(data_path),
        "sample_indices": [int(i) for i in indices],
        "shuffle": False,
        "seed": int(seed),
    }
    if val_indices is not None:
        cfg["val_sample_indices"] = [int(i) for i in val_indices]
    return cfg


def _training_cfg(epochs, batch_size, lr):
    return {
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": batch_size,
        "epoch_refresh": max(1, int(epochs)),
        "clipvalue": 1.0,
        "global_clipnorm": 1.0,
        "lr_decay_factor": None,
        "lr_decay_patience": None,
    }


def _objective_cfg():
    return {"risk_measure": "cvar", "risk_aversion": 1.0}


def _vanilla_model_cfg():
    return {
        "agent_type": "feed_forward",
        "feature_list": ["price", "delta", "time_left"],
        "network_width": 20,
        "network_depth": 3,
        "activation": "softplus",
    }


def _proto_model_cfg(prototype_path):
    return {
        "agent_type": "protopnet",
        "prototype_path": str(prototype_path),
        "feature_list": ["price", "delta", "time_left"],
        "weighted_similarity": True,
        "learn_distance_feature_weights": False,
    }


def _assert_finite_result(label, result):
    for key in REQUIRED_RESULT_KEYS:
        if key not in result:
            raise AssertionError(f"{label}: missing result['{key}']")
        values = _as_numpy(result[key])
        if not np.isfinite(values).all():
            raise AssertionError(f"{label}: non-finite values in result['{key}']")


def _summarize_result(result):
    metrics = {}
    for key in SUMMARY_KEYS:
        if key not in result:
            continue
        values = _as_numpy(result[key]).reshape(-1)
        metrics[f"{key}_mean"] = float(np.mean(values))
        metrics[f"{key}_std"] = float(np.std(values))
        metrics[f"{key}_min"] = float(np.min(values))
        metrics[f"{key}_max"] = float(np.max(values))

    actions = _as_numpy(result["actions"])
    metrics["action_abs_mean"] = float(np.mean(np.abs(actions)))
    metrics["action_abs_max"] = float(np.max(np.abs(actions)))
    metrics["action_shape"] = list(actions.shape)
    return metrics


def _evaluate(gym, data_path, split_name, indices):
    world = RealWorld_Spot_ATM_Torch(_world_cfg(data_path, indices))
    data = torchCast(world.torch_data)

    gym.eval()
    with torch.no_grad():
        result = gym.forward(data, training=False, return_paths=True)

    _assert_finite_result(split_name, result)
    metrics = _summarize_result(result)
    metrics.update(
        {
            "n_paths": int(world.nSamples),
            "n_steps": int(world.nSteps),
            "n_inst": int(world.nInst),
        }
    )
    return metrics


def _write_metrics_csv(path, summary):
    rows = []
    for model_name, split_metrics in summary["metrics"].items():
        for split_name, metrics in split_metrics.items():
            row = {"model": model_name, "split": split_name}
            row.update({k: v for k, v in metrics.items() if not isinstance(v, list)})
            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_real_data_pilot(
    data_path="Data/training_paths.npy",
    output_dir=".deephedging_real_runs/first_real_data_pilot",
    samples=None,
    train_frac=0.70,
    val_frac=0.15,
    epochs=5,
    n_prototypes=25,
    batch_size=None,
    lr=1e-3,
    seed=1234,
    max_points=None,
):
    data_path = _resolve_existing_path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_shape = _validate_data_file(data_path)
    splits = _split_indices(
        n_paths=data_shape[0],
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        samples=samples,
    )
    np.savez(
        output_dir / "splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )

    print(
        "real-data pilot split | "
        f"train={len(splits['train'])} | val={len(splits['val'])} | test={len(splits['test'])}"
    )

    train_world_cfg = _world_cfg(
        data_path=data_path,
        indices=splits["train"],
        val_indices=splits["val"],
        seed=seed,
    )
    training_cfg = _training_cfg(epochs=epochs, batch_size=batch_size, lr=lr)

    print("\n=== training vanilla real-data baseline ===")
    vanilla = run_experiment(
        override_world=train_world_cfg,
        override_model=_vanilla_model_cfg(),
        override_objective=_objective_cfg(),
        override_training=training_cfg,
    )

    prototype_path = output_dir / f"real_train_prototypes_{int(n_prototypes)}.pkl"
    payload = build_prototype_payload(
        world=vanilla["world"],
        result=vanilla["training_result"],
        n_prototypes=int(n_prototypes),
        feature_names=["price", "delta", "time_left"],
        random_state=seed,
        max_points=max_points,
    )
    save_prototype_payload(payload, prototype_path)
    print(
        "real-data prototypes extracted from train split only | "
        f"shape={np.asarray(payload['prototypes']).shape} | path={prototype_path}"
    )

    print("\n=== training ProtoHedge real-data pilot ===")
    proto = run_experiment(
        override_world=train_world_cfg,
        override_model=_proto_model_cfg(prototype_path),
        override_objective=_objective_cfg(),
        override_training=training_cfg,
    )

    torch.save(vanilla["gym"].state_dict(), output_dir / "vanilla_real_state_dict.pt")
    torch.save(proto["gym"].state_dict(), output_dir / "proto_real_state_dict.pt")

    print("\n=== evaluating clean splits ===")
    metrics = {"vanilla": {}, "proto": {}}
    for split_name, indices in splits.items():
        metrics["vanilla"][split_name] = _evaluate(vanilla["gym"], data_path, f"vanilla {split_name}", indices)
        metrics["proto"][split_name] = _evaluate(proto["gym"], data_path, f"proto {split_name}", indices)
        print(
            f"{split_name}: "
            f"vanilla utility={metrics['vanilla'][split_name]['utility_mean']:.6f} | "
            f"proto utility={metrics['proto'][split_name]['utility_mean']:.6f}"
        )

    summary = {
        "data_path": str(data_path),
        "data_shape": list(data_shape),
        "output_dir": str(output_dir),
        "seed": int(seed),
        "samples": None if samples is None else int(samples),
        "split_counts": {name: int(len(indices)) for name, indices in splits.items()},
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "epochs": int(epochs),
        "batch_size": batch_size,
        "lr": float(lr),
        "n_prototypes": int(n_prototypes),
        "prototype_path": str(prototype_path),
        "prototype_shape": list(np.asarray(payload["prototypes"]).shape),
        "prototype_feature_names": payload.get("feature_names"),
        "metrics": metrics,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(_jsonify(summary), f, indent=2, sort_keys=True)
    _write_metrics_csv(output_dir / "metrics.csv", summary)

    print(f"\nreal-data pilot completed | summary={output_dir / 'summary.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="Data/training_paths.npy")
    parser.add_argument("--output-dir", default=".deephedging_real_runs/first_real_data_pilot")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-prototypes", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    run_real_data_pilot(
        data_path=args.data_path,
        output_dir=args.output_dir,
        samples=args.samples,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        epochs=args.epochs,
        n_prototypes=args.n_prototypes,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()

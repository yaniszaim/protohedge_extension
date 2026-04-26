"""
Validate torch prototype extraction on the synthetic worlds.

This is the readiness gate between notebook parity and real-data experiments:
it trains a small vanilla model, extracts prototypes from the train-world
trajectories only, reloads those prototypes into ProtoHedge, and checks that
the resulting model produces finite path outputs.
"""

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


if __package__ in [None, ""]:
    package_root = Path(__file__).resolve().parent
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

from deephedging.prototype_extraction_torch import build_prototype_payload, save_prototype_payload
from deephedging.run_train_torch import run_experiment


REQUIRED_RESULT_KEYS = ["utility", "utility0", "pnl", "gains", "actions", "payoff"]


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _assert_finite_array(label, value):
    arr = _as_numpy(value)
    if not np.isfinite(arr).all():
        raise AssertionError(f"{label}: contains NaN or infinite values")


def _assert_finite_training_result(label, result):
    training_result = result["training_result"]
    for key in REQUIRED_RESULT_KEYS:
        if key not in training_result:
            raise AssertionError(f"{label}: missing training_result['{key}']")
        _assert_finite_array(f"{label} training_result['{key}']", training_result[key])

    print(
        f"{label}: finite outputs | "
        f"utility={_as_numpy(training_result['utility']).mean():.6f} | "
        f"pnl={_as_numpy(training_result['pnl']).mean():.6f}"
    )


def _training_cfg(epochs, batch_size):
    return {
        "epochs": int(epochs),
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


def _expected_input_dim(world):
    n_inst = int(world.nInst)
    price = np.asarray(world.data.features.per_step["price"])
    price_dim = int(price.shape[-1]) if price.ndim == 3 else 1
    return price_dim + n_inst + 1


def _validate_payload(label, payload, expected_n_prototypes, expected_input_dim):
    if "prototypes" not in payload:
        raise AssertionError(f"{label}: payload missing 'prototypes'")
    if "scaler" not in payload:
        raise AssertionError(f"{label}: payload missing 'scaler'")

    prototypes = np.asarray(payload["prototypes"])
    if prototypes.shape != (int(expected_n_prototypes), int(expected_input_dim)):
        raise AssertionError(
            f"{label}: prototype shape {prototypes.shape} != "
            f"{(int(expected_n_prototypes), int(expected_input_dim))}"
        )
    if getattr(payload["scaler"], "mean_", None) is None:
        raise AssertionError(f"{label}: scaler is not fitted")
    if np.asarray(payload["scaler"].mean_).shape[0] != int(expected_input_dim):
        raise AssertionError(f"{label}: scaler dimension does not match prototypes")
    if payload.get("input_dim") != int(expected_input_dim):
        raise AssertionError(f"{label}: input_dim metadata does not match expected dimension")
    if payload.get("feature_names") != ["delta", "price", "time_left"]:
        raise AssertionError(f"{label}: feature_names metadata is not sorted like the agent")
    _assert_finite_array(f"{label} prototypes", prototypes)

    print(
        f"{label}: payload ok | prototypes={prototypes.shape[0]} | "
        f"input_dim={prototypes.shape[1]}"
    )


def _run_case(label, world_cfg, n_prototypes, epochs, batch_size, max_points, tmp_dir):
    print(f"\n=== {label} synthetic prototype extraction ===")
    vanilla = run_experiment(
        override_world=world_cfg,
        override_model=_vanilla_model_cfg(),
        override_objective=_objective_cfg(),
        override_training=_training_cfg(epochs, batch_size),
    )
    _assert_finite_training_result(f"{label} vanilla", vanilla)

    expected_dim = _expected_input_dim(vanilla["world"])
    payload = build_prototype_payload(
        world=vanilla["world"],
        result=vanilla["training_result"],
        n_prototypes=int(n_prototypes),
        feature_names=["price", "delta", "time_left"],
        random_state=0,
        max_points=max_points,
    )
    _validate_payload(label, payload, n_prototypes, expected_dim)

    prototype_path = Path(tmp_dir) / f"{label.replace(' ', '_')}_prototypes.pkl"
    save_prototype_payload(payload, prototype_path)

    proto = run_experiment(
        override_world=world_cfg,
        override_model=_proto_model_cfg(prototype_path),
        override_objective=_objective_cfg(),
        override_training=_training_cfg(epochs, batch_size),
    )
    _assert_finite_training_result(f"{label} proto reload", proto)
    print(f"{label}: reload training ok | prototype_path={prototype_path}")


def run_validation(samples=128, steps=20, epochs=2, n_prototypes=8, batch_size=None, max_points=None):
    with tempfile.TemporaryDirectory(prefix="protohedge_synth_proto_") as tmp:
        _run_case(
            label="black_scholes",
            world_cfg={
                "world_type": "synthetic",
                "samples": int(samples),
                "steps": int(steps),
                "black_scholes": True,
            },
            n_prototypes=n_prototypes,
            epochs=epochs,
            batch_size=batch_size,
            max_points=max_points,
            tmp_dir=tmp,
        )

        _run_case(
            label="stochastic",
            world_cfg={
                "world_type": "synthetic",
                "samples": int(samples),
                "steps": int(steps),
                "black_scholes": False,
            },
            n_prototypes=n_prototypes,
            epochs=epochs,
            batch_size=batch_size,
            max_points=max_points,
            tmp_dir=tmp,
        )

    print("\nsynthetic prototype extraction validation completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-prototypes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    run_validation(
        samples=args.samples,
        steps=args.steps,
        epochs=args.epochs,
        n_prototypes=args.n_prototypes,
        batch_size=args.batch_size,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()

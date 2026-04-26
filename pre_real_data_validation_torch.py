"""
Pre-real-data smoke checks for the PyTorch ProtoHedge pipeline.

This script is intentionally lightweight. It checks that synthetic BS,
synthetic stochastic-vol, and real-data worlds can run vanilla and prototype
agents with finite outputs before launching expensive real-data experiments.
"""

import argparse
from pathlib import Path
import tempfile

import numpy as np

from deephedging.prototype_extraction_torch import build_prototype_payload, save_prototype_payload
from deephedging.run_train_torch import run_experiment


REQUIRED_RESULT_KEYS = ["utility", "utility0", "pnl", "gains", "actions", "payoff"]


def _assert_finite_result(label, result):
    training_result = result["training_result"]
    for key in REQUIRED_RESULT_KEYS:
        if key not in training_result:
            raise AssertionError(f"{label}: missing training_result['{key}']")
        values = training_result[key].detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise AssertionError(f"{label}: non-finite values in training_result['{key}']")
    print(
        f"{label}: ok | utility={training_result['utility'].mean().item():.6f} "
        f"| pnl={training_result['pnl'].mean().item():.6f}"
    )


def _validate_real_data_file(data_path):
    path = Path(data_path)
    data = np.load(path, mmap_mode="r")
    if data.ndim != 3 or data.shape[-1] < 5:
        raise AssertionError(f"Expected real data shape [N, T, >=5], got {data.shape}")
    if np.isnan(data).any():
        raise AssertionError(f"Real data contains NaNs: {path}")
    print(f"real data: ok | path={path} | shape={data.shape} | dtype={data.dtype}")


def _training_cfg(epochs):
    return {
        "epochs": int(epochs),
        "batch_size": None,
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


def _run_pair(label, world_cfg, n_prototypes, epochs, prototype_path):
    vanilla = run_experiment(
        override_world=world_cfg,
        override_model=_vanilla_model_cfg(),
        override_objective=_objective_cfg(),
        override_training=_training_cfg(epochs),
    )
    _assert_finite_result(f"{label} vanilla", vanilla)

    payload = build_prototype_payload(
        world=vanilla["world"],
        result=vanilla["training_result"],
        n_prototypes=n_prototypes,
        feature_names=["price", "delta", "time_left"],
        random_state=0,
    )
    save_prototype_payload(payload, prototype_path)

    proto = run_experiment(
        override_world=world_cfg,
        override_model=_proto_model_cfg(prototype_path),
        override_objective=_objective_cfg(),
        override_training=_training_cfg(epochs),
    )
    _assert_finite_result(f"{label} proto", proto)
    return vanilla, proto


def run_pre_real_data_smoke(samples=64, epochs=1, n_prototypes=8, data_path="Data/training_paths.npy"):
    _validate_real_data_file(data_path)

    with tempfile.TemporaryDirectory(prefix="protohedge_readiness_") as tmp:
        tmp = Path(tmp)

        _run_pair(
            label="black-scholes",
            world_cfg={
                "world_type": "synthetic",
                "samples": int(samples),
                "steps": 20,
                "black_scholes": True,
            },
            n_prototypes=n_prototypes,
            epochs=epochs,
            prototype_path=tmp / "bs_prototypes.pkl",
        )

        _run_pair(
            label="stochastic",
            world_cfg={
                "world_type": "synthetic",
                "samples": int(samples),
                "steps": 20,
                "black_scholes": False,
            },
            n_prototypes=n_prototypes,
            epochs=epochs,
            prototype_path=tmp / "stoch_prototypes.pkl",
        )

        _run_pair(
            label="real-data",
            world_cfg={
                "world_type": "real",
                "data_path": str(data_path),
                "samples": int(samples),
                "shuffle": True,
                "seed": 1234,
            },
            n_prototypes=n_prototypes,
            epochs=epochs,
            prototype_path=tmp / "real_prototypes.pkl",
        )

    print("pre-real-data smoke checks completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-prototypes", type=int, default=8)
    parser.add_argument("--data-path", default="Data/training_paths.npy")
    args = parser.parse_args()
    run_pre_real_data_smoke(
        samples=args.samples,
        epochs=args.epochs,
        n_prototypes=args.n_prototypes,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()

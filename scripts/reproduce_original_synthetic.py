#!/usr/bin/env python3
"""Reproduce the original ProtoHedge synthetic comparisons from frozen code.

The driver never imports the current checkout's TensorFlow implementation.
Instead, it exports the paper-era source and prototype files from Git, launches
each model in a fresh process, and records paths, weights, metrics, environment,
and checksums. Training caches make an interrupted model resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PAPER_SOURCE_COMMIT = "aedb450"
PAPER_TARGET_GAPS = {
    "black_scholes": -0.00023,
    "stochastic_volatility": -0.00030,
}
PROTOTYPE_PATHS = {
    "black_scholes": "notebooks/prototypes_storage/prototypes_100.pkl",
    "stochastic_volatility": (
        "notebooks/prototypes_storage/prototypes_stochastic_500.pkl"
    ),
}
MODEL_SPECS = (
    "black_scholes:deep_hedging",
    "black_scholes:protohedge",
    "stochastic_volatility:deep_hedging",
    "stochastic_volatility:protohedge",
)
PAPER_PROTOCOL = {
    "metric": "OCE CVaR at lower-tail probability 50% (lambda=1)",
    "training_samples": 10000,
    "validation_samples": 1000,
    "test_samples": 5000,
    "steps": 20,
    "dt": 0.02,
    "epochs": 800,
    "batch_size": 32,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "network_width": 20,
    "network_depth": 3,
    "network_activation": "softplus",
    "black_scholes_prototypes": 100,
    "stochastic_volatility_prototypes": 500,
    "black_scholes_drift": 0.0,
    "stochastic_volatility_drift": 0.1,
    "framework": "TensorFlow 2.13.0",
    "checkpoint_rule": "minimum training OCE loss, including initialization",
    "world_seed": 2312414312,
    "model_seed": 1,
    "test_seed_by_environment": {
        "black_scholes": 1,
        "stochastic_volatility": 42,
    },
}

BLACK_SCHOLES_GENERATED_PROTOTYPES = Path(
    "generated_inputs/black_scholes/prototypes_100_nodrift.pkl"
)


def _json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _git(repo, *args, text=False):
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    return result.stdout


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape or not np.isfinite(values).all():
        raise ValueError("Values and finite weights must have matching shapes")
    total = weights.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return float(np.dot(values, weights) / total)


def weighted_lower_tail_mean(values, weights, probability=0.5):
    """Weighted lower-tail mean with fractional mass at the quantile boundary."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape:
        raise ValueError("Values and weights must have matching shapes")
    if not 0.0 < float(probability) <= 1.0:
        raise ValueError("Tail probability must be in (0, 1]")
    weights = weights / weights.sum()
    order = np.argsort(values, kind="mergesort")
    remaining = float(probability)
    total = 0.0
    for index in order:
        mass = min(remaining, float(weights[index]))
        total += mass * float(values[index])
        remaining -= mass
        if remaining <= 1e-15:
            break
    if remaining > 1e-10:
        raise ValueError("Insufficient probability mass for tail calculation")
    return total / float(probability)


def prepare_source_snapshot(repo, output_dir, commit=PAPER_SOURCE_COMMIT):
    """Export the exact root Python files and paper prototypes from Git."""
    repo = Path(repo).resolve()
    output_dir = Path(output_dir).resolve()
    full_commit = _git(repo, "rev-parse", f"{commit}^{{commit}}", text=True).strip()
    names = _git(repo, "ls-tree", "-r", "--name-only", full_commit, text=True).splitlines()
    source_names = sorted(
        name for name in names if "/" not in name and name.endswith(".py")
    )
    selected = source_names + list(PROTOTYPE_PATHS.values())
    package_root = output_dir / "source_snapshot" / full_commit[:12] / "deephedging"
    records = []
    for source_name in selected:
        content = _git(repo, "show", f"{full_commit}:{source_name}")
        if source_name.endswith(".py") and "/" not in source_name:
            destination = package_root / source_name
        else:
            destination = package_root / source_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite changed snapshot file: {destination}")
        destination.write_bytes(content)
        records.append(
            {
                "git_path": source_name,
                "snapshot_path": str(destination.relative_to(output_dir)),
                "bytes": len(content),
                "sha256": _sha256_bytes(content),
            }
        )
    manifest = {
        "source_commit_requested": commit,
        "source_commit": full_commit,
        "files": records,
    }
    _json_dump(manifest, output_dir / "source_snapshot_manifest.json")
    return package_root.parent, manifest


def _environment_record():
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pip_freeze": freeze.stdout.splitlines(),
        "pip_freeze_stderr": freeze.stderr.strip(),
    }


def _progress_payload(progress):
    return {
        "best_epoch_zero_based": int(progress.best_epoch),
        "best_epoch_displayed_one_based": int(progress.best_epoch) + 1,
        "epochs_completed": int(progress.epoch) + 1,
        "initial_training_loss": float(progress.init_loss),
        "best_training_loss": float(progress.best_loss),
        "training_losses": [float(value) for value in progress.losses.training],
        "validation_losses": [float(value) for value in progress.losses.val],
        "epoch_seconds": [float(value) for value in progress.times],
    }


def _configure_original(config, spec, protocol, output_dir, prototype_path):
    environment, model = spec.split(":", 1)
    config.world.samples = int(protocol["training_samples"])
    config.world.steps = int(protocol["steps"])
    config.world.seed = int(protocol["world_seed"])
    config.world.black_scholes = environment == "black_scholes"
    config.world.drift = float(
        protocol[
            "black_scholes_drift"
            if environment == "black_scholes"
            else "stochastic_volatility_drift"
        ]
    )

    config.gym.objective.utility = "cvar"
    config.gym.objective.lmbda = 1.0
    if model == "deep_hedging":
        config.gym.agent.network.width = int(protocol["network_width"])
        config.gym.agent.network.depth = int(protocol["network_depth"])
        config.gym.agent.network.activation = protocol["network_activation"]
        config.gym.agent.init_delta.active = False
    else:
        config.gym.agent.agent_type = "protopnet"
        config.gym.agent.features = ["price", "delta", "time_left"]
        config.gym.agent.prototype_path = str(prototype_path)

    config.trainer.output_level = "text"
    config.trainer.train.optimizer.name = "adam"
    config.trainer.train.optimizer.learning_rate = float(protocol["learning_rate"])
    if environment == "stochastic_volatility":
        config.trainer.train.optimizer.clipvalue = 1.0
        config.trainer.train.optimizer.global_clipnorm = 1.0
    config.trainer.train.batch_size = None
    config.trainer.train.epochs = int(protocol["epochs"])
    config.trainer.caching.mode = "on"
    config.trainer.caching.directory = str(output_dir / "training_cache")
    config.trainer.caching.epoch_freq = 10
    config.trainer.visual.epoch_refresh = max(1, int(protocol["epochs"]) // 20)
    config.trainer.visual.confidence_pcnt_lo = 0.25
    config.trainer.visual.confidence_pcnt_hi = 0.75


def _generate_black_scholes_prototypes(
    gym,
    world,
    destination,
    n_prototypes,
    source_commit,
    model_seed,
):
    """Regenerate the paper's missing no-drift prototype asset."""
    from sklearn import __version__ as sklearn_version
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.preprocessing import StandardScaler

    training_result = gym(world.tf_data)
    price = np.asarray(world.data.features.per_step["price"], dtype=np.float64)
    time_left = np.asarray(
        world.data.features.per_step["time_left"], dtype=np.float64
    )
    actions = np.asarray(training_result["actions"], dtype=np.float64)[:, :, 0]
    delta = np.cumsum(actions, axis=1) - actions
    features = np.stack([delta, price, time_left], axis=2).reshape(-1, 3)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(
        n_clusters=int(n_prototypes), random_state=0, n_init="auto"
    )
    labels = kmeans.fit_predict(scaled)
    prototypes = []
    for cluster_index in range(int(n_prototypes)):
        cluster_points = scaled[labels == cluster_index]
        if cluster_points.size == 0:
            raise RuntimeError(f"Empty prototype cluster {cluster_index}")
        nearest, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_[cluster_index].reshape(1, -1),
            cluster_points,
        )
        prototypes.append(cluster_points[int(nearest[0])])

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prototypes": np.asarray(prototypes, dtype=np.float64),
        "scaler": scaler,
    }
    with open(destination, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "artifact": str(destination),
        "sha256": _sha256_file(destination),
        "n_prototypes": int(n_prototypes),
        "feature_order": ["delta", "price", "time_left"],
        "extraction_samples": int(world.nSamples),
        "extraction_steps": int(world.nSteps),
        "world_drift": 0.0,
        "source_model": "reproduced deep_hedging checkpoint",
        "source_commit": source_commit,
        "model_seed": int(model_seed),
        "kmeans_random_state": 0,
        "kmeans_n_init": "auto",
        "scikit_learn_version": sklearn_version,
        "provenance_note": (
            "The paper notebook names prototypes_100_nodrift.pkl, but that file "
            "is absent from every Git revision. This deterministic regeneration "
            "uses the notebook's documented scaler, KMeans, and medoid procedure."
        ),
    }
    _json_dump(metadata, destination.with_suffix(".metadata.json"))
    return metadata


def _worker(args):
    output_dir = Path(args.output_dir).resolve()
    protocol = json.loads(Path(args.protocol_json).read_text())
    source_parent = Path(args.source_parent).resolve()
    sys.path.insert(0, str(source_parent))

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import random
    import tensorflow as tf
    from cdxbasics.config import Config
    from deephedging.gym import VanillaDeepHedgingGym
    from deephedging.trainer import Monitor, train
    from deephedging.world import SimpleWorld_Spot_ATM

    spec = args.worker
    environment, model = spec.split(":", 1)
    model_dir = output_dir / "models" / environment / model
    model_dir.mkdir(parents=True, exist_ok=True)
    seed = int(protocol["model_seed"])
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if environment == "black_scholes":
        prototype_path = output_dir / BLACK_SCHOLES_GENERATED_PROTOTYPES
    else:
        prototype_path = source_parent / "deephedging" / PROTOTYPE_PATHS[environment]
    if model == "protohedge" and not prototype_path.is_file():
        raise FileNotFoundError(
            f"Required prototype asset is missing: {prototype_path}. "
            "Run black_scholes:deep_hedging first to regenerate the no-drift asset."
        )
    config = Config()
    _configure_original(config, spec, protocol, model_dir, prototype_path)

    original_finalize = Monitor.finalize

    def capture_finalize(self, status):
        original_finalize(self, status)
        _json_dump(_progress_payload(self.progress_data), model_dir / "training_history.json")

    Monitor.finalize = capture_finalize
    started = time.time()
    world = SimpleWorld_Spot_ATM(config.world)
    val_world = world.clone(samples=int(protocol["validation_samples"]))
    gym = VanillaDeepHedgingGym(config.gym)
    train(gym=gym, world=world, val_world=val_world, config=config.trainer)

    generated_prototype_metadata = None
    if environment == "black_scholes" and model == "deep_hedging":
        generated_prototype_metadata = _generate_black_scholes_prototypes(
            gym=gym,
            world=world,
            destination=prototype_path,
            n_prototypes=int(protocol["black_scholes_prototypes"]),
            source_commit=protocol["source_commit"],
            model_seed=seed,
        )

    test_seed = int(protocol["test_seed_by_environment"][environment])
    test_world = world.clone(samples=int(protocol["test_samples"]), seed=test_seed)
    result = gym(test_world.tf_data)
    weights = np.asarray(test_world.sample_weights, dtype=np.float64)
    gains = np.asarray(result["gains"], dtype=np.float64)
    payoff = np.asarray(result["payoff"], dtype=np.float64)
    utility = np.asarray(result["utility"], dtype=np.float64)
    utility0 = np.asarray(result["utility0"], dtype=np.float64)
    actions = np.asarray(result["actions"], dtype=np.float32)

    metrics = {
        "environment": environment,
        "model": model,
        "source_commit": protocol["source_commit"],
        "epochs": int(protocol["epochs"]),
        "training_samples": int(protocol["training_samples"]),
        "validation_samples": int(protocol["validation_samples"]),
        "test_samples": int(protocol["test_samples"]),
        "world_seed": int(protocol["world_seed"]),
        "model_seed": seed,
        "test_seed": test_seed,
        "expected_utility_frozen_training_y": weighted_mean(utility, weights),
        "expected_utility_empirical_cvar50": weighted_lower_tail_mean(gains, weights),
        "unhedged_utility_frozen_training_y": weighted_mean(utility0, weights),
        "unhedged_utility_empirical_cvar50": weighted_lower_tail_mean(payoff, weights),
        "mean_terminal_gain": weighted_mean(gains, weights),
        "mean_payoff": weighted_mean(payoff, weights),
        "elapsed_seconds": float(time.time() - started),
        "tensorflow_version": tf.__version__,
        "prototype_path": str(prototype_path) if model == "protohedge" else None,
        "prototype_sha256": (
            _sha256_file(prototype_path) if model == "protohedge" else None
        ),
        "generated_prototype_metadata": generated_prototype_metadata,
        "trainable_parameters": int(gym.num_trainable_weights),
    }
    np.savez_compressed(
        model_dir / "test_paths.npz",
        gains=gains.astype(np.float32),
        payoff=payoff.astype(np.float32),
        utility=utility.astype(np.float32),
        utility0=utility0.astype(np.float32),
        actions=actions,
        sample_weights=weights.astype(np.float64),
    )
    np.savez_compressed(
        model_dir / "model_weights.npz",
        **{
            f"weight_{index:03d}": np.asarray(value)
            for index, value in enumerate(gym.get_weights())
        },
    )
    _json_dump(metrics, model_dir / "metrics.json")
    print(json.dumps(metrics, sort_keys=True), flush=True)


def _aggregate(output_dir, selected_specs):
    output_dir = Path(output_dir)
    protocol = json.loads((output_dir / "protocol.json").read_text())
    paper_assessment_applicable = protocol["protocol_mode"] == "paper-reproduction"
    rows = []
    for spec in selected_specs:
        environment, model = spec.split(":", 1)
        path = output_dir / "models" / environment / model / "metrics.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    if rows:
        columns = sorted({key for row in rows for key in row})
        with open(output_dir / "synthetic_summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    by_environment = {
        environment: {row["model"]: row for row in rows if row["environment"] == environment}
        for environment in PAPER_TARGET_GAPS
    }
    assessments = []
    for environment, models in by_environment.items():
        if {"deep_hedging", "protohedge"} - set(models):
            continue
        metric = "expected_utility_frozen_training_y"
        gap = models["protohedge"][metric] - models["deep_hedging"][metric]
        target = PAPER_TARGET_GAPS[environment]
        assessments.append(
            {
                "environment": environment,
                "metric": metric,
                "proto_minus_deep_hedging": gap,
                "paper_reference_gap": target,
                "absolute_gap_error": abs(gap - target),
                "paper_reference_assessment_applicable": paper_assessment_applicable,
                "within_0.0025_of_paper_gap": (
                    abs(gap - target) <= 0.0025
                    if paper_assessment_applicable
                    else None
                ),
                "paper_ordering_reproduced": (
                    gap <= 0.0 if paper_assessment_applicable else None
                ),
            }
        )
    _json_dump(
        {
            "assessment_scope": (
                "The paper reports expected OCE CVaR@50% using each trained "
                "utility threshold y. Smoke tests are execution checks only; full "
                "runs are numerical checks with stochastic-optimization tolerance."
            ),
            "comparisons": assessments,
        },
        output_dir / "reproduction_assessment.json",
    )


def _write_checksums(output_dir):
    output_dir = Path(output_dir)
    ignored_parts = {"training_cache", "source_snapshot"}
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != "artifact_checksums.sha256"
        and not ignored_parts.intersection(path.parts)
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(output_dir)}" for path in files]
    (output_dir / "artifact_checksums.sha256").write_text("\n".join(lines) + "\n")


def _protocol_for_args(args, source_manifest):
    protocol = dict(PAPER_PROTOCOL)
    if args.smoke_test:
        protocol.update(
            {
                "training_samples": 128,
                "validation_samples": 32,
                "test_samples": 256,
                "epochs": 2,
                "protocol_mode": "smoke-test",
            }
        )
    elif args.runtime_probe:
        protocol.update(
            {
                "training_samples": 10000,
                "validation_samples": 1000,
                "test_samples": 256,
                "epochs": 5,
                "protocol_mode": "runtime-probe",
            }
        )
    else:
        protocol["protocol_mode"] = "paper-reproduction"
    protocol["source_commit"] = source_manifest["source_commit"]
    protocol["source_commit_requested"] = args.source_commit
    protocol["selected_models"] = list(args.models)
    protocol["paper_reference_proto_minus_dh"] = PAPER_TARGET_GAPS
    protocol["training_seed_note"] = (
        "The notebooks did not lock TensorFlow initialization before training; "
        "this reproduction explicitly locks model_seed=1."
    )
    protocol["prototype_provenance_note"] = (
        "The committed stochastic-volatility K=500 asset is used directly. "
        "The paper's zero-drift Black-Scholes K=100 asset was not committed, "
        "so it is regenerated from the reproduced DH checkpoint using the "
        "paper notebook's deterministic extraction procedure."
    )
    reference_pdf = Path(args.reference_pdf).expanduser()
    protocol["reference_pdf"] = (
        {
            "path": str(reference_pdf),
            "sha256": _sha256_file(reference_pdf),
            "bytes": reference_pdf.stat().st_size,
        }
        if reference_pdf.is_file()
        else None
    )
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="paper/original_synthetic_reproduction",
        type=Path,
    )
    parser.add_argument("--source-commit", default=PAPER_SOURCE_COMMIT)
    parser.add_argument(
        "--reference-pdf", default="/Users/yaniszaim/Downloads/ProtoHedge.pdf"
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--runtime-probe", action="store_true")
    parser.add_argument("--worker", choices=MODEL_SPECS, help=argparse.SUPPRESS)
    parser.add_argument("--source-parent", help=argparse.SUPPRESS)
    parser.add_argument("--protocol-json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke_test and args.runtime_probe:
        parser.error("--smoke-test and --runtime-probe are mutually exclusive")
    if args.worker:
        _worker(args)
        return 0

    repo = _repo_root()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_parent, source_manifest = prepare_source_snapshot(
        repo, output_dir, commit=args.source_commit
    )
    protocol = _protocol_for_args(args, source_manifest)
    protocol_path = output_dir / "protocol.json"
    _json_dump(protocol, protocol_path)
    _json_dump(_environment_record(), output_dir / "launcher_environment.json")
    requirements = repo / "requirements-original-synthetic.txt"
    if requirements.is_file():
        (output_dir / "requirements-original-synthetic.txt").write_bytes(
            requirements.read_bytes()
        )
    if args.prepare_only:
        _write_checksums(output_dir)
        print(f"Prepared frozen synthetic protocol at {output_dir}")
        return 0

    for spec in args.models:
        environment, model = spec.split(":", 1)
        metrics_path = output_dir / "models" / environment / model / "metrics.json"
        if metrics_path.exists():
            print(f"[resume] {spec} already complete", flush=True)
            continue
        log_path = output_dir / "models" / environment / model / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(output_dir),
            "--worker",
            spec,
            "--source-parent",
            str(source_parent),
            "--protocol-json",
            str(protocol_path),
        ]
        print(f"[start] {spec} | log={log_path}", flush=True)
        with open(log_path, "a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=repo,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{spec} failed; inspect {log_path}")
        print(f"[complete] {spec}", flush=True)

    _aggregate(output_dir, args.models)
    _write_checksums(output_dir)
    print(f"Synthetic reproduction complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

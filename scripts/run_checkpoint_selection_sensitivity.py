#!/usr/bin/env python3
"""Rerun checkpoint-sensitive historical fits under one locked rule.

The completed historical sweep selected 12 Deep Hedging initializations because
position diagnostics were added to validation OCE during checkpoint selection.
This restart-safe runner changes only that selection rule: both Deep Hedging and
the originally validation-selected ProtoHedge configurations use unpenalized
validation OCE. Saturation remains an evaluation diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[1]
    parent = package_root.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

from deephedging.outcome_metrics import (  # noqa: E402
    circular_block_bootstrap_indices,
    empirical_lower_tail_cvar,
)
from deephedging.real_data_analysis_torch import (  # noqa: E402
    save_model_artifact,
    slugify_name,
)
from deephedging.real_data_sweep_torch import evaluate_gym  # noqa: E402
from deephedging.run_train_torch import run_experiment  # noqa: E402


PROTOCOL_VERSION = "checkpoint-selection-sensitivity-v1"
DEFAULT_SOURCE_RUN = Path(
    ".deephedging_real_runs/submission_rerun_scientific_v4"
)
DEFAULT_AUDIT = Path(
    "paper/submission_rerun_scientific_v4/reproducibility/"
    "dh_checkpoint_audit.csv"
)
DEFAULT_OUTPUT = Path(
    ".deephedging_real_runs/checkpoint_selection_sensitivity_v1"
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonify(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_hash(value):
    encoded = json.dumps(_jsonify(value), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_checksums(output_dir):
    output_dir = Path(output_dir)
    destination = output_dir / "artifact_checksums.sha256"
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.resolve() != destination.resolve()
    )
    destination.write_text(
        "\n".join(
            f"{_sha256_file(path)}  {path.relative_to(output_dir)}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )


def _oce_cvar50_values(values, threshold_y):
    values = np.asarray(values, dtype=np.float64)
    return 2.0 * np.minimum(values + float(threshold_y), 0.0) - float(
        threshold_y
    )


def _task_dir(source_run, liability, ticker):
    matches = sorted(
        (Path(source_run) / liability).glob(f"{ticker}_full_grid_*_epochs")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one completed task for {liability}/{ticker}, found {matches}"
        )
    return matches[0]


def _artifact_dir(task_dir, seed, model, risk_measure="cvar"):
    return (
        Path(task_dir)
        / "model_artifacts"
        / f"seed_{int(seed)}"
        / f"risk_{risk_measure}"
        / slugify_name(model)
    )


def _load_metadata(artifact_dir):
    path = Path(artifact_dir) / "artifact_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def _selected_proto_models(task_dir):
    path = Path(task_dir) / "validation_selected_models.csv"
    selected = pd.read_csv(path)
    rows = []
    for row in selected.itertuples(index=False):
        record = {"selection": str(row.selection), "model": str(row.model)}
        if record not in rows:
            rows.append(record)
    return rows


def _affected_fits(audit_path, liabilities=None, max_fits=None):
    audit = pd.read_csv(audit_path)
    affected = audit[audit["selected_epoch"] == -1].copy()
    if liabilities:
        affected = affected[affected["liability"].isin(liabilities)]
    affected = affected.sort_values(["liability", "ticker", "seed"])
    if max_fits is not None:
        affected = affected.head(int(max_fits))
    return affected[["liability", "ticker", "seed"]].to_dict("records")


def _current_data_path(repo_root, metadata):
    path = Path(metadata["data_path"])
    current = repo_root / "Data" / "NEW_PANEL_DECISION_V2" / "episodes" / path.name
    if not current.is_file():
        raise FileNotFoundError(f"Historical episode tensor is missing: {current}")
    return current.resolve()


def _current_prototype_path(task_dir, metadata):
    original = metadata.get("prototype_path")
    if not original:
        return None
    current = Path(task_dir) / "prototypes" / Path(original).name
    if not current.is_file():
        raise FileNotFoundError(f"Historical prototype payload is missing: {current}")
    return current.resolve()


def _locked_overrides(
    repo_root,
    task_dir,
    metadata,
    device,
    epochs=None,
    smoke_test=False,
):
    source = metadata["config"]
    world = dict(source["world"])
    model = dict(source["model"])
    objective = dict(source["objective"])
    training = dict(source["training"])

    data_path = _current_data_path(repo_root, metadata)
    world["data_path"] = str(data_path)
    if smoke_test:
        world["sample_indices"] = list(world["sample_indices"][:128])
        world["val_sample_indices"] = list(world["val_sample_indices"][:32])

    prototype_path = _current_prototype_path(task_dir, metadata)
    if prototype_path is not None:
        model["prototype_path"] = str(prototype_path)

    training.update(
        {
            "device": str(device),
            "epochs": int(2 if smoke_test else (epochs or training["epochs"])),
            "selection_metric": "val_loss",
            "selection_alpha_action_abs": 0.0,
            "selection_alpha_delta_abs": 0.0,
            "selection_alpha_bound_occupancy": 0.0,
            "selection_alpha_path_bound_touch": 0.0,
        }
    )
    training["epoch_refresh"] = max(1, min(25, int(training["epochs"])))
    return world, model, objective, training, data_path


def _utility_threshold(gym):
    state = gym.state_dict()
    return float(state["objective.utility.y"].detach().cpu().item())


def _old_utility_threshold(task_dir, seed, model):
    state_path = _artifact_dir(task_dir, seed, model) / "gym_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    return float(state["objective.utility.y"].detach().cpu().item())


def _stable_seed(label, base=20260828):
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int(base) + int.from_bytes(digest[:4], "little")


def _fit_protocol(
    source_metadata_path,
    metadata,
    world,
    model,
    objective,
    training,
):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_artifact_metadata": str(source_metadata_path),
        "source_artifact_metadata_sha256": _sha256_file(source_metadata_path),
        "source_selected_epoch": metadata.get("history", {}).get("best_epoch"),
        "source_selection_score": metadata.get("history", {}).get("best_score"),
        "locked_config": {
            "world": world,
            "model": model,
            "objective": objective,
            "training": training,
        },
        "only_semantic_change": (
            "Checkpoint score is unpenalized validation OCE. The initialized "
            "model remains eligible, matching the original trainer convention."
        ),
    }


def _run_one(
    repo_root,
    task_dir,
    output_dir,
    liability,
    ticker,
    seed,
    model_name,
    selection_labels,
    device,
    epochs,
    smoke_test,
):
    source_artifact = _artifact_dir(task_dir, seed, model_name)
    metadata, metadata_path = _load_metadata(source_artifact)
    world, model, objective, training, data_path = _locked_overrides(
        repo_root,
        task_dir,
        metadata,
        device=device,
        epochs=epochs,
        smoke_test=smoke_test,
    )
    fit_protocol = _fit_protocol(
        metadata_path, metadata, world, model, objective, training
    )
    protocol_hash = _json_hash(fit_protocol)
    fit_dir = (
        Path(output_dir)
        / liability
        / ticker
        / f"seed_{int(seed)}"
        / slugify_name(model_name)
    )
    completion_path = fit_dir / "complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("fit_protocol_sha256") != protocol_hash:
            raise RuntimeError(
                f"Refusing to mix protocols in existing fit directory: {fit_dir}"
            )
        print(f"[resume] {liability}/{ticker}/seed-{seed}/{model_name}")
        return completion

    fit_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(fit_protocol, fit_dir / "fit_protocol.json")
    started = time.time()
    print(f"[start] {liability}/{ticker}/seed-{seed}/{model_name}", flush=True)
    result = run_experiment(
        override_world=world,
        override_model=model,
        override_objective=objective,
        override_training=training,
    )

    split_indices = {
        key: [int(index) for index in values]
        for key, values in metadata["split_indices"].items()
    }
    if smoke_test:
        split_indices["train"] = list(world["sample_indices"])
        split_indices["val"] = list(world["val_sample_indices"])
        split_indices["test"] = split_indices["test"][:64]
    world_kwargs = dict(metadata["world_kwargs"])
    evaluations = {}
    paths = {}
    threshold_y = _utility_threshold(result["gym"])
    for split in ("val", "test"):
        _, split_result, metrics = evaluate_gym(
            result["gym"],
            data_path=data_path,
            indices=split_indices[split],
            label=f"sensitivity {model_name} {split}",
            world_kwargs=world_kwargs,
        )
        offsets = np.asarray(split_result["liability_offset"], dtype=np.float64)
        utility_values = _oce_cvar50_values(offsets, threshold_y)
        _, empirical_cvar = empirical_lower_tail_cvar(offsets, alpha=0.5)
        evaluations[split] = {
            **metrics,
            "frozen_oce_cvar50_utility": float(utility_values.mean()),
            "empirical_cvar50": float(empirical_cvar),
            "utility_threshold_y": threshold_y,
        }
        paths[f"{split}_liability_offset"] = offsets.astype(np.float32)
        paths[f"{split}_frozen_oce_cvar50"] = utility_values.astype(np.float32)

    artifact_dir = save_model_artifact(
        result=result,
        artifact_root=fit_dir / "artifact",
        model_name=model_name,
        seed=seed,
        risk_measure=metadata["risk_measure"],
        split_indices=split_indices,
        data_path=data_path,
        world_kwargs=world_kwargs,
        model_family=metadata["model_family"],
        prototype_source=metadata.get("prototype_source"),
        n_prototypes=metadata.get("n_prototypes"),
        weighted_similarity=metadata.get("weighted_similarity"),
        learn_distance_feature_weights=metadata.get(
            "learn_distance_feature_weights"
        ),
        split_info=metadata.get("temporal_split"),
    )
    np.savez_compressed(fit_dir / "evaluation_paths.npz", **paths)
    completion = {
        "protocol_version": PROTOCOL_VERSION,
        "fit_protocol_sha256": protocol_hash,
        "liability": liability,
        "ticker": ticker,
        "seed": int(seed),
        "model": model_name,
        "model_family": metadata["model_family"],
        "selection_labels": list(selection_labels),
        "source_selected_epoch": metadata.get("history", {}).get("best_epoch"),
        "new_selected_epoch": result["history"].get("best_epoch"),
        "new_selected_score": result["history"].get("best_score"),
        "new_init_val_loss": result["history"].get("init_val_loss"),
        "new_best_val_loss": result["history"].get("best_val_loss"),
        "artifact_dir": str(artifact_dir),
        "evaluations": evaluations,
        "elapsed_seconds": float(time.time() - started),
    }
    _json_dump(completion, completion_path)
    print(
        f"[complete] {liability}/{ticker}/seed-{seed}/{model_name} "
        f"epoch={completion['new_selected_epoch']} "
        f"hours={completion['elapsed_seconds'] / 3600.0:.2f}",
        flush=True,
    )
    return completion


def _expected_jobs(source_run, fits, matched_proto):
    jobs = []
    for fit in fits:
        task_dir = _task_dir(
            source_run, fit["liability"], fit["ticker"]
        )
        jobs.append(
            {
                **fit,
                "task_dir": task_dir,
                "model": "vanilla",
                "selection_labels": ["benchmark"],
            }
        )
        if matched_proto:
            by_model = {}
            for selected in _selected_proto_models(task_dir):
                by_model.setdefault(selected["model"], []).append(
                    selected["selection"]
                )
            for model, labels in sorted(by_model.items()):
                jobs.append(
                    {
                        **fit,
                        "task_dir": task_dir,
                        "model": model,
                        "selection_labels": labels,
                    }
                )
    return jobs


def _write_summary(output_dir, jobs):
    rows = []
    by_fit = {}
    for job in jobs:
        fit_dir = (
            Path(output_dir)
            / job["liability"]
            / job["ticker"]
            / f"seed_{int(job['seed'])}"
            / slugify_name(job["model"])
        )
        completion_path = fit_dir / "complete.json"
        status = "complete" if completion_path.is_file() else "pending"
        row = {
            "liability": job["liability"],
            "ticker": job["ticker"],
            "seed": int(job["seed"]),
            "model": job["model"],
            "selection_labels": "|".join(job["selection_labels"]),
            "status": status,
        }
        if status == "complete":
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            row.update(
                {
                    "source_selected_epoch": completion["source_selected_epoch"],
                    "new_selected_epoch": completion["new_selected_epoch"],
                    "new_best_val_loss": completion["new_best_val_loss"],
                    "test_frozen_oce_cvar50_utility": completion["evaluations"]["test"]["frozen_oce_cvar50_utility"],
                    "test_liability_offset_mean": completion["evaluations"]["test"]["liability_offset_mean"],
                    "test_liability_offset_cvar50": completion["evaluations"]["test"]["empirical_cvar50"],
                    "test_bound_occupancy": completion["evaluations"]["test"]["pct_at_any_position_bound"],
                    "test_path_bound_touch": completion["evaluations"]["test"]["pct_paths_touch_any_position_bound"],
                    "elapsed_seconds": completion["elapsed_seconds"],
                }
            )
            key = (job["liability"], job["ticker"], int(job["seed"]))
            by_fit.setdefault(key, {})[job["model"]] = row
        rows.append(row)
    status_frame = pd.DataFrame(rows)
    status_frame.to_csv(Path(output_dir) / "run_status.csv", index=False)

    comparisons = []
    for key, models in sorted(by_fit.items()):
        benchmark = models.get("vanilla")
        if benchmark is None:
            continue
        for model, candidate in sorted(models.items()):
            if model == "vanilla":
                continue
            comparisons.append(
                {
                    "liability": key[0],
                    "ticker": key[1],
                    "seed": key[2],
                    "candidate_model": model,
                    "selection_labels": candidate["selection_labels"],
                    "proto_minus_dh_frozen_oce_cvar50": (
                        candidate["test_frozen_oce_cvar50_utility"]
                        - benchmark["test_frozen_oce_cvar50_utility"]
                    ),
                    "proto_minus_dh_liability_offset_mean": (
                        candidate["test_liability_offset_mean"]
                        - benchmark["test_liability_offset_mean"]
                    ),
                    "proto_minus_dh_empirical_cvar50": (
                        candidate["test_liability_offset_cvar50"]
                        - benchmark["test_liability_offset_cvar50"]
                    ),
                }
            )
    pd.DataFrame(comparisons).to_csv(
        Path(output_dir) / "completed_fit_comparisons.csv", index=False
    )
    return status_frame


def _bootstrap_utility_difference(
    candidate,
    benchmark,
    block_length,
    n_bootstrap,
    confidence,
    random_state,
):
    candidate = np.asarray(candidate, dtype=np.float64)
    benchmark = np.asarray(benchmark, dtype=np.float64)
    if candidate.shape != benchmark.shape or candidate.ndim != 2:
        raise ValueError("Utility arrays must share [seed, path] shape")
    samples = circular_block_bootstrap_indices(
        candidate.shape[1],
        int(block_length),
        int(n_bootstrap),
        random_state=int(random_state),
    )
    draws = np.zeros(int(n_bootstrap), dtype=np.float64)
    for seed_index in range(candidate.shape[0]):
        draws += (
            candidate[seed_index][samples].mean(axis=1)
            - benchmark[seed_index][samples].mean(axis=1)
        )
    draws /= candidate.shape[0]
    alpha = 1.0 - float(confidence)
    estimate = float(np.mean(candidate.mean(axis=1) - benchmark.mean(axis=1)))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "bootstrap_probability_gt_zero": float(np.mean(draws > 0.0)),
        "draws": draws,
    }


def _new_utility_paths(output_dir, liability, ticker, seed, model):
    path = (
        Path(output_dir)
        / liability
        / ticker
        / f"seed_{int(seed)}"
        / slugify_name(model)
        / "evaluation_paths.npz"
    )
    with np.load(path) as payload:
        return np.asarray(payload["test_frozen_oce_cvar50"], dtype=np.float64)


def _write_hybrid_panel_bootstrap(
    source_run,
    output_dir,
    fits,
    n_bootstrap=2000,
    block_length=20,
    confidence=0.95,
):
    affected = {}
    for fit in fits:
        affected.setdefault((fit["liability"], fit["ticker"]), set()).add(
            int(fit["seed"])
        )

    rows = []
    panel_draws = {}
    for liability_dir in sorted(
        path for path in Path(source_run).iterdir() if path.is_dir()
    ):
        liability = liability_dir.name
        for task_dir in sorted(liability_dir.glob("*_full_grid_*_epochs")):
            metadata_path = task_dir / "paired_test_liability_offsets_metadata.csv"
            offsets_path = task_dir / "paired_test_liability_offsets.npz"
            config_path = task_dir / "sweep_config.json"
            if not all(
                path.is_file()
                for path in (metadata_path, offsets_path, config_path)
            ):
                continue
            ticker = task_dir.name.split("_full_grid_", 1)[0]
            metadata = pd.read_csv(metadata_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            seeds = [int(seed) for seed in config["seeds"]]
            replacements = affected.get((liability, ticker), set())
            with np.load(offsets_path) as offsets:
                for selected in metadata.itertuples(index=False):
                    selection = str(selected.selection)
                    model = str(selected.candidate_model)
                    candidate_offsets = np.asarray(
                        offsets[selected.candidate_seed_key], dtype=np.float64
                    )
                    benchmark_offsets = np.asarray(
                        offsets[selected.benchmark_seed_key], dtype=np.float64
                    )
                    candidate_utility = np.empty_like(candidate_offsets)
                    benchmark_utility = np.empty_like(benchmark_offsets)
                    for seed_index, seed in enumerate(seeds):
                        candidate_utility[seed_index] = _oce_cvar50_values(
                            candidate_offsets[seed_index],
                            _old_utility_threshold(task_dir, seed, model),
                        )
                        benchmark_utility[seed_index] = _oce_cvar50_values(
                            benchmark_offsets[seed_index],
                            _old_utility_threshold(task_dir, seed, "vanilla"),
                        )
                        if seed in replacements:
                            candidate_utility[seed_index] = _new_utility_paths(
                                output_dir,
                                liability,
                                ticker,
                                seed,
                                model,
                            )
                            benchmark_utility[seed_index] = _new_utility_paths(
                                output_dir,
                                liability,
                                ticker,
                                seed,
                                "vanilla",
                            )
                    result = _bootstrap_utility_difference(
                        candidate_utility,
                        benchmark_utility,
                        block_length=block_length,
                        n_bootstrap=n_bootstrap,
                        confidence=confidence,
                        random_state=_stable_seed(
                            f"{liability}/{ticker}/{selection}"
                        ),
                    )
                    rows.append(
                        {
                            "level": "ticker",
                            "liability": liability,
                            "selection": selection,
                            "ticker": ticker,
                            "candidate_model": model,
                            "benchmark_model": "vanilla",
                            "metric": "frozen_oce_cvar50_utility_difference",
                            "estimate": result["estimate"],
                            "ci_low": result["ci_low"],
                            "ci_high": result["ci_high"],
                            "bootstrap_probability_gt_zero": result[
                                "bootstrap_probability_gt_zero"
                            ],
                            "n_replaced_seed_fits": len(replacements),
                            "n_seeds": len(seeds),
                            "n_observations": candidate_utility.shape[1],
                            "block_length": int(block_length),
                            "n_bootstrap": int(n_bootstrap),
                            "confidence": float(confidence),
                        }
                    )
                    panel_draws.setdefault((liability, selection), []).append(
                        (ticker, result["estimate"], result["draws"])
                    )

    for (liability, selection), ticker_results in sorted(panel_draws.items()):
        estimates = np.asarray(
            [item[1] for item in ticker_results], dtype=np.float64
        )
        draws = np.stack([item[2] for item in ticker_results], axis=0).mean(
            axis=0
        )
        alpha = 1.0 - float(confidence)
        rows.append(
            {
                "level": "panel",
                "liability": liability,
                "selection": selection,
                "ticker": "ALL_EQUAL_WEIGHT",
                "candidate_model": "validation-selected ProtoHedge",
                "benchmark_model": "vanilla",
                "metric": "frozen_oce_cvar50_utility_difference",
                "estimate": float(estimates.mean()),
                "ci_low": float(np.quantile(draws, alpha / 2.0)),
                "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
                "bootstrap_probability_gt_zero": float(np.mean(draws > 0.0)),
                "n_replaced_seed_fits": int(
                    sum(item["n_replaced_seed_fits"] for item in rows if item["level"] == "ticker" and item["liability"] == liability and item["selection"] == selection)
                ),
                "n_seeds": 3,
                "n_observations": None,
                "block_length": int(block_length),
                "n_bootstrap": int(n_bootstrap),
                "confidence": float(confidence),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        Path(output_dir) / "hybrid_checkpoint_sensitivity_bootstrap.csv",
        index=False,
    )
    panel = frame[frame["level"] == "panel"].copy()
    gate = {
        "protocol_version": PROTOCOL_VERSION,
        "decision_rule": (
            "Expand to the complete affected grid if any panel estimate changes "
            "sign or if the checkpoint sensitivity materially changes the paper's "
            "conclusion. Confidence intervals are reported, not used as a hidden "
            "model-selection criterion."
        ),
        "all_panel_estimates_positive": bool((panel["estimate"] > 0.0).all()),
        "all_panel_confidence_intervals_exclude_zero": bool(
            (panel["ci_low"] > 0.0).all()
        ),
        "panel_rows": panel.to_dict("records"),
    }
    _json_dump(gate, Path(output_dir) / "rerun_gate.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--liabilities", nargs="*", default=None)
    parser.add_argument("--max-fits", type=int, default=None)
    parser.add_argument("--dh-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    source_run = (repo_root / args.source_run).resolve()
    audit_path = (repo_root / args.audit).resolve()
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    if args.smoke_test and args.output_dir == DEFAULT_OUTPUT:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    fits = _affected_fits(
        audit_path,
        liabilities=args.liabilities,
        max_fits=args.max_fits,
    )
    jobs = _expected_jobs(source_run, fits, matched_proto=not args.dh_only)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "source_run": str(source_run),
        "source_audit": str(audit_path),
        "source_audit_sha256": _sha256_file(audit_path),
        "selection_metric": "val_loss",
        "selection_penalties": {
            "action_abs": 0.0,
            "delta_abs": 0.0,
            "bound_occupancy": 0.0,
            "path_bound_touch": 0.0,
        },
        "initialization_is_eligible": True,
        "saturation_is_diagnostic_only": True,
        "scope": (
            "Deep Hedging fits whose historical selected_epoch was -1 and the "
            "originally validation-selected matched ProtoHedge configurations."
        ),
        "smoke_test": bool(args.smoke_test),
        "epoch_override": args.epochs,
        "device": args.device,
        "affected_fits": fits,
        "expected_job_count": len(jobs),
        "definitive_scope": bool(
            not args.smoke_test
            and args.epochs is None
            and args.max_fits is None
            and args.liabilities is None
            and not args.dh_only
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }
    protocol_path = output_dir / "protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if _json_hash(previous) != _json_hash(protocol):
            raise RuntimeError(
                f"Existing output uses a different locked protocol: {output_dir}"
            )
    else:
        _json_dump(protocol, protocol_path)
    _write_summary(output_dir, jobs)

    if args.prepare_only or args.summarize_only:
        status = _write_summary(output_dir, jobs)
        complete = int((status["status"] == "complete").sum())
        if complete == len(jobs) and protocol["definitive_scope"]:
            _write_hybrid_panel_bootstrap(source_run, output_dir, fits)
        _write_checksums(output_dir)
        print(
            f"Prepared {len(jobs)} jobs from {len(fits)} affected fits "
            f"({complete} complete) at {output_dir}"
        )
        return 0

    for job in jobs:
        _run_one(
            repo_root=repo_root,
            task_dir=job["task_dir"],
            output_dir=output_dir,
            liability=job["liability"],
            ticker=job["ticker"],
            seed=job["seed"],
            model_name=job["model"],
            selection_labels=job["selection_labels"],
            device=args.device,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
        )
        _write_summary(output_dir, jobs)

    status = _write_summary(output_dir, jobs)
    complete = int((status["status"] == "complete").sum())
    if complete == len(jobs) and protocol["definitive_scope"]:
        _write_hybrid_panel_bootstrap(source_run, output_dir, fits)
    _write_checksums(output_dir)
    print(f"Sensitivity complete: {complete}/{len(jobs)} jobs at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

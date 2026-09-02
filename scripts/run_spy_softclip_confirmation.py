#!/usr/bin/env python3
"""Run the final small real-data confirmation needed for the paper.

This restart-safe study reuses the completed pure-validation Deep Hedging fit
for European SPY and trains only two matched ProtoHedge models.  The models use
the same seed, data, prototypes, objective, and checkpoint rule; only the
SoftClip implementation differs.  It also inventories which historical
headline models belong to the released-paper family versus our expanded grid.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
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


PROTOCOL_VERSION = "spy-european-softclip-confirmation-v1"
LIABILITY = "european_call"
TICKER = "SPY"
DEFAULT_SEED = 2345
DEFAULT_SOURCE_RUN = Path(
    ".deephedging_real_runs/submission_rerun_scientific_v4"
)
DEFAULT_BASELINE_RUN = Path(
    ".deephedging_real_runs/checkpoint_selection_sensitivity_v1"
)
DEFAULT_OUTPUT = Path(
    ".deephedging_real_runs/spy_european_softclip_confirmation_v1"
)
SOFTCLIP_MODES = ("tfp_exact", "legacy_approx")
PRIMARY_SELECTION_METRIC = "liability_offset_cvar05_avg"


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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonify(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_hash(value):
    encoded = json.dumps(_jsonify(value), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_value(repo_root, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _environment_record(repo_root):
    packages = {}
    for package in (
        "cdxbasics",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "seaborn",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "package_versions": packages,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_status_at_launch": _git_value(repo_root, "status", "--short"),
    }


def _task_dir(source_run, liability=LIABILITY, ticker=TICKER):
    matches = sorted(
        (Path(source_run) / liability).glob(f"{ticker}_full_grid_*_epochs")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one completed task for {liability}/{ticker}, found {matches}"
        )
    return matches[0]


def _artifact_dir(task_dir, seed, model_name):
    return (
        Path(task_dir)
        / "model_artifacts"
        / f"seed_{int(seed)}"
        / "risk_cvar"
        / slugify_name(model_name)
    )


def _load_metadata(artifact_dir):
    path = Path(artifact_dir) / "artifact_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source artifact metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _select_paper_faithful_candidate(task_dir):
    """Select K on validation only inside the released-paper-like family."""
    table_path = Path(task_dir) / "validation_model_selection.csv"
    table = pd.read_csv(table_path)
    eligible = table[
        (table["prototype_source"] == "vanilla")
        & (~table["weighted_similarity"].astype(bool))
        & (~table["learn_distance_feature_weights"].astype(bool))
        & (table["has_all_seed_runs"].astype(bool))
        & (table["passes_validation_robust_screen"].astype(bool))
    ].copy()
    if eligible.empty:
        raise RuntimeError("No validation-complete paper-faithful candidate exists")
    eligible = eligible.sort_values(
        [PRIMARY_SELECTION_METRIC, "liability_offset_mean_avg", "n_prototypes"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    selected = eligible.iloc[0].to_dict()
    return {
        "selection_split": "validation",
        "selection_scope": (
            "vanilla-derived prototypes, unweighted similarity, fixed feature weights"
        ),
        "selection_metric": PRIMARY_SELECTION_METRIC,
        "selected_model": str(selected["model"]),
        "selected_n_prototypes": int(selected["n_prototypes"]),
        "selected_validation_record": selected,
        "eligible_candidates": eligible.to_dict("records"),
        "validation_table": str(table_path.resolve()),
        "validation_table_sha256": _sha256_file(table_path),
    }


def _inventory_historical_families(source_run, output_dir):
    headline_rows = []
    faithful_rows = []
    for liability_dir in sorted(
        path for path in Path(source_run).iterdir() if path.is_dir()
    ):
        liability = liability_dir.name
        for task_dir in sorted(liability_dir.glob("*_full_grid_*_epochs")):
            ticker = task_dir.name.split("_full_grid_", 1)[0]
            selected_path = task_dir / "validation_selected_models.csv"
            validation_path = task_dir / "validation_model_selection.csv"
            if not selected_path.is_file() or not validation_path.is_file():
                continue

            selected = pd.read_csv(selected_path)
            for row in selected.to_dict("records"):
                reasons = []
                if str(row.get("prototype_source")) != "vanilla":
                    reasons.append("Spot-Delta-derived prototypes")
                if bool(row.get("weighted_similarity")):
                    reasons.append("hand-weighted similarity")
                if bool(row.get("learn_distance_feature_weights")):
                    reasons.append("learned similarity weights")
                headline_rows.append(
                    {
                        "liability": liability,
                        "ticker": ticker,
                        "selection": row.get("selection"),
                        "model": row.get("model"),
                        "prototype_source": row.get("prototype_source"),
                        "n_prototypes": row.get("n_prototypes"),
                        "weighted_similarity": row.get("weighted_similarity"),
                        "learn_distance_feature_weights": row.get(
                            "learn_distance_feature_weights"
                        ),
                        "paper_faithful_family": not reasons,
                        "extension_reasons": "|".join(reasons),
                    }
                )

            validation = pd.read_csv(validation_path)
            eligible = validation[
                (validation["prototype_source"] == "vanilla")
                & (~validation["weighted_similarity"].astype(bool))
                & (~validation["learn_distance_feature_weights"].astype(bool))
                & (validation["has_all_seed_runs"].astype(bool))
                & (validation["passes_validation_robust_screen"].astype(bool))
            ].copy()
            for label, objective in (
                ("paper_faithful_mean", "liability_offset_mean_avg"),
                ("paper_faithful_cvar05", "liability_offset_cvar05_avg"),
            ):
                if eligible.empty:
                    continue
                chosen = eligible.sort_values(
                    [objective, "n_prototypes"],
                    ascending=[False, True],
                    kind="mergesort",
                ).iloc[0]
                faithful_rows.append(
                    {
                        "liability": liability,
                        "ticker": ticker,
                        "selection": label,
                        "selection_split": "validation",
                        "selection_objective": objective,
                        "model": chosen["model"],
                        "n_prototypes": int(chosen["n_prototypes"]),
                        "validation_liability_offset_mean_avg": chosen[
                            "liability_offset_mean_avg"
                        ],
                        "validation_liability_offset_cvar05_avg": chosen[
                            "liability_offset_cvar05_avg"
                        ],
                    }
                )

    headline = pd.DataFrame(headline_rows)
    faithful = pd.DataFrame(faithful_rows)
    headline.to_csv(output_dir / "historical_headline_family_inventory.csv", index=False)
    faithful.to_csv(output_dir / "paper_faithful_validation_choices.csv", index=False)
    return headline, faithful


def _current_data_path(repo_root, metadata):
    original = Path(metadata["data_path"])
    current = repo_root / "Data" / "NEW_PANEL_DECISION_V2" / "episodes" / original.name
    if not current.is_file():
        raise FileNotFoundError(f"Historical episode tensor is missing: {current}")
    return current.resolve()


def _current_prototype_path(task_dir, metadata):
    original = metadata.get("prototype_path")
    if not original:
        raise FileNotFoundError("ProtoHedge source metadata has no prototype path")
    current = Path(task_dir) / "prototypes" / Path(original).name
    if not current.is_file():
        raise FileNotFoundError(f"Historical prototype payload is missing: {current}")
    return current.resolve()


def _locked_overrides(
    repo_root,
    task_dir,
    metadata,
    device,
    softclip_mode,
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
    model["prototype_path"] = str(prototype_path)
    model["softclip_mode"] = str(softclip_mode)
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
    return world, model, objective, training, data_path, prototype_path


def _utility_threshold(gym):
    state = gym.state_dict()
    return float(state["objective.utility.y"].detach().cpu().item())


def _oce_cvar50_values(values, threshold_y):
    values = np.asarray(values, dtype=np.float64)
    return 2.0 * np.minimum(values + float(threshold_y), 0.0) - float(
        threshold_y
    )


def _fit_dir(output_dir, softclip_mode):
    return Path(output_dir) / "fits" / softclip_mode


def _run_candidate(
    repo_root,
    task_dir,
    source_model,
    output_dir,
    seed,
    softclip_mode,
    device,
    epochs,
    smoke_test,
):
    source_artifact = _artifact_dir(task_dir, seed, source_model)
    metadata, metadata_path = _load_metadata(source_artifact)
    world, model, objective, training, data_path, prototype_path = _locked_overrides(
        repo_root,
        task_dir,
        metadata,
        device=device,
        softclip_mode=softclip_mode,
        epochs=epochs,
        smoke_test=smoke_test,
    )
    fit_protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "source_artifact_metadata": str(metadata_path.resolve()),
        "source_artifact_metadata_sha256": _sha256_file(metadata_path),
        "source_prototype_sha256": _sha256_file(prototype_path),
        "source_selected_epoch": metadata.get("history", {}).get("best_epoch"),
        "locked_config": {
            "world": world,
            "model": model,
            "objective": objective,
            "training": training,
        },
        "semantic_changes_from_historical_fit": [
            "Checkpoint selection uses unpenalized validation OCE.",
            f"ProtoHedge action bounding uses softclip_mode={softclip_mode}.",
        ],
        "all_other_model_and_data_settings_locked_to_source_artifact": True,
    }
    protocol_hash = _json_hash(fit_protocol)
    fit_dir = _fit_dir(output_dir, softclip_mode)
    completion_path = fit_dir / "complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("fit_protocol_sha256") != protocol_hash:
            raise RuntimeError(
                f"Refusing to mix protocols in existing fit directory: {fit_dir}"
            )
        print(f"[resume] {softclip_mode} already complete", flush=True)
        return completion

    fit_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(fit_protocol, fit_dir / "fit_protocol.json")
    started = time.time()
    print(
        f"[start] {LIABILITY}/{TICKER}/seed-{seed}/{softclip_mode}",
        flush=True,
    )
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
            label=f"SPY SoftClip {softclip_mode} {split}",
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

    artifact_model_name = f"{source_model}_softclip_{softclip_mode}"
    artifact_dir = save_model_artifact(
        result=result,
        artifact_root=fit_dir / "artifact",
        model_name=artifact_model_name,
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
        "liability": LIABILITY,
        "ticker": TICKER,
        "seed": int(seed),
        "source_model": source_model,
        "softclip_mode": softclip_mode,
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
        f"[complete] {softclip_mode} "
        f"epoch={completion['new_selected_epoch']} "
        f"minutes={completion['elapsed_seconds'] / 60.0:.1f}",
        flush=True,
    )
    return completion


def _baseline_paths(baseline_run, seed):
    return (
        Path(baseline_run)
        / LIABILITY
        / TICKER
        / f"seed_{int(seed)}"
        / "vanilla"
    )


def _load_baseline(baseline_run, seed):
    baseline_dir = _baseline_paths(baseline_run, seed)
    complete_path = baseline_dir / "complete.json"
    paths_path = baseline_dir / "evaluation_paths.npz"
    if not complete_path.is_file() or not paths_path.is_file():
        raise FileNotFoundError(
            "The completed pure-validation SPY Deep Hedging baseline is missing: "
            f"{baseline_dir}"
        )
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    if completion.get("new_selected_epoch") == -1:
        raise RuntimeError("Expected the audited Deep Hedging fit, not initialization")
    return completion, complete_path, paths_path


def _metric_row(label, completion):
    test = completion["evaluations"]["test"]
    return {
        "model": label,
        "seed": completion["seed"],
        "selected_epoch": completion["new_selected_epoch"],
        "frozen_oce_cvar50_utility": test["frozen_oce_cvar50_utility"],
        "liability_offset_mean": test["liability_offset_mean"],
        "empirical_cvar50": test["empirical_cvar50"],
        "liability_offset_cvar05": test["liability_offset_cvar05"],
        "bound_occupancy": test["pct_at_any_position_bound"],
        "path_bound_touch": test["pct_paths_touch_any_position_bound"],
        "elapsed_seconds": completion.get("elapsed_seconds"),
    }


def _bootstrap_difference(
    candidate,
    benchmark,
    n_bootstrap,
    block_length,
    confidence,
    random_state,
):
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    benchmark = np.asarray(benchmark, dtype=np.float64).reshape(-1)
    n = min(candidate.size, benchmark.size)
    candidate = candidate[:n]
    benchmark = benchmark[:n]
    samples = circular_block_bootstrap_indices(
        n,
        int(block_length),
        int(n_bootstrap),
        random_state=int(random_state),
    )
    draws = (candidate[samples] - benchmark[samples]).mean(axis=1)
    alpha = 1.0 - float(confidence)
    return {
        "estimate": float(np.mean(candidate - benchmark)),
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "bootstrap_probability_gt_zero": float(np.mean(draws > 0.0)),
        "n_observations": int(n),
        "block_length": int(block_length),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
    }


def _load_test_paths(path):
    with np.load(path) as payload:
        return {
            "utility": np.asarray(
                payload["test_frozen_oce_cvar50"], dtype=np.float64
            ),
            "offset": np.asarray(
                payload["test_liability_offset"], dtype=np.float64
            ),
        }


def _write_results(
    output_dir,
    baseline_run,
    seed,
    modes,
    n_bootstrap,
    block_length,
    confidence,
    smoke_test,
):
    baseline, _, baseline_paths_path = _load_baseline(baseline_run, seed)
    completions = {"deep_hedging": baseline}
    paths = {"deep_hedging": _load_test_paths(baseline_paths_path)}
    for mode in modes:
        fit_dir = _fit_dir(output_dir, mode)
        completion_path = fit_dir / "complete.json"
        evaluation_path = fit_dir / "evaluation_paths.npz"
        if completion_path.is_file() and evaluation_path.is_file():
            completions[mode] = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
            paths[mode] = _load_test_paths(evaluation_path)

    pd.DataFrame(
        [_metric_row(label, completion) for label, completion in completions.items()]
    ).to_csv(output_dir / "spy_model_metrics.csv", index=False)

    comparisons = []
    pairs = []
    for mode in modes:
        if mode in paths:
            pairs.append((mode, "deep_hedging"))
    if "tfp_exact" in paths and "legacy_approx" in paths:
        pairs.append(("tfp_exact", "legacy_approx"))
    for index, (candidate, benchmark) in enumerate(pairs):
        result = _bootstrap_difference(
            paths[candidate]["utility"],
            paths[benchmark]["utility"],
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            confidence=confidence,
            random_state=20260901 + index,
        )
        comparisons.append(
            {
                "candidate": candidate,
                "benchmark": benchmark,
                "metric": "frozen_oce_cvar50_utility_difference",
                **result,
                "inference_scope": (
                    "single-seed paired temporal-block sensitivity; descriptive, "
                    "not a replacement for panel inference"
                ),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(
        output_dir / "spy_paired_block_bootstrap.csv", index=False
    )

    exact = next(
        (
            row
            for row in comparisons
            if row["candidate"] == "tfp_exact"
            and row["benchmark"] == "deep_hedging"
        ),
        None,
    )
    softclip = next(
        (
            row
            for row in comparisons
            if row["candidate"] == "tfp_exact"
            and row["benchmark"] == "legacy_approx"
        ),
        None,
    )
    lines = [
        "# SPY European SoftClip Confirmation",
        "",
        "## What this test answers",
        "",
        "This small check asks whether replacing the historical PyTorch SoftClip "
        "approximation with the exact TensorFlow Probability formula changes one "
        "paper-faithful ProtoHedge comparison. It does not replace the completed "
        "10-ticker study and it does not tune on the SPY test set.",
        "",
        "## Locked design",
        "",
        f"- Liability/ticker: `{LIABILITY}/{TICKER}`",
        f"- Seed: `{seed}`",
        "- ProtoHedge family: vanilla-derived prototypes, unweighted similarity, "
        "fixed feature weights",
        "- Checkpoint rule: best unpenalized validation OCE",
        "- Primary comparison metric: frozen-threshold OCE CVaR@50% utility",
        "- Test paths are used once for reporting, never for model selection",
        "",
        "## Result",
        "",
    ]
    if exact is None:
        lines.append(
            "The exact-SoftClip fit is not complete yet. Re-run the same command; "
            "completed fits will be resumed rather than repeated."
        )
    else:
        direction = "higher" if exact["estimate"] > 0 else "lower"
        lines.extend(
            [
                "Exact-SoftClip ProtoHedge utility is "
                f"**{direction}** than Deep Hedging by `{exact['estimate']:+.6f}` "
                f"in this one-seed SPY check (descriptive block interval "
                f"`[{exact['ci_low']:+.6f}, {exact['ci_high']:+.6f}]`).",
                "",
            ]
        )
    if softclip is not None:
        lines.extend(
            [
                "Changing only SoftClip moves ProtoHedge utility by "
                f"`{softclip['estimate']:+.6f}` (exact minus historical "
                "approximation).",
                "",
            ]
        )
    lines.extend(
        [
            "## How to use this in the paper",
            "",
            "- If exact and historical SoftClip are close, report this as a "
            "small implementation-fidelity robustness check.",
            "- If they differ, report the sensitivity and describe the completed "
            "panel as the expanded PyTorch ProtoHedge implementation rather than "
            "an exact numerical port.",
            "- Either outcome is reportable. A one-seed sensitivity result should "
            "not be used to rewrite or discard the full-panel statistical result.",
            "- The confidence interval here describes paired SPY paths for one "
            "training seed; it is not training-seed or cross-asset uncertainty.",
            "",
            f"Smoke-test mode: `{bool(smoke_test)}`",
        ]
    )
    (output_dir / "SPY_CONFIRMATION_INTERPRETATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return len(completions) - 1


def _write_family_report(output_dir, headline):
    if headline.empty:
        counts = {"total": 0, "paper_faithful": 0, "expanded": 0}
    else:
        faithful_count = int(headline["paper_faithful_family"].sum())
        counts = {
            "total": int(len(headline)),
            "paper_faithful": faithful_count,
            "expanded": int(len(headline) - faithful_count),
        }
    lines = [
        "# Historical ProtoHedge Family Scope",
        "",
        "The completed 10-ticker run is preserved exactly as historical evidence. "
        "Its headline model selection searched an expanded ProtoHedge family.",
        "",
        f"- Headline liability/ticker selections: `{counts['total']}`",
        f"- Released-paper-like selections: `{counts['paper_faithful']}`",
        f"- Selections using at least one extension: `{counts['expanded']}`",
        "",
        "A released-paper-like candidate means vanilla-DH-derived prototypes, "
        "unweighted similarity, and fixed feature weights. Spot-Delta-derived "
        "prototypes and weighted similarity are extensions. This classification "
        "does not invalidate the expanded-family result; it determines how the "
        "method must be named in the paper.",
        "",
        "`paper_faithful_validation_choices.csv` records, without test-set tuning, "
        "which released-paper-like K would be selected for each task. The small "
        "SPY run tests one such configuration under exact SoftClip.",
    ]
    (output_dir / "HISTORICAL_FAMILY_SCOPE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _json_dump(counts, output_dir / "historical_family_counts.json")


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


def _estimated_minutes(baseline_run, seed, n_modes):
    parent = _baseline_paths(baseline_run, seed).parent
    proto_minutes = []
    for path in parent.glob("proto*/complete.json"):
        completion = json.loads(path.read_text(encoding="utf-8"))
        proto_minutes.append(float(completion["elapsed_seconds"]) / 60.0)
    per_fit = float(np.median(proto_minutes)) if proto_minutes else 30.0
    return per_fit * int(n_modes), per_fit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--modes", nargs="+", choices=SOFTCLIP_MODES, default=list(SOFTCLIP_MODES)
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    source_run = (
        (repo_root / args.source_run).resolve()
        if not args.source_run.is_absolute()
        else args.source_run.resolve()
    )
    baseline_run = (
        (repo_root / args.baseline_run).resolve()
        if not args.baseline_run.is_absolute()
        else args.baseline_run.resolve()
    )
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    if args.smoke_test and args.output_dir == DEFAULT_OUTPUT:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dir = _task_dir(source_run)
    selection = _select_paper_faithful_candidate(task_dir)
    source_model = selection["selected_model"]
    source_metadata, source_metadata_path = _load_metadata(
        _artifact_dir(task_dir, args.seed, source_model)
    )
    source_data_path = _current_data_path(repo_root, source_metadata)
    source_prototype_path = _current_prototype_path(task_dir, source_metadata)
    _, baseline_complete_path, baseline_paths_path = _load_baseline(
        baseline_run, args.seed
    )
    headline, _ = _inventory_historical_families(source_run, output_dir)
    _write_family_report(output_dir, headline)

    code_paths = [
        Path(__file__).resolve(),
        repo_root / "agents_torch.py",
        repo_root / "softclip_torch.py",
        repo_root / "run_train_torch.py",
    ]
    estimate_minutes, per_fit_minutes = _estimated_minutes(
        baseline_run, args.seed, len(args.modes)
    )
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "one European SPY sensitivity; no full-panel rerun",
        "liability": LIABILITY,
        "ticker": TICKER,
        "seed": int(args.seed),
        "source_run": str(source_run),
        "source_task": str(task_dir),
        "source_task_sweep_config_sha256": _sha256_file(
            task_dir / "sweep_config.json"
        ),
        "reused_deep_hedging_baseline": str(baseline_complete_path),
        "reused_deep_hedging_baseline_sha256": _sha256_file(
            baseline_complete_path
        ),
        "reused_deep_hedging_paths_sha256": _sha256_file(baseline_paths_path),
        "paper_faithful_candidate_selection": selection,
        "selected_source_artifact_metadata": str(source_metadata_path.resolve()),
        "selected_source_artifact_metadata_sha256": _sha256_file(
            source_metadata_path
        ),
        "episode_tensor": str(source_data_path),
        "episode_tensor_sha256": _sha256_file(source_data_path),
        "prototype_payload": str(source_prototype_path),
        "prototype_payload_sha256": _sha256_file(source_prototype_path),
        "softclip_modes": list(args.modes),
        "checkpoint_selection": "unpenalized validation OCE",
        "test_used_for_selection": False,
        "epochs": int(args.epochs) if args.epochs is not None else 800,
        "device": args.device,
        "smoke_test": bool(args.smoke_test),
        "estimated_runtime_minutes": estimate_minutes,
        "estimated_minutes_per_proto_fit_from_prior_spy_runs": per_fit_minutes,
        "code_sha256": {
            str(path.relative_to(repo_root)): _sha256_file(path)
            for path in code_paths
        },
        "interpretation_rule": (
            "Report the result as a one-seed implementation sensitivity regardless "
            "of direction; do not use it to select a new model or erase panel evidence."
        ),
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
    _json_dump(_environment_record(repo_root), output_dir / "environment.json")

    completed = _write_results(
        output_dir=output_dir,
        baseline_run=baseline_run,
        seed=args.seed,
        modes=args.modes,
        n_bootstrap=args.n_bootstrap,
        block_length=args.block_length,
        confidence=args.confidence,
        smoke_test=args.smoke_test,
    )
    if args.prepare_only or args.status_only:
        _write_checksums(output_dir)
        print(
            f"SPY confirmation status: {completed}/{len(args.modes)} ProtoHedge "
            f"fits complete at {output_dir}"
        )
        print(
            f"Estimated fresh runtime: about {estimate_minutes:.0f} minutes "
            f"({per_fit_minutes:.1f} minutes per matched ProtoHedge fit)."
        )
        return 0

    for mode in args.modes:
        _run_candidate(
            repo_root=repo_root,
            task_dir=task_dir,
            source_model=source_model,
            output_dir=output_dir,
            seed=args.seed,
            softclip_mode=mode,
            device=args.device,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
        )
        _write_results(
            output_dir=output_dir,
            baseline_run=baseline_run,
            seed=args.seed,
            modes=args.modes,
            n_bootstrap=args.n_bootstrap,
            block_length=args.block_length,
            confidence=args.confidence,
            smoke_test=args.smoke_test,
        )

    _write_checksums(output_dir)
    print(f"SPY confirmation complete: {output_dir}")
    print(f"Read: {output_dir / 'SPY_CONFIRMATION_INTERPRETATION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

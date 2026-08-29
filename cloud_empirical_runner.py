"""Restart-safe cloud launcher for the definitive empirical paper sweep.

The production defaults reproduce the locked v4 notebook protocol. Work is
partitioned at the ticker/liability level so completed tasks survive process or
VM restarts. Run this module from the directory containing the deephedging
package, for example: ``python -m deephedging.cloud_empirical_runner``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback

import pandas as pd
import torch
from threadpoolctl import threadpool_limits


if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))

from deephedging.base_torch import resolve_torch_device
from deephedging.hedge_accounting import HEDGE_ACCOUNTING_VERSION
from deephedging.outcome_metrics import OUTCOME_DEFINITION_VERSION, PREMIUM_INCLUDED
from deephedging.panel_data_pipeline import (
    EPISODE_TIMING_VERSION,
    OPTION_PATH_VERSION,
    TEMPORAL_SPLIT_VERSION,
)
from deephedging.payoff_state import (
    model_features_for_liability,
    payoff_state_version_for_liability,
)
from deephedging.real_data_analysis_torch import list_saved_artifacts
from deephedging.real_data_sweep_torch import (
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    DEFAULT_SPOT_DELTA_BAND_GRID,
    INFERENCE_VERSION,
    MODEL_SELECTION_VERSION,
    VALIDATION_SELECTIONS,
    run_real_data_sweep,
)


RUN_VERSION = "submission_rerun_scientific_v5_gpu"
TICKERS = ("AAPL", "IWM", "NVDA", "QQQ", "SPY", "TLT", "XLE", "XLF", "XLK", "XLV")
LIABILITIES = ("european_call", "asian_call")
SEEDS = (1234, 2345, 3456)
EPOCHS = 800
LEARNING_RATE = 1e-3
BATCH_SIZE = None
EPOCH_REFRESH = None
PROTOTYPE_COUNTS = (10, 25, 50, 100)
PROTOTYPE_SOURCES = ("spot_delta", "vanilla")
WEIGHTED_SIMILARITY_OPTIONS = (False, True)
LEARN_DISTANCE_FEATURE_WEIGHTS_OPTIONS = (False,)
RISK_MEASURES = ("cvar",)
TRADE_BOUNDS = {"lbnd_as": -1.0, "ubnd_as": 1.0, "lbnd_av": -1.0, "ubnd_av": 1.0}
POSITION_BOUNDS = {
    "lbnd_delta_s": -1.0,
    "ubnd_delta_s": 1.0,
    "lbnd_delta_v": -1.0,
    "ubnd_delta_v": 1.0,
}
TRAIN_SELECTION = {
    "selection_metric": "val_loss",
    "selection_alpha_action_abs": 0.005,
    "selection_alpha_delta_abs": 0.010,
    "selection_alpha_bound_occupancy": 0.100,
    "selection_alpha_path_bound_touch": 0.0,
}
TRAIN_REGULARIZATION = {
    "action_penalty_weight": 0.001,
    "delta_penalty_weight": 0.002,
}
MAX_VALIDATION_BOUND_OCCUPANCY = 0.50
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_CONFIDENCE = DEFAULT_BOOTSTRAP_CONFIDENCE
BOOTSTRAP_BLOCK_LENGTH = 20

COMPLETION_FILES = (
    "fixed_chronological_splits.npz",
    "fixed_episode_metadata.csv",
    "sweep_metrics.csv",
    "sweep_test_summary.csv",
    "paper_model_comparison.csv",
    "paper_best_models.csv",
    "validation_model_selection.csv",
    "validation_selected_models.csv",
    "paired_block_bootstrap.csv",
    "paired_test_liability_offsets.npz",
    "paired_test_liability_offsets_metadata.csv",
    "paper_selected_table.csv",
    "interpretation.md",
    "sweep_test_liability_offset_mean.png",
    "sweep_test_liability_offset_cvar05.png",
    "sweep_test_liability_offset_rmse.png",
    "sweep_test_liability_offset_downside_deviation.png",
    "sweep_test_bound_saturation.png",
    "proto_validation_offset_vs_n_prototypes.png",
    "proto_validation_offset_cvar05_vs_n_prototypes.png",
    "sweep_config.json",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp.replace(path)


def _task_paths(panel_root: Path, ticker: str) -> dict[str, Path]:
    paths = {
        "data_path": panel_root / "episodes" / f"{ticker}_training_paths.npy",
        "split_path": panel_root / "episodes" / f"{ticker}_chronological_splits.npz",
        "episode_metadata_path": panel_root / "episode_metadata" / f"{ticker}_episode_metadata.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing corrected panel files: " + ", ".join(missing))
    return paths


def _task_output(run_root: Path, liability: str, ticker: str, epochs: int) -> Path:
    return run_root / liability / f"{ticker}_full_grid_{epochs}_epochs"


def _protocol_matches(cfg: dict, liability: str, *, smoke: bool) -> bool:
    expected_seeds = (1234,) if smoke else SEEDS
    expected_epochs = int(cfg.get("epochs", -1)) if smoke else EPOCHS
    expected_counts = (10,) if smoke else PROTOTYPE_COUNTS
    expected_sources = ("spot_delta",) if smoke else PROTOTYPE_SOURCES
    expected_weighted = (False,) if smoke else WEIGHTED_SIMILARITY_OPTIONS
    expected_bootstrap = 100 if smoke else BOOTSTRAP_REPETITIONS
    temporal = cfg.get("temporal_split", {})
    return all(
        [
            cfg.get("outcome_definition") == OUTCOME_DEFINITION_VERSION,
            cfg.get("premium_included") is PREMIUM_INCLUDED,
            cfg.get("model_selection_version") == MODEL_SELECTION_VERSION,
            cfg.get("hedge_accounting_version") == HEDGE_ACCOUNTING_VERSION,
            cfg.get("inference_version") == INFERENCE_VERSION,
            cfg.get("episode_timing_version") == EPISODE_TIMING_VERSION,
            cfg.get("liability_type") == liability,
            temporal.get("split_version") == TEMPORAL_SPLIT_VERSION,
            temporal.get("option_path_version") == OPTION_PATH_VERSION,
            temporal.get("episode_timing_version") == EPISODE_TIMING_VERSION,
            int(temporal.get("n_steps", -1)) == 20,
            int(temporal.get("n_observations", -1)) == 21,
            sorted(cfg.get("model_features", []))
            == sorted(model_features_for_liability(liability)),
            tuple(cfg.get("seeds", [])) == expected_seeds,
            int(cfg.get("epochs", -1)) == expected_epochs,
            float(cfg.get("lr", -1.0)) == LEARNING_RATE,
            cfg.get("batch_size") == BATCH_SIZE,
            cfg.get("epoch_refresh") == EPOCH_REFRESH,
            cfg.get("normalize") is True,
            cfg.get("hedge_mode") == "step",
            cfg.get("position_bounds") is True,
            cfg.get("selection_metric") == TRAIN_SELECTION["selection_metric"],
            float(cfg.get("selection_alpha_action_abs", -1.0))
            == TRAIN_SELECTION["selection_alpha_action_abs"],
            float(cfg.get("selection_alpha_delta_abs", -1.0))
            == TRAIN_SELECTION["selection_alpha_delta_abs"],
            float(cfg.get("selection_alpha_bound_occupancy", -1.0))
            == TRAIN_SELECTION["selection_alpha_bound_occupancy"],
            float(cfg.get("selection_alpha_path_bound_touch", -1.0))
            == TRAIN_SELECTION["selection_alpha_path_bound_touch"],
            float(cfg.get("action_penalty_weight", -1.0))
            == TRAIN_REGULARIZATION["action_penalty_weight"],
            float(cfg.get("delta_penalty_weight", -1.0))
            == TRAIN_REGULARIZATION["delta_penalty_weight"],
            float(cfg.get("max_bound_occupancy", -1.0))
            == MAX_VALIDATION_BOUND_OCCUPANCY,
            cfg.get("max_path_touch_rate") is None,
            tuple(cfg.get("spot_delta_band_grid", []))
            == tuple(DEFAULT_SPOT_DELTA_BAND_GRID),
            cfg.get("tuned_baseline_metric") == "liability_offset_mean",
            cfg.get("validation_selection_objectives") == VALIDATION_SELECTIONS,
            cfg.get("payoff_state_version")
            == payoff_state_version_for_liability(liability),
            cfg.get("asian_average_type") == "arithmetic",
            int(cfg.get("asian_start_step", -1)) == 1,
            cfg.get("asian_end_step") is None,
            tuple(cfg.get("prototype_counts", [])) == expected_counts,
            tuple(cfg.get("prototype_sources", [])) == expected_sources,
            tuple(cfg.get("weighted_similarity_options", [])) == expected_weighted,
            tuple(cfg.get("learn_distance_feature_weights_options", []))
            == LEARN_DISTANCE_FEATURE_WEIGHTS_OPTIONS,
            tuple(cfg.get("risk_measures", [])) == RISK_MEASURES,
            int(cfg.get("bootstrap_repetitions", -1)) == expected_bootstrap,
            float(cfg.get("bootstrap_confidence", -1.0)) == BOOTSTRAP_CONFIDENCE,
            int(cfg.get("bootstrap_block_length", -1)) == BOOTSTRAP_BLOCK_LENGTH,
            cfg.get("trade_bounds") == TRADE_BOUNDS,
            cfg.get("cumulative_bounds") == POSITION_BOUNDS,
        ]
    )


def _is_complete(output_dir: Path, liability: str, *, smoke: bool) -> bool:
    required = COMPLETION_FILES
    if not all((output_dir / name).is_file() for name in required):
        return False
    if any((output_dir / name).stat().st_size == 0 for name in required):
        return False
    try:
        cfg = json.loads((output_dir / "sweep_config.json").read_text())
        if cfg.get("checkpoint_stage") != "complete":
            return False
        if not _protocol_matches(cfg, liability, smoke=smoke):
            return False
        artifacts = list_saved_artifacts(output_dir)
        expected_artifacts = 2 if smoke else 51
        if len(artifacts) != expected_artifacts:
            return False
        metrics = pd.read_csv(output_dir / "sweep_metrics.csv")
        if metrics.empty:
            return False
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _assert_partial_compatible(output_dir: Path, liability: str, *, smoke: bool) -> None:
    config_path = output_dir / "sweep_config.json"
    if not config_path.exists():
        return
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unreadable partial sweep configuration in {output_dir}; choose a fresh run root."
        ) from exc
    if not _protocol_matches(cfg, liability, smoke=smoke):
        raise RuntimeError(
            f"Partial output in {output_dir} uses a different scientific protocol. "
            "Do not mix it with this run; choose a fresh --run-root."
        )


def _run_task(
    panel_root: Path,
    run_root: Path,
    ticker: str,
    liability: str,
    device: str,
    *,
    smoke: bool,
    smoke_epochs: int,
) -> float:
    seeds = (1234,) if smoke else SEEDS
    epochs = int(smoke_epochs) if smoke else EPOCHS
    prototype_counts = (10,) if smoke else PROTOTYPE_COUNTS
    prototype_sources = ("spot_delta",) if smoke else PROTOTYPE_SOURCES
    weighted_options = (False,) if smoke else WEIGHTED_SIMILARITY_OPTIONS
    bootstrap_repetitions = 100 if smoke else BOOTSTRAP_REPETITIONS
    output_dir = _task_output(run_root, liability, ticker, epochs)
    output_dir.mkdir(parents=True, exist_ok=True)

    lock_path = output_dir / ".cloud_task.lock"
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker is already running {liability}/{ticker}") from exc

        if _is_complete(output_dir, liability, smoke=smoke):
            print(f"[{liability}] {ticker}: verified complete; skipping", flush=True)
            return 0.0
        _assert_partial_compatible(output_dir, liability, smoke=smoke)

        paths = _task_paths(panel_root, ticker)
        started = time.monotonic()
        print(
            f"[{liability}] {ticker}: starting {epochs}-epoch sweep on {device}",
            flush=True,
        )
        run_real_data_sweep(
            **paths,
            output_dir=output_dir,
            samples=256 if smoke else None,
            seeds=seeds,
            epochs=epochs,
            lr=LEARNING_RATE,
            device=device,
            batch_size=BATCH_SIZE,
            epoch_refresh=EPOCH_REFRESH,
            spot_delta_band_grid=tuple(DEFAULT_SPOT_DELTA_BAND_GRID),
            prototype_counts=prototype_counts,
            prototype_sources=prototype_sources,
            weighted_similarity_options=weighted_options,
            learn_distance_feature_weights_options=LEARN_DISTANCE_FEATURE_WEIGHTS_OPTIONS,
            risk_measures=RISK_MEASURES,
            normalize=True,
            hedge_mode="step",
            position_bounds=True,
            trade_bounds=TRADE_BOUNDS,
            cumulative_bounds=POSITION_BOUNDS,
            liability_type=liability,
            asian_average_type="arithmetic",
            asian_start_step=1,
            asian_end_step=None,
            selection_metric=TRAIN_SELECTION["selection_metric"],
            selection_alpha_action_abs=TRAIN_SELECTION["selection_alpha_action_abs"],
            selection_alpha_delta_abs=TRAIN_SELECTION["selection_alpha_delta_abs"],
            selection_alpha_bound_occupancy=TRAIN_SELECTION[
                "selection_alpha_bound_occupancy"
            ],
            selection_alpha_path_bound_touch=TRAIN_SELECTION[
                "selection_alpha_path_bound_touch"
            ],
            action_penalty_weight=TRAIN_REGULARIZATION["action_penalty_weight"],
            delta_penalty_weight=TRAIN_REGULARIZATION["delta_penalty_weight"],
            max_bound_occupancy=MAX_VALIDATION_BOUND_OCCUPANCY,
            max_path_touch_rate=None,
            tuned_baseline_metric="liability_offset_mean",
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
            bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
        )
        if not _is_complete(output_dir, liability, smoke=smoke):
            raise RuntimeError(f"Completion gate failed after sweep: {output_dir}")
        elapsed = time.monotonic() - started
        print(
            f"[{liability}] {ticker}: complete in {elapsed / 3600.0:.2f} hours",
            flush=True,
        )
        return elapsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    package_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=package_root / "Data" / "NEW_PANEL_DECISION_V2",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=package_root / ".deephedging_real_runs" / RUN_VERSION,
    )
    parser.add_argument("--tickers", nargs="+", choices=TICKERS, default=list(TICKERS))
    parser.add_argument(
        "--liabilities",
        nargs="+",
        choices=LIABILITIES,
        default=list(LIABILITIES),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Production defaults to CUDA and fails rather than silently using CPU.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")

    panel_root = args.panel_root.resolve()
    run_root = args.run_root.resolve()
    if args.smoke_test:
        run_root = run_root.with_name(run_root.name + "_smoke")
        tasks = [("european_call", "AAPL")]
    else:
        tasks = [
            (liability, ticker)
            for liability in args.liabilities
            for ticker in args.tickers
        ]
    tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]

    print(f"ProtoHedge cloud runner {RUN_VERSION}")
    print(f"host={socket.gethostname()} | platform={platform.platform()}")
    print(f"panel_root={panel_root}")
    print(f"run_root={run_root}")
    print(f"worker shard={args.shard_index}/{args.num_shards} | tasks={len(tasks)}")
    print("task order=" + ", ".join(f"{liability}/{ticker}" for liability, ticker in tasks))
    if args.dry_run:
        return 0

    device = resolve_torch_device(args.device)
    if not args.smoke_test and device.type != "cuda":
        raise RuntimeError("The production cloud runner requires CUDA; use --smoke-test for CPU checks.")
    torch.set_num_threads(args.torch_threads)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        print(
            f"CUDA verified | torch={torch.__version__} | "
            f"device={device} | accelerator={torch.cuda.get_device_name(device)}",
            flush=True,
        )

    worker_name = f"shard_{args.shard_index}_of_{args.num_shards}"
    status_path = run_root / "cloud_status" / f"{worker_name}.json"
    status = {
        "run_version": RUN_VERSION,
        "worker": worker_name,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "started_at_unix": time.time(),
        "state": "running",
        "completed": [],
        "failed": [],
        "pending": [f"{liability}/{ticker}" for liability, ticker in tasks],
    }
    _atomic_json(status_path, status)

    with threadpool_limits(limits=args.torch_threads):
        for liability, ticker in tasks:
            task_name = f"{liability}/{ticker}"
            status["current_task"] = task_name
            _atomic_json(status_path, status)
            try:
                elapsed = _run_task(
                    panel_root,
                    run_root,
                    ticker,
                    liability,
                    str(device),
                    smoke=args.smoke_test,
                    smoke_epochs=args.smoke_epochs,
                )
                status["completed"].append(
                    {"task": task_name, "elapsed_seconds": elapsed, "finished_at_unix": time.time()}
                )
                status["pending"].remove(task_name)
            except Exception as exc:
                status["state"] = "failed"
                status["failed"].append(
                    {"task": task_name, "error": repr(exc), "traceback": traceback.format_exc()}
                )
                _atomic_json(status_path, status)
                raise
            finally:
                _atomic_json(status_path, status)

    status["state"] = "complete"
    status["current_task"] = None
    status["finished_at_unix"] = time.time()
    _atomic_json(status_path, status)
    print(f"All assigned tasks passed completion gates | status={status_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

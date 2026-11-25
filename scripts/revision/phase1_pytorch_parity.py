#!/usr/bin/env python3
"""Phase 1 (problem P15): does this repository's PyTorch port behave like the
authors' TensorFlow ProtoHedge code on the paper's simulated worlds?

The historical experiments use the PyTorch port. The authors' code was
reproduced separately with their unmodified TensorFlow implementation
(``deephedging moveable prototypes/protohedge_repro``). This script trains the
PyTorch Deep Hedging and ProtoHedge agents on the same worlds, with the same
test sets, so the two implementations can be compared number for number.

Protocol: ``paper/revision_2026_11/PHASE1_PROTOCOL.md`` (written before any
full run). Commands:

    chain     --env bs|sv --rep paper|r1|r2   DH -> prototypes -> PH -> evaluate
    summarize                                  table against the TensorFlow runs

Each stage writes ``complete.json`` atomically and is skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

RUN_ROOT = REPO / ".deephedging_real_runs" / "revision_v1" / "phase1_pytorch_parity"
TF_ROWS = (
    REPO.parent
    / "deephedging moveable prototypes"
    / ".deephedging_real_runs"
    / "protohedge_repro_tf"
    / "paper800"
    / "summary"
    / "reproduction_rows.csv"
)

FEATURES = ["price", "delta", "time_left"]
PAPER_K = {"bs": 100, "sv": 500}
PAPER_TEST = {"bs": (1, 5000), "sv": (42, 5000)}
LARGE_TEST = {"bs": (20261111, 100_000), "sv": (20261112, 100_000)}
# Same training-data seeds as the TensorFlow replicates. Initial weights cannot
# be matched across frameworks, so each replicate is an independent restart.
REPLICATES = {
    "paper": dict(world_seed=2312414312, torch_seed=3100),
    "r1": dict(world_seed=1101, torch_seed=3101),
    "r2": dict(world_seed=1102, torch_seed=3102),
    "r3": dict(world_seed=1103, torch_seed=3103),
    "r4": dict(world_seed=1104, torch_seed=3104),
}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def cvar_lower(x, alpha: float) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    m = max(1, int(np.floor(alpha * len(x))))
    return float(x[:m].mean())


def world_cfg(env: str, rep: str, samples: int) -> dict:
    seed = REPLICATES[rep]["world_seed"]
    return {
        "world_type": "synthetic",
        "steps": 20,
        "samples": int(samples),
        "seed": int(seed),
        "drift": 0.1,  # default of the authors' world, used by their notebooks
        "dt": 0.02,
        "black_scholes": env == "bs",
        "val_world": {"seed": int(seed % 1_000_000 + 7), "samples": int(samples) // 10},
    }


def training_cfg(env: str, rep: str, epochs: int) -> dict:
    clip = 1.0 if env == "sv" else None  # the SV notebook cells set both clips
    return {
        "epochs": int(epochs),
        "lr": 1e-3,
        "device": "cpu",
        "batch_size": 32,
        "epoch_refresh": 10,
        "seed": int(REPLICATES[rep]["torch_seed"]),
        "clipvalue": clip,
        "global_clipnorm": clip,
        "selection_metric": "train_loss",  # reference trainer: best training loss
    }


def dh_model_cfg() -> dict:
    return {
        "agent_type": "feed_forward",
        "feature_list": list(FEATURES),
        "network_width": 20,
        "network_depth": 3,
        "activation": "softplus",
    }


def ph_model_cfg(prototype_path: Path) -> dict:
    # Same agent settings as the historical v4 runs (unweighted distance,
    # default bounded prototype actions).
    return {
        "agent_type": "protopnet",
        "prototype_path": str(prototype_path),
        "feature_list": list(FEATURES),
        "weighted_similarity": False,
        "learn_distance_feature_weights": False,
    }


def evaluate(gym, world, label: str) -> dict:
    import torch
    from deephedging.base_torch import torchCast

    data = torchCast(world.torch_data, device="cpu")
    gym.eval()
    with torch.no_grad():
        r = gym.forward(data, training=False, return_paths=True)
    out = {k: np.asarray(r[k].detach().cpu().numpy(), dtype=np.float64) for k in
           ("utility", "utility0", "gains", "payoff", "cost")}
    actions = r["actions"].detach().cpu().numpy()
    return {
        "label": label,
        "n": int(world.nSamples),
        "utility": float(out["utility"].mean()),
        "utility0": float(out["utility0"].mean()),
        "cvar50": cvar_lower(out["gains"], 0.5),
        "cvar05": cvar_lower(out["gains"], 0.05),
        "unhedged_cvar50": cvar_lower(out["payoff"], 0.5),
        "payoff_mean": float(out["payoff"].mean()),
        "cost_mean": float(out["cost"].mean()),
        "action_abs_mean": float(np.abs(actions).mean()),
        "_utility_path": out["utility"].astype(np.float32),
        "_gains_path": out["gains"].astype(np.float32),
    }


def run_stage(kind: str, env: str, rep: str, epochs: int, samples: int, root: Path) -> None:
    import torch
    from deephedging.run_train_torch import run_experiment
    from deephedging.prototype_extraction_torch import (
        build_prototype_payload,
        save_prototype_payload,
    )

    d = root / env / rep / kind
    if (d / "complete.json").exists():
        print(f"[skip] {env}/{rep}/{kind} already complete", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    k = PAPER_K[env]
    proto_path = root / env / rep / "prototypes" / f"prototypes_{env}_{k}.pkl"
    if kind == "ph" and not proto_path.exists():
        raise SystemExit(f"missing prototype bank {proto_path}")

    write_json(d / "status.json", {"state": "training", "started": time.strftime("%F %T"),
                                   "pid": os.getpid(), "epochs": epochs})
    t0 = time.time()
    result = run_experiment(
        override_world=world_cfg(env, rep, samples),
        override_model=dh_model_cfg() if kind == "dh" else ph_model_cfg(proto_path),
        override_objective={"risk_measure": "cvar", "risk_aversion": 1.0},
        override_training=training_cfg(env, rep, epochs),
    )
    seconds = time.time() - t0
    gym, world, history = result["gym"], result["world"], result["history"]

    if kind == "dh":
        payload = build_prototype_payload(
            world=world,
            result=result["training_result"],
            n_prototypes=k,
            feature_names=list(FEATURES),
            random_state=0,
        )
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        save_prototype_payload(payload, proto_path)

    torch.save({n: v.detach().cpu() for n, v in gym.state_dict().items()}, d / "gym_state.pt")
    write_json(d / "history.json", {
        "training_loss": history["losses"]["training"],
        "val_loss": history["losses"]["val"],
        "training_util": history["utilities"]["training_util"],
        "val_util": history["utilities"]["val_util"],
        "init_loss": history["init_loss"],
        "init_val_loss": history["init_val_loss"],
        "best_epoch": history["best_epoch"],
        "best_loss": history["best_loss"],
    })

    tests = {}
    for name, (seed, n) in (("paper_test", PAPER_TEST[env]), ("large_test", LARGE_TEST[env])):
        n_eff = n if samples >= 10000 else min(n, 2000)
        test_world = world.clone(seed=int(seed), samples=int(n_eff))
        ev = evaluate(gym, test_world, name)
        np.savez_compressed(d / f"{name}_paths.npz", utility=ev.pop("_utility_path"),
                            gains=ev.pop("_gains_path"))
        tests[name] = ev

    n_params = int(sum(p.numel() for p in gym.parameters() if p.requires_grad))
    write_json(d / "complete.json", {
        "kind": kind, "env": env, "rep": rep, "k": k if kind == "ph" else None,
        "epochs": int(epochs), "samples": int(world.nSamples),
        "train_seconds": seconds, "trainable_parameters": n_params,
        "best_epoch": int(history["best_epoch"]),
        "init_train_loss": float(history["init_loss"]),
        "best_train_loss": float(history["best_loss"]),
        "final_train_loss": float(history["losses"]["training"][-1]),
        "world_seed": REPLICATES[rep]["world_seed"],
        "torch_seed": REPLICATES[rep]["torch_seed"],
        "tests": tests,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "finished": time.strftime("%F %T"),
    })
    write_json(d / "status.json", {"state": "complete", "finished": time.strftime("%F %T")})
    print(f"[done] {env}/{rep}/{kind} in {seconds/60:.1f} min, best epoch "
          f"{history['best_epoch']}, large-test utility {tests['large_test']['utility']:.6f}",
          flush=True)


def cmd_chain(args) -> None:
    import torch
    torch.set_num_threads(int(args.threads))
    root = RUN_ROOT / args.profile
    for kind in ("dh", "ph"):
        run_stage(kind, args.env, args.rep, args.epochs, args.samples, root)


def cmd_summarize(args) -> None:
    import pandas as pd

    root = RUN_ROOT / args.profile
    rows = []
    for env in ("bs", "sv"):
        for rep in REPLICATES:
            recs = {}
            for kind in ("dh", "ph"):
                p = root / env / rep / kind / "complete.json"
                if p.exists():
                    recs[kind] = json.load(open(p))
            if len(recs) < 2:
                continue
            for test in ("paper_test", "large_test"):
                dh, ph = recs["dh"]["tests"][test], recs["ph"]["tests"][test]
                up = np.load(root / env / rep / "ph" / f"{test}_paths.npz")["utility"].astype(np.float64)
                ud = np.load(root / env / rep / "dh" / f"{test}_paths.npz")["utility"].astype(np.float64)
                diff = up - ud
                rows.append({
                    "env": env, "rep": rep, "set": test, "n": dh["n"],
                    "unhedged_cvar50": dh["unhedged_cvar50"],
                    "dh": dh["utility"], "ph": ph["utility"],
                    "dh_vs_unhedged": dh["utility"] - dh["utility0"],
                    "gap": float(diff.mean()),
                    "gap_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                    "rel_gap": float(diff.mean() / abs(dh["utility"])),
                    "cvar05_dh": dh["cvar05"], "cvar05_ph": ph["cvar05"],
                    "dh_best_epoch": recs["dh"]["best_epoch"],
                    "ph_best_epoch": recs["ph"]["best_epoch"],
                    "dh_minutes": recs["dh"]["train_seconds"] / 60,
                    "ph_minutes": recs["ph"]["train_seconds"] / 60,
                })
    if not rows:
        print("no completed chains")
        return
    df = pd.DataFrame(rows)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pytorch_parity_rows.csv", index=False)

    summary = {"rows": len(df), "by_env": {}}
    tf = pd.read_csv(TF_ROWS) if TF_ROWS.exists() else None
    for env in ("bs", "sv"):
        pt = df[(df.env == env) & (df.set == "large_test")]
        if pt.empty:
            continue
        rec = {
            "pytorch_reps": pt.rep.tolist(),
            "pytorch_dh_mean": float(pt.dh.mean()),
            "pytorch_dh_by_rep": dict(zip(pt.rep, pt.dh)),
            "pytorch_rel_gap_by_rep": dict(zip(pt.rep, pt.rel_gap)),
            "pytorch_rel_gap_median": float(pt.rel_gap.median()),
            "pytorch_stalled_reps": pt.rep[pt.rel_gap < -0.03].tolist(),
        }
        paper = df[(df.env == env) & (df.set == "paper_test")]
        rec["pytorch_unhedged_cvar50_paper_test"] = sorted(set(round(v, 6) for v in paper.unhedged_cvar50))
        if tf is not None:
            t = tf[(tf.env == env) & (tf.arm == "ph") & (tf.set == "large_test")]
            rec.update({
                "tf_dh_mean": float(t.dh.mean()),
                "tf_dh_sd": float(t.dh.std(ddof=1)),
                "tf_rel_gap_by_rep": dict(zip(t.rep, t.rel_gap)),
                "tf_rel_gap_median": float(t.rel_gap.median()),
                "tf_unhedged_cvar50_paper_test": sorted(set(
                    round(v, 6) for v in tf[(tf.env == env) & (tf.set == "paper_test")].unhedged_cvar50)),
            })
            rec["G1_data_identity"] = bool(
                rec["pytorch_unhedged_cvar50_paper_test"] == rec["tf_unhedged_cvar50_paper_test"])
            rec["G2_dh_within_1pct_of_tf"] = bool(
                abs(rec["pytorch_dh_mean"] - rec["tf_dh_mean"]) <= 0.01 * abs(rec["tf_dh_mean"]))
        rec["G3_median_rel_gap_no_worse_than_minus_1pct"] = bool(rec["pytorch_rel_gap_median"] >= -0.01)
        summary["by_env"][env] = rec
    write_json(out / "pytorch_parity_summary.json", summary)
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2, default=float))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chain")
    c.add_argument("--env", choices=["bs", "sv"], required=True)
    c.add_argument("--rep", choices=list(REPLICATES), required=True)
    c.add_argument("--epochs", type=int, default=800)
    c.add_argument("--samples", type=int, default=10000)
    c.add_argument("--threads", type=int, default=1)
    c.add_argument("--profile", default="paper800")
    c.set_defaults(func=cmd_chain)
    s = sub.add_parser("summarize")
    s.add_argument("--profile", default="paper800")
    s.set_defaults(func=cmd_summarize)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

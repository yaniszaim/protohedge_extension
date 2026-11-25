#!/usr/bin/env python3
"""Plain-distance ProtoHedge ("phd"): similarity measured in the input space.

The released ProtoHedge layer passes states and prototypes through a learned
encoder before measuring distance, so the heaviest prototype of a decision
need not be a similar market state. The authors' earlier layer, the one their
Black-Scholes notebook run used, had no encoder:

    weights = softmax(-sum_j w_j (x_j - p_j)^2),   output = weights @ bounded prototype outputs

With w fixed at 1 this is their plain layer; with trainable w it is their
feature-weight layer. Here w is trainable and positive, so "similar" means
close in a per-feature rescaling of the standardised state, which can be read
off directly.

    python scripts/revision/rev_plain.py --study S --ticker SPY --liability european_call \
        --mode residual --objective var_he --state s0 --seed 0

Trains the plain layer for an existing cell (reusing that cell's Deep Hedging
agent for the prototype bank, as ProtoHedge does) and writes ``<cell>/phd/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import phase3_fit  # noqa: E402


PlainPHPolicy = rc.PlainPHPolicy   # defined in rev_core


def run(args):
    torch.set_num_threads(1)
    a = {k: getattr(args, k) for k in ("ticker", "liability", "mode", "objective", "state", "seed")}
    cell = phase3_fit.cell_dir(args.study, a)
    out = cell / args.tag
    if (out / "complete.json").exists():
        print("[skip] phd complete", flush=True)
        return
    if not (cell / "dh" / "complete.json").exists():
        raise SystemExit(f"Deep Hedging fit missing for {cell}")
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%T')} {a['liability'][:5]} {a['ticker']} {a['mode']} {a['objective']} "
              f"{a['state']} s{a['seed']} | {msg}", flush=True)

    train_ep = rc.Episodes(a["ticker"], a["liability"], "train")
    val_ep = rc.Episodes(a["ticker"], a["liability"], "val")
    scaler = rc.Scaler(train_ep, a["state"])
    train, val = train_ep.tensors(a["state"]), val_ep.tensors(a["state"])
    seed = rc.stable_seed(a["ticker"], a["liability"], a["mode"], a["objective"], a["state"], a["seed"], args.tag)
    torch.manual_seed(seed)
    dh = rc.DHPolicy(scaler)
    dh.load_state_dict(torch.load(cell / "dh" / "policy.pt", weights_only=True))
    bank = rc.extract_prototypes(dh, train, a["mode"], scaler, k=args.k, seed=seed)
    np.savez_compressed(out / "prototypes.npz", **bank,
                        source_episode_index=train_ep.index[bank["source_path"]],
                        source_start_date=train_ep.start_date[bank["source_path"]].astype(str),
                        state_names=np.array(scaler.names))
    policy = PlainPHPolicy(scaler, bank["prototypes_scaled"], init_weight=args.init_weight)
    objective = rc.Objective(a["objective"])
    fit = rc.train_policy(policy, objective, train, val, a["mode"], seed, max_epochs=args.max_epochs,
                          patience=args.patience, log=lambda m: log(f"phd {m}"))
    torch.save({k_: v.detach().cpu() for k_, v in policy.state_dict().items()}, out / "policy.pt")
    rc.write_json(out / "history.json", fit["history"])
    evals = {}
    for name, ep in (("train", train_ep), ("val", val_ep)):
        ev = rc.evaluate_policy(policy, ep, a["state"], a["mode"])
        evals[name] = {**ev["metrics"], "objective_value": objective.exact_value(ev["offset"], ep.np["premium"])}
        np.savez_compressed(out / f"{name}_paths.npz", offset=ev["offset"].astype(np.float32),
                            positions=ev["positions"].astype(np.float16))
    rc.write_json(out / "complete.json", {
        **a, "family": args.tag, "init_weight": args.init_weight, "k": args.k, "seed_used": seed, "best_epoch": fit["best_epoch"],
        "epochs_run": fit["epochs_run"], "seconds": fit["seconds"], "init_val": fit["history"]["val"][0],
        "init_train": fit["history"]["train"][0], "final_train": fit["history"]["train"][-1],
        "best_val": fit["best_val"], "max_epochs": args.max_epochs, "patience": args.patience,
        "trainable_parameters": int(sum(p.numel() for p in policy.parameters() if p.requires_grad)),
        "feature_weights": dict(zip(scaler.names, policy.feature_weights.detach().numpy().tolist())),
        "evaluations": evals, "finished": time.strftime("%F %T")})
    log(f"[done] phd best epoch {fit['best_epoch']} of {fit['epochs_run']}, val objective {fit['best_val']:+.5f}, "
        f"{fit['seconds'] / 60:.1f} min; feature weights "
        f"{ {n: round(float(w), 2) for n, w in zip(scaler.names, policy.feature_weights.detach())} }")


def main():
    p = argparse.ArgumentParser()
    for name in ("study", "ticker", "liability", "mode", "objective"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--state", default="s0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--max-epochs", dest="max_epochs", type=int, default=800)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--init-weight", dest="init_weight", type=float, default=1.0)
    p.add_argument("--tag", default="phd", help="output folder name of this variant")
    run(p.parse_args())


if __name__ == "__main__":
    main()

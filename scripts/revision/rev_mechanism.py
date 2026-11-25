#!/usr/bin/env python3
"""Mechanism experiments (problems P9, P13): how do the two families respond to
the amount of training data?

  subset : train on the most recent fraction of one ticker's training episodes
           (the validation episodes are unchanged)
  pooled : train one model per liability on the training episodes of all ten
           tickers together, validate on the pooled validation episodes, and
           score it on each ticker separately

Both reuse the main study's policies, objective, optimiser and checkpoint
rule (``rev_core``). ProtoHedge takes its prototypes from the Deep Hedging
agent trained on the same data, as everywhere else. Training and validation
only; the test episodes are opened by ``phase3_final.py``.

    python scripts/revision/rev_mechanism.py subset --ticker SPY --liability european_call --fraction 0.25 --seed 0
    python scripts/revision/rev_mechanism.py pooled --liability european_call --seed 0
    python scripts/revision/rev_mechanism.py queue --spec spec.json --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import phase2_diagnostics as p2  # noqa: E402
import phase3_fit  # noqa: E402

ROOT = phase3_fit.ROOT


class EpisodeSet:
    """Concatenation and/or slice of ``rev_core.Episodes`` with the fields training needs."""

    def __init__(self, parts, tickers, keep=None):
        first = parts[0]
        self.liability, self.split = first.liability, first.split
        self._state_names = first.state_names
        cat = lambda arrays: np.concatenate(arrays, axis=0)
        self.features = {k: cat([p.features[k] for p in parts]) for k in first.features}
        self.np = {k: cat([p.np[k] for p in parts]) for k in first.np}
        self.index = cat([p.index for p in parts])
        self.start_date = cat([p.start_date for p in parts])
        self.end_date = cat([p.end_date for p in parts])
        self.ticker = cat([np.full(p.n, t) for p, t in zip(parts, tickers)])
        if keep is not None:
            self.features = {k: v[keep] for k, v in self.features.items()}
            self.np = {k: v[keep] for k, v in self.np.items()}
            self.index, self.start_date = self.index[keep], self.start_date[keep]
            self.end_date, self.ticker = self.end_date[keep], self.ticker[keep]
        self.n = len(self.index)

    def state_names(self, state):
        return self._state_names(state)

    def exogenous(self, state):
        return np.stack([self.features[k] for k in self.state_names(state)], axis=2)

    def tensors(self, state):
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        return {"x": t(self.exogenous(state)), "hedges": t(self.np["hedges"]), "cost": t(self.np["cost"]),
                "payoff": t(self.np["payoff"]), "premium": t(self.np["premium"]),
                "liab_delta": t(self.np["liab_delta"])}


def recent_fraction(ep: rc.Episodes, fraction: float) -> EpisodeSet:
    """The most recent ``fraction`` of the episodes (they are in chronological order)."""
    n_keep = int(np.ceil(fraction * ep.n))
    return EpisodeSet([ep], [ep.ticker], keep=np.arange(ep.n - n_keep, ep.n))


def pooled(tickers, liability, split) -> tuple:
    parts = [rc.Episodes(t, liability, split) for t in tickers]
    return EpisodeSet(parts, tickers), parts


def fit_families(out: Path, families, train_set, val_set, eval_sets: dict, a: dict, default_k, max_epochs,
                 patience, batch_size, log):
    """Train each family on train_set, checkpoint on val_set, evaluate on eval_sets
    ({label: rev_core.Episodes}). Mirrors phase3_fit.run_family."""
    scaler = rc.Scaler(train_set, a["state"])
    train, val = train_set.tensors(a["state"]), val_set.tensors(a["state"])
    for spec in families:
        fam = rc.parse_family(spec)
        family, kind = fam["name"], fam["kind"]
        d = out / family
        if (d / "complete.json").exists():
            log(f"[skip] {family} complete")
            continue
        d.mkdir(parents=True, exist_ok=True)
        seed = rc.stable_seed(a["tag"], a["liability"], a["mode"], a["objective"], a["state"], a["seed"], family)
        torch.manual_seed(seed)
        k = None
        if kind == "dh":
            policy = rc.DHPolicy(scaler, width=fam["width"])
        else:
            k = fam["k"] or default_k
            dh, _ = rc.load_policy(out, "dh", scaler)
            bank = rc.extract_prototypes(dh, train, a["mode"], scaler, k=k, seed=seed)
            np.savez_compressed(d / "prototypes.npz", **bank,
                                source_episode_index=train_set.index[bank["source_path"]],
                                source_start_date=train_set.start_date[bank["source_path"]].astype(str),
                                source_ticker=train_set.ticker[bank["source_path"]].astype(str),
                                state_names=np.array(scaler.names))
            policy = (rc.PHPolicy if kind == "ph" else rc.PlainPHPolicy)(scaler, bank["prototypes_scaled"])
        objective = rc.Objective(a["objective"])
        fit = rc.train_policy(policy, objective, train, val, a["mode"], seed, max_epochs=max_epochs,
                              patience=patience, batch_size=batch_size, log=lambda m: log(f"{family} {m}"))
        torch.save({k_: v.detach().cpu() for k_, v in policy.state_dict().items()}, d / "policy.pt")
        rc.write_json(d / "history.json", fit["history"])
        evals = {}
        for label, ep in eval_sets.items():
            ev = rc.evaluate_policy(policy, ep, a["state"], a["mode"])
            evals[label] = {**ev["metrics"], "objective_value": objective.exact_value(ev["offset"], ep.np["premium"])}
        rc.write_json(d / "complete.json", {
            **a, "family": family, "kind": kind, "width": fam["width"] if kind == "dh" else None, "k": k,
            "seed_used": seed, "n_train": int(train_set.n), "n_val": int(val_set.n),
            "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"], "seconds": fit["seconds"],
            "best_val": fit["best_val"], "max_epochs": max_epochs, "patience": patience, "batch_size": batch_size,
            "trainable_parameters": int(sum(p.numel() for p in policy.parameters() if p.requires_grad)),
            "evaluations": evals, "finished": time.strftime("%F %T")})
        log(f"[done] {family} best epoch {fit['best_epoch']} of {fit['epochs_run']}, "
            f"val objective {fit['best_val']:+.5f}, {fit['seconds'] / 60:.1f} min")


def make_log(prefix):
    return lambda msg: print(f"{time.strftime('%T')} {prefix} | {msg}", flush=True)


def cmd_subset(args):
    torch.set_num_threads(1)
    pct = int(round(100 * args.fraction))
    a = {"tag": f"{args.ticker}_frac{pct}", "ticker": args.ticker, "liability": args.liability, "mode": args.mode,
         "objective": args.objective, "state": args.state, "seed": args.seed, "fraction": args.fraction}
    out = (ROOT / args.study / args.liability / args.ticker
           / f"{args.mode}__{args.objective}__{args.state}__frac{pct}" / f"seed_{args.seed}")
    full_train = rc.Episodes(args.ticker, args.liability, "train")
    val_ep = rc.Episodes(args.ticker, args.liability, "val")
    train_set = recent_fraction(full_train, args.fraction)
    fit_families(out, args.families.split(","), train_set, val_ep, {"val": val_ep}, a, args.k, args.max_epochs,
                 args.patience, 32, make_log(f"subset {args.liability[:5]} {args.ticker} {pct}% s{args.seed}"))


def cmd_pooled(args):
    torch.set_num_threads(1)
    a = {"tag": "POOLED", "ticker": "POOLED", "liability": args.liability, "mode": args.mode,
         "objective": args.objective, "state": args.state, "seed": args.seed}
    out = (ROOT / args.study / args.liability / "POOLED"
           / f"{args.mode}__{args.objective}__{args.state}" / f"seed_{args.seed}")
    train_set, _ = pooled(p2.TICKERS, args.liability, "train")
    val_set, val_parts = pooled(p2.TICKERS, args.liability, "val")
    eval_sets = {f"val_{t}": ep for t, ep in zip(p2.TICKERS, val_parts)}
    fit_families(out, args.families.split(","), train_set, val_set, eval_sets, a, args.k, args.max_epochs,
                 args.patience, args.batch_size, make_log(f"pooled {args.liability[:5]} s{args.seed}"))


def cmd_queue(args):
    spec = json.load(open(args.spec))
    logs = ROOT / spec["study"] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rc.write_json(ROOT / spec["study"] / f"spec_{Path(args.spec).stem}.json", spec)
    common = ["--study", spec["study"], "--mode", spec["mode"], "--objective", spec["objective"],
              "--state", spec["state"], "--families", spec["families"], "--k", str(spec["k"]),
              "--max-epochs", str(spec["max_epochs"]), "--patience", str(spec["patience"])]
    jobs = []
    for seed in spec["seeds"]:
        for liability in spec["liabilities"]:
            if spec.get("pooled"):
                jobs.append((f"pooled_{liability}_{seed}",
                             ["pooled", "--liability", liability, "--seed", str(seed),
                              "--batch-size", str(spec.get("pooled_batch_size", 32))] + common))
            for ticker in spec.get("subset_tickers", []):
                for fraction in spec.get("fractions", []):
                    jobs.append((f"subset_{ticker}_{liability}_{fraction}_{seed}",
                                 ["subset", "--ticker", ticker, "--liability", liability,
                                  "--fraction", str(fraction), "--seed", str(seed)] + common))
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def run(job):
        name, cmd = job
        with open(logs / f"{name}.log", "a") as f:
            code = subprocess.call([sys.executable, __file__] + cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"{time.strftime('%T')} finished {name} (exit {code})", flush=True)
        return code

    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    print(f"queue done: {sum(c == 0 for c in codes)} ok, {sum(c != 0 for c in codes)} failed", flush=True)


def add_common(p):
    p.add_argument("--study", default="E4_mechanism")
    p.add_argument("--liability", required=True, choices=["european_call", "asian_call"])
    p.add_argument("--mode", default="residual", choices=rc.MODES)
    p.add_argument("--objective", default="var_he", choices=list(rc.OBJECTIVES))
    p.add_argument("--state", default="s1", choices=list(rc.STATES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", default="dh,ph")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--max-epochs", dest="max_epochs", type=int, default=800)
    p.add_argument("--patience", type=int, default=150)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    add_common(s)
    s.add_argument("--ticker", required=True)
    s.add_argument("--fraction", type=float, required=True)
    s.set_defaults(func=cmd_subset)
    q = sub.add_parser("pooled")
    add_common(q)
    q.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    q.set_defaults(func=cmd_pooled)
    r = sub.add_parser("queue")
    r.add_argument("--spec", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.set_defaults(func=cmd_queue)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

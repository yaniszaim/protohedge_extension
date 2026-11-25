#!/usr/bin/env python3
"""Phase 3 runner: one cell = ticker x liability x mode x objective x state x seed.

A cell trains Deep Hedging, extracts prototypes from the states that agent
visits on the training episodes (the original recipe), then trains ProtoHedge.
Both are evaluated on training and validation episodes only. Test episodes are
locked in ``rev_core.Episodes`` and are opened only by the final evaluation.

    cell   --ticker SPY --liability european_call --mode target --objective cvar10_he --state s0 --seed 0
    queue  --spec spec.json --workers 4
    table  --study NAME

Outputs: ``.deephedging_real_runs/revision_v1/phase3/<study>/...``; each stage
writes ``complete.json`` atomically and is skipped when present.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402

ROOT = rc.REPO / ".deephedging_real_runs" / "revision_v1" / "phase3"


def cell_dir(study, a) -> Path:
    return (ROOT / study / a["liability"] / a["ticker"] / f"{a['mode']}__{a['objective']}__{a['state']}"
            / f"seed_{a['seed']}")


def run_family(spec, a, d: Path, train_ep, val_ep, scaler, default_k, max_epochs, patience, log):
    """spec: 'dh', 'dh:w=64', 'ph', 'ph:k=25' or 'phd' (see rev_core.parse_family).
    ProtoHedge variants always take their prototypes from the cell's default Deep
    Hedging agent (width 20), as in the original recipe."""
    import torch
    fam = rc.parse_family(spec)
    family, kind = fam["name"], fam["kind"]
    out = d / family
    if (out / "complete.json").exists():
        log(f"[skip] {family} complete")
        return
    out.mkdir(parents=True, exist_ok=True)
    seed = rc.stable_seed(a["ticker"], a["liability"], a["mode"], a["objective"], a["state"], a["seed"], family)
    train, val = train_ep.tensors(a["state"]), val_ep.tensors(a["state"])
    torch.manual_seed(seed)
    k = None
    extra = {}
    if kind == "dh":
        policy = rc.DHPolicy(scaler, width=fam["width"])
    else:
        if not (d / "dh" / "policy.pt").exists():
            raise SystemExit(f"{family} needs the cell's default Deep Hedging fit first")
        k = fam["k"] or default_k
        dh = rc.DHPolicy(scaler)
        dh.load_state_dict(torch.load(d / "dh" / "policy.pt", weights_only=True))
        bank = rc.extract_prototypes(dh, train, a["mode"], scaler, k=k, seed=seed)
        np.savez_compressed(out / "prototypes.npz", **bank,
                            source_episode_index=train_ep.index[bank["source_path"]],
                            source_start_date=train_ep.start_date[bank["source_path"]].astype(str),
                            state_names=np.array(scaler.names))
        policy = (rc.PHPolicy if kind == "ph" else rc.PlainPHPolicy)(scaler, bank["prototypes_scaled"])
    objective = rc.Objective(a["objective"])
    fit = rc.train_policy(policy, objective, train, val, a["mode"], seed, max_epochs=max_epochs,
                          patience=patience, log=lambda m: log(f"{family} {m}"))
    torch.save({k_: v.detach().cpu() for k_, v in policy.state_dict().items()}, out / "policy.pt")
    rc.write_json(out / "history.json", fit["history"])
    if kind == "phd":
        extra["feature_weights"] = dict(zip(scaler.names, policy.feature_weights.detach().numpy().tolist()))
    evals = {}
    for name, ep in (("train", train_ep), ("val", val_ep)):
        ev = rc.evaluate_policy(policy, ep, a["state"], a["mode"])
        evals[name] = {**ev["metrics"], "objective_value": objective.exact_value(ev["offset"], ep.np["premium"])}
        np.savez_compressed(out / f"{name}_paths.npz", offset=ev["offset"].astype(np.float32),
                            positions=ev["positions"].astype(np.float16))
    rc.write_json(out / "complete.json", {
        **a, "family": family, "kind": kind, "width": fam["width"] if kind == "dh" else None, "k": k,
        "seed_used": seed, "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
        "seconds": fit["seconds"], "init_val": fit["history"]["val"][0], "init_train": fit["history"]["train"][0],
        "final_train": fit["history"]["train"][-1], "best_val": fit["best_val"],
        "max_epochs": max_epochs, "patience": patience,
        "trainable_parameters": int(sum(p.numel() for p in policy.parameters() if p.requires_grad)),
        **extra, "evaluations": evals, "finished": time.strftime("%F %T")})
    log(f"[done] {family} best epoch {fit['best_epoch']} of {fit['epochs_run']}, "
        f"val objective {fit['best_val']:+.5f}, {fit['seconds'] / 60:.1f} min")


def cmd_cell(args):
    import torch
    torch.set_num_threads(1)
    a = {k: getattr(args, k) for k in ("ticker", "liability", "mode", "objective", "state", "seed")}
    d = cell_dir(args.study, a)
    d.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%T')} {a['liability'][:5]} {a['ticker']} {a['mode']} {a['objective']} "
              f"{a['state']} s{a['seed']} | {msg}", flush=True)

    train_ep = rc.Episodes(a["ticker"], a["liability"], "train")
    val_ep = rc.Episodes(a["ticker"], a["liability"], "val")
    scaler = rc.Scaler(train_ep, a["state"])
    if not (d / "rules.json").exists():
        rc.write_json(d / "rules.json", {"train": rc.rule_metrics(train_ep, train_ep),
                                         "val": rc.rule_metrics(val_ep, train_ep)})
    for family in args.families.split(","):
        run_family(family, a, d, train_ep, val_ep, scaler, args.k, args.max_epochs, args.patience, log)


def cmd_queue(args):
    spec = json.load(open(args.spec))
    study = spec["study"]
    keys = ("ticker", "liability", "mode", "objective", "state", "seed")
    if "cells" in spec:                           # explicit list of cells
        cells = [dict(zip(keys, c)) for c in spec["cells"]]
    else:                                         # seeds outermost: every ticker gets seed 0 first
        cells = [dict(zip(keys, (t, l, m, o, st, sd))) for sd, t, l, m, o, st in itertools.product(
            spec["seeds"], spec["tickers"], spec["liabilities"], spec["modes"], spec["objectives"], spec["states"])]
    logs = ROOT / study / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rc.write_json(ROOT / study / f"spec_{Path(args.spec).stem}.json", spec)
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def run(a):
        name = "_".join(str(a[k]) for k in keys)
        cmd = [sys.executable, __file__, "cell", "--study", study,
               "--families", spec.get("families", "dh,ph"), "--k", str(spec.get("k", 100)),
               "--max-epochs", str(spec.get("max_epochs", 800)), "--patience", str(spec.get("patience", 200))]
        for k in keys:
            cmd += [f"--{k}", str(a[k])]
        with open(logs / f"{name}.log", "a") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"{time.strftime('%T')} finished {name} (exit {code})", flush=True)
        return code

    print(f"{len(cells)} cells, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, cells))
    print(f"queue done: {sum(c == 0 for c in codes)} ok, {sum(c != 0 for c in codes)} failed", flush=True)


RULES = ("liability_delta", "delta_vega", "unhedged", "static_oce", "static_minvar", "hold_share")
RULE_COLS = ("offset_cvar50", "he_std", "he_cvar10", "he_cvar05", "he_mean", "cost_mean")
_RULE_CACHE = {}


def references(ticker, liability):
    """Rule-based references on training and validation episodes (computed, not read from disk)."""
    key = (ticker, liability)
    if key not in _RULE_CACHE:
        train_ep, val_ep = rc.Episodes(ticker, liability, "train"), rc.Episodes(ticker, liability, "val")
        _RULE_CACHE[key] = {"train": rc.rule_metrics(train_ep, train_ep), "val": rc.rule_metrics(val_ep, train_ep)}
    return _RULE_CACHE[key]


def load_rows(study):
    import pandas as pd
    rows = []
    for comp in sorted((ROOT / study).glob("*/*/*/seed_*/*/complete.json")):
        c = json.load(open(comp))
        rules = references(c["ticker"], c["liability"])
        base = {k: c[k] for k in ("ticker", "liability", "mode", "objective", "state", "seed", "family",
                                  "best_epoch", "epochs_run", "seconds")}
        for split in ("train", "val"):
            e = c["evaluations"][split]
            row = {**base, "split": split, **{k: v for k, v in e.items() if isinstance(v, (int, float))}}
            for rule in RULES:
                for m in RULE_COLS:
                    row[f"{rule}__{m}"] = rules[split][rule][m]
            rows.append(row)
    return pd.DataFrame(rows)


def cmd_table(args):
    import pandas as pd
    df = load_rows(args.study)
    if df.empty:
        print("no completed fits")
        return
    out = ROOT / args.study / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    ref_rows = []
    for (ticker, liability) in sorted(set(zip(df.ticker, df.liability))):
        for split in ("train", "val"):
            for rule in RULES:
                ref_rows.append({"liability": liability, "ticker": ticker, "split": split, "rule": rule,
                                 **{m: references(ticker, liability)[split][rule][m] for m in RULE_COLS}})
    ref = pd.DataFrame(ref_rows).groupby(["liability", "split", "rule"], sort=False)[list(RULE_COLS)].mean()
    cols = ["offset_cvar50", "he_std", "he_cvar10", "he_cvar05", "he_mean", "cost_mean", "pos_spot_mean",
            "pos_call_mean", "corr_hedge_delta_liability_delta", "best_epoch", "seconds"]
    g = df.groupby(["liability", "split", "objective", "mode", "state", "family"])[cols].mean()
    print(ref.round(4).to_string())
    print(g.round(4).to_string())
    g.to_csv(out / "by_cell.csv")
    ref.to_csv(out / "references.csv")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--study", required=True)
    c.add_argument("--ticker", required=True)
    c.add_argument("--liability", required=True, choices=["european_call", "asian_call"])
    c.add_argument("--mode", required=True, choices=rc.MODES)
    c.add_argument("--objective", required=True, choices=list(rc.OBJECTIVES))
    c.add_argument("--state", default="s0", choices=list(rc.STATES))
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--families", default="dh,ph")
    c.add_argument("--k", type=int, default=100)
    c.add_argument("--max-epochs", dest="max_epochs", type=int, default=800)
    c.add_argument("--patience", type=int, default=200)
    c.set_defaults(func=cmd_cell)
    q = sub.add_parser("queue")
    q.add_argument("--spec", required=True)
    q.add_argument("--workers", type=int, default=4)
    q.set_defaults(func=cmd_queue)
    t = sub.add_parser("table")
    t.add_argument("--study", required=True)
    t.set_defaults(func=cmd_table)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

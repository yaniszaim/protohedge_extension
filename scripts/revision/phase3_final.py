#!/usr/bin/env python3
"""Final evaluation of the main study on the test episodes (2022-2024).

This is the only script that opens the test episodes. It reads a final spec
that names, for every arm, the frozen configuration chosen on training and
validation data; evaluates those fits and the rule-based hedges on the same
test episodes; and reports seed-averaged, equal-ticker results with
calendar-aligned block-bootstrap intervals and Holm-adjusted p-values for the
contrasts declared in the spec.

    python scripts/revision/phase3_final.py evaluate --spec paper/revision_2026_11/phase3/final_spec.json
    python scripts/revision/phase3_final.py report   --spec ...

The first ``evaluate`` writes ``TEST_OPENED.json`` with the spec's hash. A
later run with a different spec stops unless ``--acknowledge-reopen`` is
given, and the reopening is recorded.

``--rehearsal`` runs the same code on the validation episodes instead, writes
to ``final_rehearsal_val`` and never touches the test episodes or the marker.
It exists to check the pipeline end to end before the test episodes are
opened; its numbers are validation numbers (optimistic for the learned
policies, whose checkpoints were chosen on these episodes).

An arm with ``"pooled": true`` is a model trained on all tickers together
(``rev_mechanism.py``); it is standardised with the pooled training statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import phase2_diagnostics as p2  # noqa: E402
import phase3_fit  # noqa: E402
import rev_inference as inf  # noqa: E402

RULES = ["unhedged", "liability_delta", "delta_vega", "static_minvar", "hold_share", "spot_delta_listed"]


def spec_hash(spec) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def out_dir(spec, rehearsal=False) -> Path:
    return phase3_fit.ROOT / spec["study"] / ("final_rehearsal_val" if rehearsal else "final")


_POOLED_SCALERS = {}


def arm_cell_and_scaler(spec, arm, ticker, liability, seed, train_ep):
    """Folder of the fit and the standardisation it was trained with."""
    study = arm.get("study", spec["study"])
    if arm.get("pooled"):
        import rev_mechanism as mech
        key = (liability, arm["state"])
        if key not in _POOLED_SCALERS:
            pooled_train, _ = mech.pooled(p2.TICKERS, liability, "train")
            _POOLED_SCALERS[key] = rc.Scaler(pooled_train, arm["state"])
        cell = (phase3_fit.ROOT / study / liability / "POOLED"
                / f"{arm['mode']}__{arm['objective']}__{arm['state']}" / f"seed_{seed}")
        return cell, _POOLED_SCALERS[key]
    a = {"ticker": ticker, "liability": liability, "mode": arm["mode"], "objective": arm["objective"],
         "state": arm["state"], "seed": seed}
    return phase3_fit.cell_dir(study, a), rc.Scaler(train_ep, arm["state"])


def rule_targets(ep: rc.Episodes, train_ep: rc.Episodes) -> dict:
    n = ep.n
    liab = np.zeros((n, rc.N_STEPS, 2))
    liab[:, :, 0] = ep.np["liab_delta"]
    listed = np.zeros((n, rc.N_STEPS, 2))
    listed[:, :, 0] = ep.np["listed_delta"]
    static = p2.select_static_positions(train_ep.world.data.market)
    return {"unhedged": p2.constant_targets(n, 0, 0), "liability_delta": liab,
            "delta_vega": rc.delta_vega_targets(ep), "static_minvar": p2.constant_targets(n, *static["static_minvar"]),
            "hold_share": p2.constant_targets(n, 1, 0), "spot_delta_listed": listed}


def cmd_evaluate(args):
    spec = json.load(open(args.spec))
    out = out_dir(spec, args.rehearsal)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "TEST_OPENED.json"
    h = spec_hash(spec)
    if args.rehearsal:
        rc.write_json(out / "REHEARSAL.json", {"time": time.strftime("%F %T"), "split": "val", "spec_hash": h})
    elif marker.exists():
        prev = json.load(open(marker))
        if prev["spec_hash"] != h and not args.acknowledge_reopen:
            raise SystemExit("The test episodes were already opened with a different spec "
                             f"({prev['spec_hash']}). Re-running changes what the test set was used for; "
                             "pass --acknowledge-reopen to proceed and record it.")
        if prev["spec_hash"] != h:
            prev.setdefault("reopened", []).append({"time": time.strftime("%F %T"), "spec_hash": h})
            prev["spec_hash"] = h
            rc.write_json(marker, prev)
    else:
        rc.write_json(marker, {"first_opened": time.strftime("%F %T"), "spec_hash": h, "spec_path": str(args.spec)})

    rows = []
    for liability in spec["liabilities"]:
        for ticker in spec["tickers"]:
            train_ep = rc.Episodes(ticker, liability, "train")
            test_ep = (rc.Episodes(ticker, liability, "val") if args.rehearsal
                       else rc.Episodes(ticker, liability, "test", allow_test=True))
            arrays = {"premium": test_ep.np["premium"].astype(np.float32),
                      "start_date": test_ep.start_date.astype(str),
                      "liab_delta": test_ep.np["liab_delta"].astype(np.float32)}
            base = {"liability": liability, "ticker": ticker, "n_test": test_ep.n}
            for name, target in rule_targets(test_ep, train_ep).items():
                sim = p2.simulate_targets(test_ep.world.data.market, target)
                arrays[f"offset__{name}"] = sim["offset"].astype(np.float32)[None, :]
                rows.append({**base, "arm": name, "seed": -1, "kind": "rule",
                             **p2.outcome_metrics(sim["offset"], test_ep.np["premium"]),
                             "he_cvar10": float(p2.lower_cvar(sim["offset"] + test_ep.np["premium"], 0.10)),
                             "cost_mean": float(sim["cost"].mean()),
                             **p2.position_metrics(sim["positions"], test_ep.np["listed_delta"], test_ep.np["liab_delta"])})
            unhedged_std = float((arrays["offset__unhedged"][0] + test_ep.np["premium"]).std())
            for arm in spec["arms"]:
                offsets = []
                for seed in arm["seeds"]:
                    cell, scaler = arm_cell_and_scaler(spec, arm, ticker, liability, seed, train_ep)
                    policy, meta = rc.load_policy(cell, arm["family"], scaler)
                    ev = rc.evaluate_policy(policy, test_ep, arm["state"], arm["mode"])
                    he_std = ev["metrics"]["he_std"]
                    rows.append({**base, "arm": arm["name"], "seed": seed, "kind": meta.get("kind", arm["family"]),
                                 **ev["metrics"], "best_epoch": meta["best_epoch"],
                                 "val_objective": meta["best_val"],
                                 "failed": bool(he_std > unhedged_std)})
                    offsets.append(ev["offset"].astype(np.float32))
                    if arm.get("save_positions"):
                        arrays[f"positions__{arm['name']}__{seed}"] = ev["positions"].astype(np.float16)
                arrays[f"offset__{arm['name']}"] = np.stack(offsets)
            (out / "paths" / liability).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "paths" / liability / f"{ticker}.npz", **arrays)
            print(f"{time.strftime('%T')} evaluated {liability} {ticker}", flush=True)
    pd.DataFrame(rows).to_csv(out / "rows_by_ticker_seed.csv", index=False)
    print("wrote", out)


def cmd_report(args):
    spec = json.load(open(args.spec))
    out = out_dir(spec, args.rehearsal)
    rows = pd.read_csv(out / "rows_by_ticker_seed.csv")
    num = [c for c in rows.columns if rows[c].dtype.kind in "fib" and c not in ("seed", "n_test")]
    by_ticker = rows.groupby(["liability", "ticker", "arm"], sort=False)[num].mean().reset_index()
    by_ticker.to_csv(out / "scoreboard_by_ticker.csv", index=False)
    panel = by_ticker.groupby(["liability", "arm"], sort=False)[num].mean().reset_index()
    panel.to_csv(out / "scoreboard_panel.csv", index=False)
    seed_sd = (rows[rows.seed >= 0].groupby(["liability", "arm", "seed"])[["he_std", "he_cvar05"]].mean()
               .groupby(level=[0, 1]).std().add_suffix("_sd_across_seeds").reset_index())
    seed_sd.to_csv(out / "seed_dispersion.csv", index=False)

    arms = [a["name"] for a in spec["arms"]] + RULES
    records = []
    for liability in spec["liabilities"]:
        outcomes, premiums, dates = {m: [] for m in arms}, [], []
        for ticker in spec["tickers"]:
            z = np.load(out / "paths" / liability / f"{ticker}.npz")
            for m in arms:
                outcomes[m].append(z[f"offset__{m}"].astype(np.float64))
            premiums.append(z["premium"].astype(np.float64))
            dates.append(z["start_date"])
        for block in spec.get("blocks", [20, 10, 40]):
            res = inf.panel_bootstrap(outcomes, premiums, dates, block=block, draws=spec.get("draws", 2000),
                                      seed=20261114 + block, calendar=True)
            for c in spec["contrasts"]:
                for metric in c["metrics"]:
                    records.append({"liability": liability, "block": block, "primary": bool(c.get("primary", False)),
                                    **inf.contrast(res, c["candidate"], c["benchmark"], metric)})
            print(f"{time.strftime('%T')} bootstrap {liability} block {block}", flush=True)
    contrasts = pd.DataFrame(records)
    main = contrasts[(contrasts.block == spec.get("blocks", [20])[0]) & contrasts.primary].copy()
    main["p_holm"] = inf.holm(main.p_boot.to_numpy())
    contrasts = contrasts.merge(main[["liability", "candidate", "benchmark", "metric", "block", "p_holm"]],
                                how="left", on=["liability", "candidate", "benchmark", "metric", "block"])
    contrasts.to_csv(out / "contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["liability", "arm", "he_std", "he_cvar05", "he_cvar10", "offset_cvar50", "he_mean", "cost_mean"]
    print(panel[cols].round(4).to_string(index=False))
    print(main.round(5).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--spec", required=True)
    e.add_argument("--acknowledge-reopen", dest="acknowledge_reopen", action="store_true")
    e.add_argument("--rehearsal", action="store_true", help="validation episodes; test stays closed")
    e.set_defaults(func=cmd_evaluate)
    r = sub.add_parser("report")
    r.add_argument("--spec", required=True)
    r.add_argument("--rehearsal", action="store_true")
    r.set_defaults(func=cmd_report)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

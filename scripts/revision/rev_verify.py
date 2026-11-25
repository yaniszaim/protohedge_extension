#!/usr/bin/env python3
"""Integrity checks on a finished Phase 3 study (training and validation data only).

For every fit a queue spec promises, this checks that

  1. the fit exists with its policy, history, prototype bank and saved paths;
  2. the recorded budget (epoch cap, patience) and size (width, K) match the spec;
  3. every stored number is finite;
  4. the stored policy is the checkpoint the rule names: the best validation
     value in the history, equal to the stored validation objective;
  5. reloading the policy and re-running it reproduces the saved validation
     outcomes;
  6. the outcomes agree with an independent simulator (the Phase 2 accounting
     that reproduced the submitted paper's numbers) fed the same positions;
  7. positions and per-step trades respect the +/-1 limits;
  8. every prototype is a state the cell's width-20 Deep Hedging agent visited
     on a training episode, with the stored source date;
  9. no two fits share a seed;
 10. training episodes end before validation episodes start, and the test
     episodes are still locked and unopened.

It also counts fits that reached the epoch cap and fits that failed (more
dispersion than the unhedged position on validation).

    python scripts/revision/rev_verify.py --specs paper/revision_2026_11/phase3/M1_A_spec.json ...
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import phase2_diagnostics as p2  # noqa: E402
import phase3_fit  # noqa: E402

KEYS = ("ticker", "liability", "mode", "objective", "state", "seed")


def expected(spec):
    cells = [dict(zip(KEYS, (t, l, m, o, st, sd))) for sd, t, l, m, o, st in itertools.product(
        spec["seeds"], spec["tickers"], spec["liabilities"], spec["modes"], spec["objectives"], spec["states"])]
    fams = [rc.parse_family(f) for f in spec["families"].split(",")]
    return cells, fams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.set_num_threads(1)

    specs = [json.load(open(s)) for s in args.specs]
    study = specs[0]["study"]
    assert all(s["study"] == study for s in specs)
    todo = {}                                           # (cell tuple, family name) -> (cell, fam, spec)
    for spec in specs:
        cells, fams = expected(spec)
        for a in cells:
            for fam in fams:
                todo.setdefault((tuple(a[k] for k in KEYS), fam["name"]), (a, fam, spec))

    problems = defaultdict(list)                        # check name -> list of messages
    counts = Counter()
    seeds = defaultdict(list)
    worst = {"reload": 0.0, "simulator": 0.0, "checkpoint": 0.0, "prototype": 0.0}
    by_data = defaultdict(list)
    for key, item in todo.items():
        by_data[(key[0][0], key[0][1])].append(item)

    def bad(check, a, family, msg):
        problems[check].append(f"{a['liability']} {a['ticker']} {a['mode']} {a['objective']} s{a['seed']} {family}: {msg}")

    for (ticker, liability), items in sorted(by_data.items()):
        train_ep = rc.Episodes(ticker, liability, "train")
        val_ep = rc.Episodes(ticker, liability, "val")
        if not np.max(train_ep.end_date) < np.min(val_ep.start_date):
            problems["split_order"].append(f"{ticker} {liability}: training ends {np.max(train_ep.end_date)}, "
                                           f"validation starts {np.min(val_ep.start_date)}")
        counts["split_order_checked"] += 1
        unhedged_std = float((p2.simulate_targets(val_ep.world.data.market, p2.constant_targets(val_ep.n, 0, 0))["offset"]
                              + val_ep.np["premium"]).std())
        scalers, dh_states = {}, {}
        for a, fam, spec in items:
            family = fam["name"]
            counts["expected"] += 1
            d = phase3_fit.cell_dir(study, a) / family
            needed = ["complete.json", "policy.pt", "history.json", "val_paths.npz", "train_paths.npz"]
            if fam["kind"] != "dh":
                needed.append("prototypes.npz")
            missing = [f for f in needed if not (d / f).exists()]
            if missing:
                bad("missing", a, family, ", ".join(missing))
                continue
            counts["present"] += 1
            c = json.load(open(d / "complete.json"))
            hist = json.load(open(d / "history.json"))

            # 2. recorded budget and size
            if c["max_epochs"] != spec["max_epochs"] or c["patience"] != spec["patience"]:
                bad("budget", a, family, f"cap {c['max_epochs']}, patience {c['patience']}")
            if fam["kind"] == "dh" and int(c["width"]) != int(fam["width"] or 20):
                bad("size", a, family, f"width {c['width']}")
            if fam["kind"] != "dh" and int(c["k"]) != int(fam["k"] or spec["k"]):
                bad("size", a, family, f"k {c['k']}")
            if c["seed_used"] != rc.stable_seed(*[a[k] for k in KEYS], family):
                bad("seed", a, family, "seed differs from the hashed seed")
            seeds[c["seed_used"]].append((key_str := f"{a} {family}"))

            # 3. finite numbers
            numbers = [v for e in c["evaluations"].values() for v in e.values() if isinstance(v, (int, float))]
            if not (np.all(np.isfinite(numbers)) and np.all(np.isfinite(hist["val"])) and np.all(np.isfinite(hist["train"]))):
                bad("finite", a, family, "non-finite metric or history value")

            # 4. checkpoint rule (history index 0 is the untrained policy, epoch -1)
            val_hist = np.asarray(hist["val"])
            if int(np.argmax(val_hist)) - 1 != int(c["best_epoch"]):
                bad("checkpoint", a, family, f"best epoch {c['best_epoch']}, history argmax {int(np.argmax(val_hist)) - 1}")
            gap = abs(float(val_hist.max()) - c["evaluations"]["val"]["objective_value"])
            worst["checkpoint"] = max(worst["checkpoint"], gap)
            if gap > 2e-6 or abs(c["best_val"] - float(val_hist.max())) > 1e-9:
                bad("checkpoint", a, family, f"stored policy scores {c['evaluations']['val']['objective_value']:.7f}, "
                                             f"best in history {val_hist.max():.7f}")
            if len(val_hist) - 1 != c["epochs_run"]:
                bad("checkpoint", a, family, "history length differs from epochs run")
            stopped_by_cap = c["epochs_run"] >= c["max_epochs"]
            counts[f"cap_reached__{family}__{a['mode']}__{a['objective']}"] += int(stopped_by_cap)
            if not stopped_by_cap and c["epochs_run"] - c["best_epoch"] - 1 < c["patience"]:
                bad("checkpoint", a, family, "stopped before the patience ran out")

            # 5. reload and re-run on validation
            if a["state"] not in scalers:
                scalers[a["state"]] = rc.Scaler(train_ep, a["state"])
            scaler = scalers[a["state"]]
            policy, _ = rc.load_policy(d.parent, family, scaler)
            ev = rc.evaluate_policy(policy, val_ep, a["state"], a["mode"])
            saved = np.load(d / "val_paths.npz")
            diff = float(np.max(np.abs(ev["offset"] - saved["offset"])))
            worst["reload"] = max(worst["reload"], diff)
            if diff > 2e-6:
                bad("reload", a, family, f"max outcome difference {diff:.2e}")
            if abs(ev["metrics"]["he_std"] - c["evaluations"]["val"]["he_std"]) > 1e-7:
                bad("reload", a, family, "validation std differs from the stored value")

            # 6. independent simulator on the same positions
            sim = p2.simulate_targets(val_ep.world.data.market, ev["positions"])
            diff = float(np.max(np.abs(sim["offset"] - ev["offset"])))
            worst["simulator"] = max(worst["simulator"], diff)
            if diff > 5e-6:
                bad("simulator", a, family, f"max outcome difference {diff:.2e}")

            # 7. limits
            pos = ev["positions"]
            trades = np.diff(np.concatenate([np.zeros_like(pos[:, :1]), pos], axis=1), axis=1)
            if np.abs(pos).max() > 1 + 1e-6 or np.abs(trades).max() > 1 + 1e-6:
                bad("limits", a, family, f"max |position| {np.abs(pos).max():.4f}, max |trade| {np.abs(trades).max():.4f}")

            # 8. prototypes are training states visited by the cell's width-20 Deep Hedging agent
            if fam["kind"] != "dh":
                bank = np.load(d / "prototypes.npz", allow_pickle=False)
                cell_key = tuple(a[k] for k in KEYS)
                if cell_key not in dh_states:
                    dh, _ = rc.load_policy(d.parent, "dh", scaler)
                    with torch.no_grad():
                        tp = rc.rollout(dh, train_ep.tensors(a["state"]), a["mode"], return_paths=True)["positions"].numpy()
                    prev = np.concatenate([np.zeros((train_ep.n, 1, 2)), tp[:, :-1, :]], axis=1)
                    dh_states[cell_key] = np.concatenate([train_ep.exogenous(a["state"]), prev], axis=2)
                states = dh_states[cell_key]
                src = states[bank["source_path"], bank["source_step"]]
                diff = float(np.max(np.abs(src - bank["prototypes_raw"])))
                worst["prototype"] = max(worst["prototype"], diff)
                if diff > 1e-5:
                    bad("prototypes", a, family, f"prototype differs from its source state by {diff:.2e}")
                if not np.array_equal(bank["source_start_date"], train_ep.start_date[bank["source_path"]].astype(str)):
                    bad("prototypes", a, family, "source dates differ from the training episodes")
                if len(bank["prototypes_raw"]) != int(c["k"]):
                    counts["fewer_prototypes_than_k"] += 1
                if policy_k(policy) != len(bank["prototypes_raw"]):
                    bad("prototypes", a, family, "policy and bank disagree on the number of prototypes")
                counts["prototype_banks_checked"] += 1

            failed = ev["metrics"]["he_std"] > unhedged_std
            counts[f"failed__{family}__{a['mode']}__{a['objective']}"] += int(failed)
            counts[f"fits__{family}__{a['mode']}__{a['objective']}"] += 1
            counts["verified"] += 1
        print(f"{time.strftime('%T')} checked {ticker} {liability} ({counts['verified']} fits so far)", flush=True)

    shared = {s: v for s, v in seeds.items() if len(v) > 1}
    if shared:
        problems["shared_seed"] = [f"{s}: {v}" for s, v in shared.items()]

    # 10. test lock
    try:
        rc.Episodes(specs[0]["tickers"][0], specs[0]["liabilities"][0], "test")
        problems["test_lock"].append("test episodes can be loaded without the explicit flag")
    except RuntimeError:
        pass
    test_opened = (phase3_fit.ROOT / study / "final" / "TEST_OPENED.json").exists()

    report = {"study": study, "specs": [str(s) for s in args.specs], "time": time.strftime("%F %T"),
              "expected_fits": counts["expected"], "present": counts["present"], "verified": counts["verified"],
              "largest_differences": worst, "test_opened": test_opened,
              "problems": {k: v for k, v in problems.items()}, "counts": dict(counts)}
    out = Path(args.out) if args.out else phase3_fit.ROOT / study / "verification.json"
    rc.write_json(out, report)

    print(f"\n{counts['verified']} of {counts['expected']} expected fits verified; test opened: {test_opened}")
    print("largest differences:", {k: f"{v:.2e}" for k, v in worst.items()})
    for group in ("cap_reached", "failed"):
        print(group + ":")
        for k in sorted(counts):
            if k.startswith(group + "__") and True:
                total = counts["fits__" + k.split("__", 1)[1]]
                print(f"   {k.split('__', 1)[1]:40s} {counts[k]:4d} of {total}")
    if problems:
        print("\nPROBLEMS")
        for check, msgs in problems.items():
            print(f"  {check}: {len(msgs)}")
            for m in msgs[:8]:
                print("     ", m)
    else:
        print("\nno problem found")
    print("wrote", out)


def policy_k(policy) -> int:
    return int(rc.prototype_outputs(policy).shape[0])


if __name__ == "__main__":
    main()

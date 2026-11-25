#!/usr/bin/env python3
"""Extension X3: do ProtoHedge prototypes correspond to real market events?

Pre-declared in ``paper/revision_2026_11/PHASE3_PROTOCOL.md``, "Extension X3"
(2026-11-19, written before this was run). No model is trained or changed;
the frozen arm-A fits are re-run on the test episodes (opened once on
2026-11-15 by the main evaluation; this script never writes the marker).

    python scripts/revision/rev_events.py run --allow-test --workers 8
    python scripts/revision/rev_events.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import phase2_diagnostics as p2  # noqa: E402
import phase3_fit  # noqa: E402
import rev_faithful as rf  # noqa: E402

TRAIN_WINDOWS = [("2008-09-15", "2009-03-31", "financial crisis"), ("2010-05-06", "2010-06-30", "flash crash, euro crisis"),
                 ("2011-08-01", "2011-10-31", "US downgrade, euro crisis"), ("2015-08-17", "2015-09-30", "China devaluation"),
                 ("2018-02-02", "2018-02-28", "volatility shock"), ("2018-10-01", "2018-12-31", "Q4 2018 selloff")]
TEST_WINDOWS = [("2022-01-21", "2022-03-15", "rate fears, invasion of Ukraine"), ("2022-05-05", "2022-05-20", "May 2022 selloff"),
                ("2022-06-10", "2022-06-24", "June 2022 inflation shock"), ("2022-09-23", "2022-10-21", "autumn 2022 lows"),
                ("2023-03-09", "2023-03-24", "US bank failures"), ("2024-08-01", "2024-08-09", "yen carry unwind")]
OUT = phase3_fit.ROOT / rf.STUDY / "extensions_X" / "events"


def window_label(dates, windows):
    """Label of the window containing each date ('' if none)."""
    dates = np.asarray(dates, dtype=str)
    lab = np.full(dates.shape, "", dtype=object)
    for lo, hi, name in windows:
        lab[(dates >= lo) & (dates <= hi)] = name
    return lab


def decision_dates(cp: pd.DataFrame, indices) -> np.ndarray:
    """Trading date of every decision [n, T] for the given episode indices."""
    sub = cp[cp.episode_index.isin(indices) & (cp.step < rc.N_STEPS)]
    table = sub.pivot(index="episode_index", columns="step", values="date")
    return table.loc[np.asarray(indices)].to_numpy(dtype=str)


def process_unit(job):
    liability, ticker = job
    torch.set_num_threads(1)
    t0 = time.time()
    train_ep = rc.Episodes(ticker, liability, "train")
    ep = rc.Episodes(ticker, liability, "test", allow_test=True)
    scaler = rc.Scaler(train_ep, rf.STATE)
    iv_idx = scaler.names.index("ivol")
    cp = pd.read_csv(rc.PANEL / "contract_paths" / f"{ticker}_contract_paths.csv", usecols=["episode_index", "step", "date"])
    date_of = {(int(e), int(s)): d for e, s, d in cp[["episode_index", "step", "date"]].itertuples(index=False)}
    train_dates = decision_dates(cp, train_ep.index)
    test_dates = decision_dates(cp, ep.index)
    base_rate = float(np.mean(window_label(train_dates, TRAIN_WINDOWS) != ""))
    iv_p95 = float(np.quantile(train_ep.features["ivol"], 0.95))
    ivol = ep.features["ivol"]
    stressed_iv = ivol >= np.quantile(ivol, 0.90)
    stressed_named = window_label(test_dates, TEST_WINDOWS) != ""
    arrays = {"start_date": ep.start_date.astype(str), "stressed_iv": stressed_iv, "stressed_named": stressed_named}
    fits, examples = [], []
    for seed in rf.SEEDS:
        a = {"ticker": ticker, "liability": liability, "mode": rf.MODE, "objective": rf.OBJECTIVE, "state": rf.STATE,
             "seed": seed}
        cell = phase3_fit.cell_dir(rf.STUDY, a)
        for layer, fam in rf.LAYERS.items():
            policy, _ = rc.load_policy(cell, fam, scaler)
            bank = np.load(cell / fam / "prototypes.npz", allow_pickle=False)
            p_dates = np.array([date_of[(int(e), int(s))] for e, s in zip(bank["source_episode_index"], bank["source_step"])])
            p_label = window_label(p_dates, TRAIN_WINDOWS)
            p_iv = bank["prototypes_raw"][:, iv_idx].astype(np.float64)
            masks = {"named": p_label != "", "iv": p_iv >= iv_p95}
            ev = rc.evaluate_policy(policy, ep, rf.STATE, rf.MODE)
            W, P = ev["weights"].astype(np.float64), ev["positions"]
            prev = np.concatenate([np.zeros_like(P[:, :1]), P[:, :-1]], axis=1)
            trade = np.abs(P - prev).sum(axis=2)
            big = trade >= np.quantile(trade, 0.95)
            top1 = W.argmax(axis=2)
            key = f"{layer}__{seed}"
            arrays[f"{key}__big"] = big
            for name, m in masks.items():
                arrays[f"{key}__s_{name}"] = W[..., m].sum(axis=2).astype(np.float32)
                arrays[f"{key}__top1_{name}"] = m[top1]
            fits.append({"liability": liability, "ticker": ticker, "seed": seed, "layer": layer, "K": int(len(p_dates)),
                         "n_stress_named": int(masks["named"].sum()), "n_stress_iv": int(masks["iv"].sum()),
                         "train_base_rate_named": base_rate,
                         "prototypes": [{"date": str(d), "ivol": float(v), "event": str(l), "usage": float(u)}
                                        for d, v, l, u in zip(p_dates, p_iv, p_label, W.reshape(-1, W.shape[-1]).mean(axis=0))]})
            if ticker in ("SPY", "QQQ") and liability == "european_call" and seed == 0:
                i, t = np.unravel_index(int(np.argmax(ivol)), ivol.shape)
                order = np.argsort(-W[i, t])[:3]
                examples.append({"ticker": ticker, "layer": layer, "decision_date": str(test_dates[i, t]),
                                 "decision_ivol": float(ivol[i, t]),
                                 "test_window": str(window_label(test_dates[i:i + 1, t:t + 1], TEST_WINDOWS)[0, 0]),
                                 "top3": [{"date": str(p_dates[k]), "weight": float(W[i, t, k]), "ivol": float(p_iv[k]),
                                           "event": str(p_label[k])} for k in order]})
    d = OUT / liability / ticker
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "arrays.npz", **arrays)
    rc.write_json(d / "fits.json", {"fits": fits, "examples": examples, "iv_p95_train": iv_p95,
                                    "train_base_rate_named": base_rate})
    return f"{time.strftime('%T')} events {liability} {ticker} in {time.time() - t0:.0f}s"


def cmd_run(args):
    rf.check_test_allowed("test", args.allow_test)
    OUT.mkdir(parents=True, exist_ok=True)
    rc.write_json(OUT / "RUN.json", {"started": time.strftime("%F %T"),
                                     "protocol": "PHASE3_PROTOCOL.md: Extension X3 (2026-11-19)"})
    jobs = [(l, t) for l in rf.LIABILITIES for t in p2.TICKERS]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(process_unit, jobs):
            print(msg, flush=True)


def cmd_report(args):
    rows, per, examples = [], {}, []
    for liability in rf.LIABILITIES:
        for ticker in p2.TICKERS:
            d = OUT / liability / ticker
            j = json.load(open(d / "fits.json"))
            rows += j["fits"]
            examples += j["examples"]
            per[(liability, ticker)] = dict(np.load(d / "arrays.npz"))
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "prototypes"} for r in rows])
    rf.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    protos = pd.DataFrame([{**{k: r[k] for k in ("liability", "ticker", "seed", "layer")}, **p}
                           for r in rows for p in r["prototypes"]])
    protos.to_csv(rf.SUMMARY_DIR / "x3_prototype_dates.csv", index=False)

    # X3a representation
    rep = []
    for (liability, layer), g in df.groupby(["liability", "layer"]):
        t = g.groupby("ticker").agg(share=("n_stress_named", lambda x: x.mean() / 10), base=("train_base_rate_named", "first"),
                                    any_named=("n_stress_named", lambda x: (x > 0).mean()),
                                    share_iv=("n_stress_iv", lambda x: x.mean() / 10))
        rep.append({"liability": liability, "layer": layer, "prototype_share_named": t.share.mean(),
                    "training_share_named": t.base.mean(), "ratio": t.share.mean() / t.base.mean(),
                    "banks_with_any_named": t.any_named.mean(), "prototype_share_iv": t.share_iv.mean(),
                    "training_share_iv": 0.05, "ratio_iv": t.share_iv.mean() / 0.05,
                    "tickers_ratio_ge_1_5": int((t.share / t.base >= 1.5).sum())})
    rep = pd.DataFrame(rep)

    # X3b and X3c with calendar-aligned block intervals
    def stat_fn(layer, proto_def, stress_key, kind):
        def fn(z, idx):
            vals = []
            for seed in rf.SEEDS:
                key = f"{layer}__{seed}"
                s = z[f"{key}__s_{proto_def}"][idx]
                if z[f"{key}__s_{proto_def}"].max() == 0:            # bank without a stress prototype
                    continue
                if kind == "weight":
                    st = z[stress_key][idx]
                    x = s
                elif kind == "top1":
                    st = z[stress_key][idx]
                    x = z[f"{key}__top1_{proto_def}"][idx].astype(np.float64)
                else:                                              # large trades
                    st = z[f"{key}__big"][idx]
                    x = z[f"{key}__top1_{proto_def}"][idx].astype(np.float64)
                if st.any() and (~st).any():
                    vals.append(float(x[st].mean() - x[~st].mean()))
            return float(np.mean(vals)) if vals else float("nan")
        return fn

    res = []
    tests = [("X3b_weight", "named", "stressed_iv", "weight"), ("X3b_top1", "named", "stressed_iv", "top1"),
             ("X3c_large_trades", "named", None, "big"),
             ("sec_weight_ivproto", "iv", "stressed_iv", "weight"), ("sec_weight_namedtest", "named", "stressed_named", "weight"),
             ("sec_large_trades_ivproto", "iv", None, "big")]
    for liability in rf.LIABILITIES:
        for layer in rf.LAYERS:
            for i, (name, pdef, skey, kind) in enumerate(tests):
                fn = stat_fn(layer, pdef, skey, kind)
                tick = [t for t in p2.TICKERS if np.isfinite(fn(per[(liability, t)], np.arange(len(per[(liability, t)]["start_date"]))))]
                point, lo, hi = rf.calendar_stat_bootstrap([per[(liability, t)] for t in tick],
                                                           [per[(liability, t)]["start_date"] for t in tick], fn,
                                                           block=20, draws=args.draws, seed=20261119 + i)
                res.append({"liability": liability, "layer": layer, "test": name, "estimate": point, "ci95_low": lo,
                            "ci95_high": hi, "tickers_used": len(tick)})
                print(f"{time.strftime('%T')} {liability} {layer} {name}: {point:+.3f} [{lo:+.3f}, {hi:+.3f}] ({len(tick)} tickers)",
                      flush=True)
    res = pd.DataFrame(res)
    rep.to_csv(rf.SUMMARY_DIR / "x3a_representation.csv", index=False)
    res.to_csv(rf.SUMMARY_DIR / "x3bc_invocation.csv", index=False)
    verdict = {}
    for liability in rf.LIABILITIES:
        for layer in rf.LAYERS:
            r = rep[(rep.liability == liability) & (rep.layer == layer)].iloc[0]
            q = res[(res.liability == liability) & (res.layer == layer)].set_index("test")
            verdict[f"{liability}/{layer}"] = {"X3a": bool(r.ratio >= 1.5),
                                               "X3b": bool(q.loc["X3b_weight", "ci95_low"] > 0),
                                               "X3c": bool(q.loc["X3c_large_trades", "ci95_low"] > 0)}
    rc.write_json(rf.SUMMARY_DIR / "x3_verdict.json", {"verdict": verdict, "examples": examples,
                                                       "written": time.strftime("%F %T")})
    pd.set_option("display.width", 250)
    print(rep.round(3).to_string(index=False))
    print(res.round(3).to_string(index=False))
    print(json.dumps(verdict, indent=1))
    for e in examples:
        print(f"\n{e['ticker']} {e['layer']}: highest-volatility test decision {e['decision_date']} "
              f"(implied vol {e['decision_ivol']:.2f}; window: {e['test_window'] or 'none'})")
        for p in e["top3"]:
            print(f"   weight {p['weight']:.2f}  prototype {p['date']}  implied vol {p['ivol']:.2f}  {p['event'] or '-'}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--allow-test", dest="allow_test", action="store_true")
    r.add_argument("--workers", type=int, default=8)
    r.set_defaults(func=cmd_run)
    q = sub.add_parser("report")
    q.add_argument("--draws", type=int, default=2000)
    q.set_defaults(func=cmd_report)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

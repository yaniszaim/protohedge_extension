#!/usr/bin/env python3
"""Inference for the revision (problems P14, P17).

``panel_bootstrap`` resamples whole calendar dates in circular blocks and keeps,
for every resampled date, the episode of every ticker that starts on it. A
market-wide shock therefore stays in all tickers at once, which the submitted
paper's within-ticker bootstrap did not guarantee. ``calendar=False`` gives the
old scheme (independent blocks per ticker) for comparison.

Run as a script to quantify the difference on the Phase 2 outputs:

    python scripts/revision/rev_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_diagnostics as p2  # noqa: E402

METRICS = {
    "offset_cvar50": lambda x, prem: p2.lower_cvar(x, 0.5),
    "offset_mean": lambda x, prem: x.mean(axis=-1),
    "he_std": lambda x, prem: (x + prem).std(axis=-1),
    "he_cvar05": lambda x, prem: p2.lower_cvar(x + prem, 0.05),
    "he_cvar10": lambda x, prem: p2.lower_cvar(x + prem, 0.10),
}
LOWER_IS_BETTER = {"he_std"}


def block_indices(n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)[None, :]) % n).reshape(-1)[:n]


def panel_bootstrap(outcomes: dict, premiums: list, dates: list, block: int = 20, draws: int = 2000,
                    seed: int = 0, calendar: bool = True, metrics=METRICS) -> dict:
    """outcomes[model] = list over tickers of arrays [seeds, n_i]; premiums and dates
    are lists over tickers of arrays [n_i]. Returns point[model][metric] and
    draws[model][metric] ([draws]), both averaged over seeds, then tickers."""
    rng = np.random.default_rng(seed)
    n_tickers = len(premiums)
    calendar_dates = np.array(sorted(set(np.concatenate([np.asarray(d) for d in dates]))))
    lookup = []
    for d in dates:                                   # position of each calendar date in the ticker, -1 if absent
        pos = {v: i for i, v in enumerate(np.asarray(d))}
        lookup.append(np.array([pos.get(v, -1) for v in calendar_dates]))

    def stat(model, ti, idx):
        x = outcomes[model][ti][:, idx]
        prem = premiums[ti][idx]
        return {m: float(np.mean(fn(x, prem))) for m, fn in metrics.items()}

    point = {m: {k: float(np.mean([stat(m, ti, np.arange(len(premiums[ti])))[k] for ti in range(n_tickers)]))
                 for k in metrics} for m in outcomes}
    out = {m: {k: np.zeros(draws) for k in metrics} for m in outcomes}
    for b in range(draws):
        if calendar:
            cal_idx = block_indices(len(calendar_dates), block, rng)
            idx_by_ticker = [lk[cal_idx][lk[cal_idx] >= 0] for lk in lookup]
        else:
            idx_by_ticker = [block_indices(len(p), block, rng) for p in premiums]
        for m in outcomes:
            acc = {k: 0.0 for k in metrics}
            for ti in range(n_tickers):
                s = stat(m, ti, idx_by_ticker[ti])
                for k in metrics:
                    acc[k] += s[k] / n_tickers
            for k in metrics:
                out[m][k][b] = acc[k]
    return {"point": point, "draws": out}


def contrast(result: dict, candidate: str, benchmark: str, metric: str) -> dict:
    d = result["draws"][candidate][metric] - result["draws"][benchmark][metric]
    est = result["point"][candidate][metric] - result["point"][benchmark][metric]
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = float(min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0)) + 1.0 / len(d)))
    return {"candidate": candidate, "benchmark": benchmark, "metric": metric, "difference": float(est),
            "ci95_low": float(lo), "ci95_high": float(hi), "ci_width": float(hi - lo), "p_boot": p}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def phase2_inputs(liability: str, models):
    from rev_core import PANEL
    outcomes, premiums, dates = {m: [] for m in models}, [], []
    for ticker in p2.TICKERS:
        z = np.load(p2.OUT / "per_task" / liability / f"{ticker}.npz")
        for m in models:
            outcomes[m].append(z[f"offset__{m}"].astype(np.float64))
        premiums.append(z["premium"].astype(np.float64))
        splits = np.load(PANEL / "episodes" / f"{ticker}_chronological_splits.npz")
        meta = pd.read_csv(PANEL / "episode_metadata" / f"{ticker}_episode_metadata.csv").set_index("episode_index")
        dates.append(meta.loc[np.asarray(splits["test"]), "start_date"].to_numpy().astype(str))
    return outcomes, premiums, dates


def main():
    """How much do intervals change with calendar-aligned blocks? (Phase 2 models, test period.)"""
    models = ["dh_hybrid", "ph_tail_hybrid", "ph_fixed", "liability_delta", "static_oce"]
    pairs = [("ph_tail_hybrid", "dh_hybrid"), ("ph_fixed", "dh_hybrid"),
             ("dh_hybrid", "liability_delta"), ("ph_tail_hybrid", "liability_delta")]
    rows = []
    for liability in p2.LIABILITIES:
        outcomes, premiums, dates = phase2_inputs(liability, models)
        for calendar, block in ((False, 20), (True, 10), (True, 20), (True, 40)):
            res = panel_bootstrap(outcomes, premiums, dates, block=block, draws=1000, seed=7, calendar=calendar)
            for cand, bench in pairs:
                for metric in ("offset_cvar50", "he_std", "he_cvar05"):
                    rows.append({"liability": liability, "scheme": "calendar" if calendar else "within-ticker",
                                 "block": block, **contrast(res, cand, bench, metric)})
            print(liability, calendar, block, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(p2.OUT / "bootstrap_scheme_comparison.csv", index=False)
    base = df[df.scheme == "within-ticker"].set_index(["liability", "candidate", "benchmark", "metric"]).ci_width
    cal = df[(df.scheme == "calendar") & (df.block == 20)].set_index(["liability", "candidate", "benchmark", "metric"]).ci_width
    ratio = (cal / base).rename("width_ratio_calendar_vs_within")
    pd.set_option("display.width", 200)
    print(ratio.round(2).to_string())
    print("median ratio", round(float(ratio.median()), 2), "max", round(float(ratio.max()), 2))
    wide = df.pivot_table(index=["liability", "candidate", "benchmark", "metric"], columns=["scheme", "block"],
                          values="ci_width")
    print((wide * 1e4).round(1).to_string())


if __name__ == "__main__":
    main()

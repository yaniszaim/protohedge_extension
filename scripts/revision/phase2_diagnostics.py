#!/usr/bin/env python3
"""Phase 2: diagnose the completed v4 historical results without any training.

Every saved v4 model that the submitted paper reports (Deep Hedging and the two
validation-selected ProtoHedge models, three seeds each), plus one fixed
original-style ProtoHedge configuration, is reloaded and re-evaluated on the
test episodes. Rule-based baselines are computed on the same episodes with the
same accounting:

    unhedged            no hedge
    spot_delta_listed   spot position = vendor delta of the LISTED call (the
                        paper's "Spot-Delta" baseline)
    liability_delta     spot position = model delta of the LIABILITY itself
                        (Black-Scholes for the European call, Levy moment
                        matching for the arithmetic Asian call), using the
                        listed call's implied volatility
    hold_share          buy one share at the first step and hold
    hold_call           buy one listed call at the first step and hold
    hold_both           both of the above
    static_oce          best constant (spot, call) position on the TRAINING
                        episodes under the training objective (CVaR@50%)
    static_minvar       constant position with the lowest TRAINING variance

Outcomes are reported premium-excluded (the paper's "liability offset") and
premium-inclusive (model premium of the liability at inception added).

Nothing under the v4 run tree is modified. Outputs:
``.deephedging_real_runs/revision_v1/phase2_diagnostics/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

V4 = REPO / ".deephedging_real_runs" / "submission_rerun_scientific_v4"
OUT = REPO / ".deephedging_real_runs" / "revision_v1" / "phase2_diagnostics"
TICKERS = ["AAPL", "IWM", "NVDA", "QQQ", "SPY", "TLT", "XLE", "XLF", "XLK", "XLV"]
LIABILITIES = ["european_call", "asian_call"]
SEEDS = [1234, 2345, 3456]
FIXED_PH = "proto_source=vanilla_k=100_weighted=0_learnw=0"  # declared before evaluation
N_STEPS = 20
DT = 1.0 / 252.0
BOUND_TOL = 1e-4
BOOT_DRAWS = 2000
BOOT_BLOCK = 20

SENS = REPO / ".deephedging_real_runs" / "checkpoint_selection_sensitivity_v1"
V4_LEARNED = ["dh", "ph_mean", "ph_tail", "ph_fixed"]
# "hybrid" = the submitted manuscript's headline provenance: the 36 fits that
# were retrained with unpenalised validation-OCE checkpointing replace their
# v4 counterparts; every other fit is the v4 fit.
HYBRID = {"dh": "dh_hybrid", "ph_mean": "ph_mean_hybrid", "ph_tail": "ph_tail_hybrid"}
LEARNED = V4_LEARNED + list(HYBRID.values())
RULES = ["unhedged", "spot_delta_listed", "liability_delta", "hold_share", "hold_call",
         "hold_both", "static_oce", "static_minvar"]
MODEL_ORDER = RULES[:3] + ["dh", "dh_hybrid", "ph_mean", "ph_mean_hybrid", "ph_tail", "ph_tail_hybrid", "ph_fixed"] + RULES[3:]


# --------------------------------------------------------------------------
# Liability model values (prices normalised by the initial spot, strike = 1)
# --------------------------------------------------------------------------

def european_price_delta(spot, sigma, tau, strike=1.0):
    v = np.maximum(sigma * np.sqrt(tau), 1e-10)
    d1 = (np.log(spot / strike) + 0.5 * v * v) / v
    d2 = d1 - v
    return spot * norm.cdf(d1) - strike * norm.cdf(d2), norm.cdf(d1)


def asian_price_delta(spot, sigma, running_sum, n_done, n_total=N_STEPS, strike=1.0, dt=DT):
    """Levy (lognormal moment-matching) value of an arithmetic Asian call.

    ``n_done`` fixings are already in ``running_sum``; the remaining
    ``n_total - n_done`` fixings fall 1, 2, ... steps ahead. Zero rates.
    """
    spot = np.asarray(spot, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n_left = int(n_total - n_done)
    j = np.arange(1, n_left + 1, dtype=np.float64)
    min_jk = np.minimum(j[:, None], j[None, :])
    a = (sigma ** 2) * dt
    m1 = spot * n_left
    m2 = spot ** 2 * np.exp(a[:, None, None] * min_jk[None, :, :]).sum(axis=(1, 2))
    k_sum = n_total * strike - np.asarray(running_sum, dtype=np.float64)
    v = np.sqrt(np.maximum(np.log(m2 / m1 ** 2), 1e-20))
    safe_k = np.where(k_sum > 0, k_sum, 1.0)
    d1 = (np.log(m1 / safe_k) + 0.5 * v * v) / v
    d2 = d1 - v
    price = np.where(k_sum > 0, (m1 * norm.cdf(d1) - k_sum * norm.cdf(d2)) / n_total,
                     (m1 - k_sum) / n_total)
    delta = np.where(k_sum > 0, norm.cdf(d1), 1.0) * n_left / n_total
    return price, delta


def liability_premium_and_delta(world):
    """Model premium at inception [n] and liability delta by step [n, T]."""
    per_step = world.data.features.per_step
    spot = np.asarray(per_step["spot"], dtype=np.float64)
    ivol = np.asarray(per_step["ivol"], dtype=np.float64)
    n, n_steps = spot.shape
    delta = np.zeros((n, n_steps))
    if world.liability_type == "european_call":
        premium, _ = european_price_delta(spot[:, 0], ivol[:, 0], n_steps * DT)
        for t in range(n_steps):
            _, delta[:, t] = european_price_delta(spot[:, t], ivol[:, t], (n_steps - t) * DT)
    else:
        running_avg = np.asarray(per_step["asian_running_average"], dtype=np.float64)
        premium, _ = asian_price_delta(spot[:, 0], ivol[:, 0], np.zeros(n), 0)
        for t in range(n_steps):
            # fixings at observations 1..t are known at decision step t
            _, delta[:, t] = asian_price_delta(spot[:, t], ivol[:, t], running_avg[:, t] * t, t)
    return premium, delta


# --------------------------------------------------------------------------
# Accounting identical to gym_torch.DeepHedgingGymTorch (inventory-step mode)
# --------------------------------------------------------------------------

def simulate_targets(market, targets):
    hedges = np.asarray(market["hedges"], dtype=np.float64)
    tc = np.asarray(market["cost"], dtype=np.float64)
    payoff = np.asarray(market["payoff"], dtype=np.float64).reshape(-1)
    ub_a, lb_a = np.asarray(market["ubnd_a"]), np.asarray(market["lbnd_a"])
    ub_d, lb_d = np.asarray(market["ubnd_delta"]), np.asarray(market["lbnd_delta"])
    n, n_steps, n_inst = hedges.shape
    pos = np.zeros((n, n_inst))
    positions = np.zeros((n, n_steps, n_inst))
    pnl, cost = np.zeros(n), np.zeros(n)
    for t in range(n_steps):
        action = np.clip(targets[:, t, :] - pos, lb_a[:, t, :], ub_a[:, t, :])
        action = np.clip(pos + action, lb_d[:, t, :], ub_d[:, t, :]) - pos
        pos = pos + action
        pnl += (pos * hedges[:, t, :]).sum(axis=1)
        cost += (np.abs(action) * tc[:, t, :]).sum(axis=1)
        positions[:, t, :] = pos
    return {"offset": payoff + pnl - cost, "cost": cost, "positions": positions}


def constant_targets(n, spot_pos, call_pos):
    targets = np.zeros((n, N_STEPS, 2))
    targets[:, :, 0] = spot_pos
    targets[:, :, 1] = call_pos
    return targets


def lower_cvar(x, alpha):
    """Mean of the worst ``alpha`` share of outcomes (exact with ties).

    Equal to max_y E[min(x + y, 0) / alpha - y], the OCE form the models are
    trained on. ``outcome_metrics.empirical_lower_tail_cvar`` averages every
    value <= the quantile instead, which is wrong when many outcomes tie at the
    quantile (a buy-and-hold share against a short call ties on every path that
    finishes in the money).
    """
    xs = np.sort(np.asarray(x, dtype=np.float64), axis=-1)
    n = xs.shape[-1]
    k = alpha * n
    f = int(np.floor(k + 1e-9))
    total = xs[..., :f].sum(axis=-1)
    if f < n and k - f > 1e-9:
        total = total + (k - f) * xs[..., f]
    return total / k


def select_static_positions(market):
    """Grid search of constant positions on the training episodes."""
    hedges = np.asarray(market["hedges"], dtype=np.float64)
    tc = np.asarray(market["cost"], dtype=np.float64)
    payoff = np.asarray(market["payoff"], dtype=np.float64).reshape(-1)
    gain = hedges.sum(axis=1)                      # [n, 2] total move of each instrument
    grid = np.round(np.linspace(-1.0, 1.0, 21), 2)
    a, b = [g.reshape(-1) for g in np.meshgrid(grid, grid, indexing="ij")]
    offsets = (payoff[None, :] + a[:, None] * gain[None, :, 0] + b[:, None] * gain[None, :, 1]
               - np.abs(a)[:, None] * tc[None, :, 0, 0] - np.abs(b)[:, None] * tc[None, :, 0, 1])
    cvar50 = lower_cvar(offsets, 0.5)
    std = offsets.std(axis=1)
    i_oce, i_var = int(np.argmax(cvar50)), int(np.argmin(std))
    return {
        "static_oce": (float(a[i_oce]), float(b[i_oce])),
        "static_minvar": (float(a[i_var]), float(b[i_var])),
        "train_cvar50_static_oce": float(cvar50[i_oce]),
        "train_cvar50_unhedged": float(cvar50[int(np.argmin(np.abs(a) + np.abs(b)))]),
        "train_std_static_minvar": float(std[i_var]),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def outcome_metrics(offset, premium):
    he = offset + premium
    return {
        "offset_mean": float(offset.mean()),
        "offset_std": float(offset.std()),
        "offset_cvar50": float(lower_cvar(offset, 0.5)),
        "offset_cvar05": float(lower_cvar(offset, 0.05)),
        "he_mean": float(he.mean()),
        "he_std": float(he.std()),
        "he_rmse": float(np.sqrt(np.mean(he ** 2))),
        "he_cvar50": float(lower_cvar(he, 0.5)),
        "he_cvar05": float(lower_cvar(he, 0.05)),
    }


def position_metrics(positions, listed_delta, liab_delta):
    trades = np.diff(np.concatenate([np.zeros_like(positions[:, :1]), positions], axis=1), axis=1)
    abs_trades = np.abs(trades)
    total_delta = positions[:, :, 0] + positions[:, :, 1] * listed_delta
    net = total_delta - liab_delta
    flat_h, flat_l = total_delta.reshape(-1), liab_delta.reshape(-1)
    corr = float(np.corrcoef(flat_h, flat_l)[0, 1]) if flat_h.std() > 1e-9 else float("nan")
    total_trading = abs_trades.sum()
    return {
        "pos_spot_mean": float(positions[:, :, 0].mean()),
        "pos_call_mean": float(positions[:, :, 1].mean()),
        "pos_abs_mean": float(np.abs(positions).mean()),
        "spot_at_upper": float(np.mean(positions[:, :, 0] >= 1 - BOUND_TOL)),
        "spot_at_lower": float(np.mean(positions[:, :, 0] <= -1 + BOUND_TOL)),
        "call_at_upper": float(np.mean(positions[:, :, 1] >= 1 - BOUND_TOL)),
        "call_at_lower": float(np.mean(positions[:, :, 1] <= -1 + BOUND_TOL)),
        "first_step_trade_share": float(abs_trades[:, 0].sum() / total_trading) if total_trading > 0 else float("nan"),
        "later_turnover_per_path": float(abs_trades[:, 1:].sum(axis=(1, 2)).mean()),
        "hedge_delta_mean": float(total_delta.mean()),
        "net_delta_mean": float(net.mean()),
        "net_delta_abs_mean": float(np.abs(net).mean()),
        "corr_hedge_delta_liability_delta": corr,
    }


def prototype_metrics(weights):
    w = np.asarray(weights, dtype=np.float64).reshape(-1, weights.shape[-1])
    entropy = -(w * np.log(np.maximum(w, 1e-30))).sum(axis=1)
    top = np.sort(w, axis=1)[:, ::-1]
    return {
        "proto_effective_n": float(np.exp(entropy).mean()),
        "proto_top1_weight": float(top[:, 0].mean()),
        "proto_top3_weight": float(top[:, :3].sum(axis=1).mean()),
        "proto_used_share": float(np.mean(w.max(axis=0) > 0.05)),
    }


# --------------------------------------------------------------------------
# One task = one liability x ticker
# --------------------------------------------------------------------------

def task_dir(liability, ticker):
    return V4 / liability / f"{ticker}_full_grid_800_epochs"


def slug(name):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def run_task(liability, ticker, out_dir):
    import torch
    from deephedging.real_data_analysis_torch import load_model_artifact
    from deephedging.real_data_sweep_torch import evaluate_gym, world_cfg
    from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch
    from deephedging.historical_evidence import frozen_oce_cvar50

    out_path = out_dir / "per_task" / liability / f"{ticker}.npz"
    rows_path = out_dir / "per_task" / liability / f"{ticker}_rows.json"
    if out_path.exists() and rows_path.exists():
        return json.load(open(rows_path))

    tdir = task_dir(liability, ticker)
    selected = pd.read_csv(tdir / "validation_selected_models.csv").set_index("selection")
    model_names = {
        "dh": "vanilla",
        "ph_mean": selected.loc["best_screened_proto_mean", "model"],
        "ph_tail": selected.loc["best_screened_proto_cvar05", "model"],
        "ph_fixed": FIXED_PH,
    }
    saved = pd.read_csv(tdir / "sweep_metrics.csv")
    art_root = tdir / "model_artifacts"

    meta0 = json.load(open(art_root / f"seed_{SEEDS[0]}" / "risk_cvar" / "vanilla" / "artifact_metadata.json"))
    data_path, world_kwargs = meta0["data_path"], dict(meta0["world_kwargs"])
    splits = meta0["split_indices"]

    def build_world(split):
        return RealWorld_Spot_ATM_Torch(world_cfg(data_path=data_path, indices=splits[split], **world_kwargs))

    test_world, train_world = build_world("test"), build_world("train")
    market = test_world.data.market
    n = test_world.nSamples
    listed_delta = np.asarray(test_world.data.features.per_step["call_delta"], dtype=np.float64)
    premium, liab_delta = liability_premium_and_delta(test_world)

    static = select_static_positions(train_world.data.market)
    targets = {
        "unhedged": constant_targets(n, 0.0, 0.0),
        "hold_share": constant_targets(n, 1.0, 0.0),
        "hold_call": constant_targets(n, 0.0, 1.0),
        "hold_both": constant_targets(n, 1.0, 1.0),
        "static_oce": constant_targets(n, *static["static_oce"]),
        "static_minvar": constant_targets(n, *static["static_minvar"]),
    }
    for name, d in (("spot_delta_listed", listed_delta), ("liability_delta", liab_delta)):
        tg = np.zeros((n, N_STEPS, 2))
        tg[:, :, 0] = d
        targets[name] = tg

    rows, arrays = [], {"premium": premium.astype(np.float32),
                        "liability_delta": liab_delta.astype(np.float32),
                        "listed_delta": listed_delta.astype(np.float32)}
    base = {"liability": liability, "ticker": ticker, "n_test": n}

    for name, tg in targets.items():
        sim = simulate_targets(market, tg)
        row = {**base, "model": name, "seed": -1, "config": "",
               **outcome_metrics(sim["offset"], premium),
               "cost_mean": float(sim["cost"].mean()),
               **position_metrics(sim["positions"], listed_delta, liab_delta)}
        if name.startswith("static"):
            row["config"] = "spot=%+.1f call=%+.1f" % static[name]
        rows.append(row)
        arrays[f"offset__{name}"] = sim["offset"].astype(np.float32)[None, :]
        arrays[f"profile__{name}"] = sim["positions"].mean(axis=0).astype(np.float32)[None]

    jobs = [(key, name, False) for key, name in model_names.items()]
    jobs += [(HYBRID[key], model_names[key], True) for key in HYBRID]
    n_retrained = {}
    for key, model_name, hybrid in jobs:
        offsets, profiles = [], []
        for seed in SEEDS:
            adir = art_root / f"seed_{seed}" / "risk_cvar" / slug(model_name)
            retrained = False
            if hybrid:
                alt = (SENS / liability / ticker / f"seed_{seed}" / slug(model_name) / "artifact"
                       / f"seed_{seed}" / "risk_cvar" / slug(model_name))
                if (alt / "gym_state.pt").exists():
                    adir, retrained = alt, True
                    n_retrained[key] = n_retrained.get(key, 0) + 1
            bundle = load_model_artifact(adir)
            _, result, _ = evaluate_gym(bundle["gym"], data_path, splits["test"], f"{key} test", world_kwargs)
            offset = result["liability_offset"].numpy().astype(np.float64).reshape(-1)
            positions = result["deltas"].numpy().astype(np.float64)
            state = torch.load(adir / "gym_state.pt", map_location="cpu", weights_only=True)
            y = float(state["objective.utility.y"])
            hist = bundle["metadata"].get("history", {})
            row = {**base, "model": key, "seed": seed, "config": model_name,
                   **outcome_metrics(offset, premium),
                   "frozen_oce": frozen_oce_cvar50(offset, y),
                   "cost_mean": float(result["cost"].numpy().mean()),
                   "selected_epoch": hist.get("best_epoch"), "retrained": retrained,
                   **position_metrics(positions, listed_delta, liab_delta)}
            if "prototype_weights" in result:
                row.update(prototype_metrics(result["prototype_weights"].numpy()))
            sv = saved[(saved.model == model_name) & (saved.seed == seed) & (saved.split == "test")]
            if len(sv) == 1 and not hybrid:
                row["saved_offset_mean"] = float(sv.liability_offset_mean.iloc[0])
                row["saved_offset_std"] = float(sv.liability_offset_std.iloc[0])
            rows.append(row)
            offsets.append(offset.astype(np.float32))
            profiles.append(positions.mean(axis=0).astype(np.float32))
            if key in ("dh", "ph_tail", "dh_hybrid", "ph_tail_hybrid"):
                arrays[f"positions__{key}__{seed}"] = positions.astype(np.float16)
        arrays[f"offset__{key}"] = np.stack(offsets)
        arrays[f"profile__{key}"] = np.stack(profiles)

    for r in rows:
        r.update({f"static_{k}": v for k, v in static.items() if not isinstance(v, tuple)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_path)
    with open(str(rows_path) + ".tmp", "w") as f:
        json.dump(rows, f, default=float)
    os.replace(str(rows_path) + ".tmp", rows_path)
    return rows


# --------------------------------------------------------------------------
# Panel aggregation and bootstrap
# --------------------------------------------------------------------------

BOOT_METRICS = ["offset_mean", "offset_std", "offset_cvar50", "offset_cvar05", "he_std", "he_cvar05"]
CONTRASTS = [
    ("ph_tail_hybrid", "dh_hybrid"), ("ph_mean_hybrid", "dh_hybrid"), ("ph_fixed", "dh_hybrid"),
    ("dh_hybrid", "liability_delta"), ("ph_tail_hybrid", "liability_delta"),
    ("dh_hybrid", "static_oce"), ("ph_tail_hybrid", "static_oce"),
    ("dh_hybrid", "static_minvar"), ("ph_tail_hybrid", "static_minvar"),
    ("dh_hybrid", "hold_share"), ("dh_hybrid", "unhedged"), ("ph_tail_hybrid", "unhedged"),
    ("ph_tail", "dh"), ("ph_mean", "dh"), ("ph_fixed", "dh"),
    ("dh", "liability_delta"), ("ph_tail", "liability_delta"), ("ph_fixed", "liability_delta"),
    ("dh", "spot_delta_listed"), ("ph_tail", "spot_delta_listed"),
    ("dh", "static_oce"), ("ph_tail", "static_oce"), ("ph_fixed", "static_oce"),
    ("dh", "hold_share"), ("liability_delta", "spot_delta_listed"), ("liability_delta", "unhedged"),
    ("dh", "unhedged"), ("ph_tail", "unhedged"),
]


def boot_metric_matrix(x, premium):
    """x: [draws, n] resampled offsets. Returns {metric: [draws]}."""
    he = x + premium
    xs = np.sort(x, axis=1)
    return {
        "offset_mean": x.mean(axis=1), "offset_std": x.std(axis=1),
        "offset_cvar50": lower_cvar(xs, 0.5), "offset_cvar05": lower_cvar(xs, 0.05),
        "he_std": he.std(axis=1), "he_cvar05": lower_cvar(he, 0.05),
    }


def bootstrap(out_dir):
    from deephedging.outcome_metrics import circular_block_bootstrap_indices

    records = []
    for liability in LIABILITIES:
        point = {m: {k: [] for k in BOOT_METRICS} for m in MODEL_ORDER}
        draws = {m: {k: [] for k in BOOT_METRICS} for m in MODEL_ORDER}
        for ti, ticker in enumerate(TICKERS):
            z = np.load(out_dir / "per_task" / liability / f"{ticker}.npz")
            premium = z["premium"].astype(np.float64)
            n = len(premium)
            idx = circular_block_bootstrap_indices(n, BOOT_BLOCK, BOOT_DRAWS,
                                                   random_state=20261113 + 100 * ti + (0 if liability == "european_call" else 7))
            for m in MODEL_ORDER:
                off = z[f"offset__{m}"].astype(np.float64)        # [seeds or 1, n]
                pm = [boot_metric_matrix(o[None, :], premium[None, :]) for o in off]
                bm = [boot_metric_matrix(o[idx], premium[idx]) for o in off]
                for k in BOOT_METRICS:
                    point[m][k].append(np.mean([p[k][0] for p in pm]))
                    draws[m][k].append(np.mean([b[k] for b in bm], axis=0))
        for cand, bench in CONTRASTS:
            for k in BOOT_METRICS:
                est = float(np.mean(point[cand][k]) - np.mean(point[bench][k]))
                d = np.mean(draws[cand][k], axis=0) - np.mean(draws[bench][k], axis=0)
                lo, hi = np.quantile(d, [0.025, 0.975])
                records.append({"liability": liability, "candidate": cand, "benchmark": bench,
                                "metric": k, "difference": est, "ci95_low": float(lo),
                                "ci95_high": float(hi), "tickers_candidate_better": int(np.sum(
                                    (np.array(point[cand][k]) - np.array(point[bench][k]))
                                    * (-1 if k.endswith("std") else 1) > 0))})
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "contrasts_bootstrap.csv", index=False)
    return df


def profiles_table(out_dir):
    recs = []
    for liability in LIABILITIES:
        for m in MODEL_ORDER:
            per_ticker = []
            for ticker in TICKERS:
                z = np.load(out_dir / "per_task" / liability / f"{ticker}.npz")
                per_ticker.append(z[f"profile__{m}"].astype(np.float64).mean(axis=0))   # [T, 2]
            prof = np.mean(per_ticker, axis=0)
            for t in range(N_STEPS):
                recs.append({"liability": liability, "model": m, "step": t,
                             "spot_position": prof[t, 0], "call_position": prof[t, 1]})
    df = pd.DataFrame(recs)
    df.to_csv(out_dir / "position_profiles.csv", index=False)
    return df


def classify_dh(rows):
    d = rows[rows.model.isin(LEARNED)].copy()

    def label(r):
        if r.selected_epoch is not None and r.selected_epoch <= 0:
            return "initialisation or first epoch"
        if r.first_step_trade_share >= 0.9:
            return "static (>=90% of trading at first step)"
        return "dynamic"
    d["fit_type"] = d.apply(label, axis=1)
    return d


def aggregate(out_dir, all_rows):
    rows = pd.DataFrame(all_rows)
    rows.to_csv(out_dir / "rows_by_ticker_seed.csv", index=False)
    num = [c for c in rows.columns if rows[c].dtype.kind in "fi" and c not in ("seed", "n_test")]
    by_ticker = rows.groupby(["liability", "ticker", "model"], sort=False)[num].mean().reset_index()
    by_ticker.to_csv(out_dir / "scoreboard_by_ticker.csv", index=False)
    panel = by_ticker.groupby(["liability", "model"], sort=False)[num].mean().reset_index()
    panel["model"] = pd.Categorical(panel["model"], MODEL_ORDER, ordered=True)
    panel = panel.sort_values(["liability", "model"], ascending=[False, True])
    panel.to_csv(out_dir / "scoreboard_panel.csv", index=False)

    fits = classify_dh(rows)
    fits.to_csv(out_dir / "learned_fit_classification.csv", index=False)
    fit_counts = fits.groupby(["liability", "model", "fit_type"]).size().unstack(fill_value=0)
    fit_counts.to_csv(out_dir / "learned_fit_counts.csv")

    prov = rows.dropna(subset=["saved_offset_mean"]) if "saved_offset_mean" in rows else rows.iloc[:0]
    prov_summary = {
        "rows_checked": int(len(prov)),
        "max_abs_diff_mean": float((prov.offset_mean - prov.saved_offset_mean).abs().max()) if len(prov) else None,
        "max_abs_diff_std": float((prov.offset_std - prov.saved_offset_std).abs().max()) if len(prov) else None,
    }
    static = rows[rows.model.isin(["static_oce", "static_minvar"])][["liability", "ticker", "model", "config"]]
    static.to_csv(out_dir / "static_positions_selected_on_train.csv", index=False)
    return rows, by_ticker, panel, fit_counts, prov_summary


def headline_check(out_dir, rows):
    """Re-derive the submitted table's frozen-threshold OCE differences."""
    out = {}
    for liability in LIABILITIES:
        d = rows[(rows.liability == liability) & rows.model.isin(LEARNED)]
        t = d.groupby(["ticker", "model"]).frozen_oce.mean().unstack()
        out[liability] = {f"{m}_minus_dh_v4": float((t[m] - t["dh"]).mean()) for m in ("ph_mean", "ph_tail", "ph_fixed")}
        out[liability].update({
            "ph_mean_hybrid_minus_dh_hybrid": float((t["ph_mean_hybrid"] - t["dh_hybrid"]).mean()),
            "ph_tail_hybrid_minus_dh_hybrid": float((t["ph_tail_hybrid"] - t["dh_hybrid"]).mean()),
            "ph_fixed_minus_dh_hybrid": float((t["ph_fixed"] - t["dh_hybrid"]).mean()),
            "tickers_ph_fixed_better_than_dh_hybrid": int(((t["ph_fixed"] - t["dh_hybrid"]) > 0).sum()),
            "retrained_fits": int(d[d.model.isin(list(HYBRID.values()))].retrained.sum()),
        })
    return out


def selftest():
    rng = np.random.default_rng(0)
    sigma, n_paths = 0.25, 400_000
    z = rng.standard_normal((n_paths, N_STEPS))
    s = np.exp(np.cumsum(-0.5 * sigma ** 2 * DT + sigma * np.sqrt(DT) * z, axis=1))
    mc_asian = np.maximum(s.mean(axis=1) - 1.0, 0.0).mean()
    mc_euro = np.maximum(s[:, -1] - 1.0, 0.0).mean()
    levy, levy_delta = asian_price_delta(np.ones(1), np.full(1, sigma), np.zeros(1), 0)
    bs, bs_delta = european_price_delta(np.ones(1), np.full(1, sigma), N_STEPS * DT)
    bump = 1e-4
    up, _ = asian_price_delta(np.full(1, 1 + bump), np.full(1, sigma), np.zeros(1), 0)
    print(f"European: MC {mc_euro:.5f}  BS {bs[0]:.5f}  delta {bs_delta[0]:.4f}")
    print(f"Asian:    MC {mc_asian:.5f}  Levy {levy[0]:.5f}  delta {levy_delta[0]:.4f}  "
          f"finite-difference delta {(up[0] - levy[0]) / bump:.4f}")
    assert abs(mc_euro - bs[0]) < 3e-4 and abs(mc_asian - levy[0]) < 3e-4
    assert abs((up[0] - levy[0]) / bump - levy_delta[0]) < 2e-3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--only", default=None, help="liability:ticker for a single task")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(l, t) for l in LIABILITIES for t in TICKERS]
    if args.only:
        l, t = args.only.split(":")
        tasks = [(l, t)]
    all_rows, t0 = [], time.time()
    for i, (liability, ticker) in enumerate(tasks):
        all_rows.extend(run_task(liability, ticker, out_dir))
        print(f"[{i + 1}/{len(tasks)}] {liability} {ticker} done ({time.time() - t0:.0f}s)", flush=True)
    if args.only:
        df = pd.DataFrame(all_rows)
        cols = ["model", "seed", "offset_mean", "offset_std", "offset_cvar50", "offset_cvar05", "he_std",
                "pos_spot_mean", "pos_call_mean", "first_step_trade_share", "net_delta_mean"]
        print(df[cols].to_string())
        if "saved_offset_mean" in df:
            print("max diff vs saved:", (df.offset_mean - df.saved_offset_mean).abs().max(),
                  (df.offset_std - df.saved_offset_std).abs().max())
        return
    rows, by_ticker, panel, fit_counts, prov = aggregate(out_dir, all_rows)
    contrasts = bootstrap(out_dir)
    profiles_table(out_dir)
    summary = {"provenance_check": prov, "headline_frozen_oce": headline_check(out_dir, rows),
               "generated": time.strftime("%F %T"), "bootstrap": {"draws": BOOT_DRAWS, "block": BOOT_BLOCK},
               "fixed_ph_config": FIXED_PH}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print(fit_counts)


if __name__ == "__main__":
    main()

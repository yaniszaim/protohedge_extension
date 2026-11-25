#!/usr/bin/env python3
"""Extensions X1 (are ProtoHedge's explanations faithful?) and X2 (are they useful?).

Pre-declared in ``paper/revision_2026_11/PHASE3_PROTOCOL.md``, section
"Extensions X1 ... and X2 ...", written before any of these analyses were run.
No model is trained or changed: the frozen arm-A fits are re-run on the
validation or test episodes, with their prototype weights edited (deletion,
top-k) or their output gated (fallback to the liability delta hedge).

    python scripts/revision/rev_faithful.py run --split val --workers 8
    python scripts/revision/rev_faithful.py run --split test --allow-test --workers 8
    python scripts/revision/rev_faithful.py report --split val
    python scripts/revision/rev_faithful.py report --split test
    python scripts/revision/rev_faithful.py orig --split test --allow-test     # X2d (arm F)

``--split test`` refuses to run unless the main evaluation already opened the
test episodes (``M1_main/final/TEST_OPENED.json``); it never writes that marker.
Per-fit outputs go to ``.deephedging_real_runs/revision_v1/phase3/M1_main/extensions_X/<split>/``,
summaries to ``paper/revision_2026_11/phase3/extensions/``.
"""

from __future__ import annotations

import argparse
import hashlib
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
import rev_inference as inf  # noqa: E402

STUDY, MODE, OBJECTIVE, STATE = "M1_main", "residual", "var_he", "s1"
LAYERS = {"ph": "ph_k10", "phd": "phd_k10"}          # released encoder layer, plain-distance layer
DH_FAMILY = "dh_w128"
SEEDS = (0, 1, 2, 3, 4)
LIABILITIES = ("european_call", "asian_call")
MARKET = ["spot", "call", "ivol", "listed_log_moneyness"]          # + asian_moneyness for the Asian call
NOISE_STD = 0.10
N_RANDOM = 20
SUMMARY_DIR = rc.REPO / "paper" / "revision_2026_11" / "phase3" / "extensions"


def out_root(split: str) -> Path:
    return phase3_fit.ROOT / STUDY / "extensions_X" / split


def market_names(liability: str):
    return MARKET + (["asian_moneyness"] if liability == "asian_call" else [])


def check_test_allowed(split: str, allow_test: bool):
    if split == "test":
        if not allow_test or not (phase3_fit.ROOT / STUDY / "final" / "TEST_OPENED.json").exists():
            raise SystemExit("The test episodes are opened by phase3_final.py first; then pass --allow-test.")


# --------------------------------------------------------------------------
# Prototype mixtures with edited weights, and a rollout with a fallback mask
# --------------------------------------------------------------------------

def logits(policy, x):
    """Pre-softmax similarity scores of a prototype policy, [B, K]."""
    if isinstance(policy, rc.PHPolicy):
        _, d = policy.agent.compute_similarity(x, return_distances=True)
        return -d
    z = (x - policy.mean) / policy.std
    return -(((z[:, None, :] - policy.prototypes[None, :, :]) ** 2) * policy.feature_weights).sum(dim=-1)


class Edited(torch.nn.Module):
    """The policy's own mixture, sum_k w_k a_k, with some prototypes removed (``drop``)
    or only the ``topk`` heaviest kept; weights are renormalised (softmax over the rest)."""

    def __init__(self, base, drop=None, topk=None):
        super().__init__()
        self.base, self.drop, self.topk = base, drop, topk
        self.A = torch.tensor(rc.prototype_outputs(base), dtype=torch.float32)

    def forward(self, x, return_weights=False):
        lg = logits(self.base, x).clone()
        if self.drop is not None:
            lg[:, list(self.drop)] = -float("inf")
        if self.topk is not None:
            idx = lg.topk(self.topk, dim=1).indices
            mask = torch.full_like(lg, -float("inf"))
            mask.scatter_(1, idx, 0.0)
            lg = lg + mask
        w = torch.softmax(lg, dim=1)
        out = w @ self.A
        return (out, w) if return_weights else out


def rollout_keep(policy, data: dict, mode: str, keep: torch.Tensor) -> dict:
    """``rev_core.rollout`` (same accounting and limits) with the output multiplied by
    keep[:, t]; in residual mode keep = 0 is the liability delta hedge on the spot."""
    x, hedges, tc = data["x"], data["hedges"], data["cost"]
    n, n_steps, _ = hedges.shape
    pos = torch.zeros(n, 2)
    pnl, cost = torch.zeros(n), torch.zeros(n)
    with torch.no_grad():
        for t in range(n_steps):
            state = torch.cat([x[:, t, :], pos], dim=1)
            out = policy(state) * keep[:, t:t + 1]
            if mode == "residual":
                target = torch.stack([data["liab_delta"][:, t], torch.zeros(n)], dim=1) + out
            elif mode == "trade":
                target = pos + out
            else:
                target = out
            target = torch.clamp(target, -rc.POS_BOUND, rc.POS_BOUND)
            trade = torch.clamp(target - pos, -rc.TRADE_BOUND, rc.TRADE_BOUND)
            pos = pos + trade
            pnl = pnl + (pos * hedges[:, t, :]).sum(dim=1)
            cost = cost + (trade.abs() * tc[:, t, :]).sum(dim=1)
    return {"offset": (data["payoff"] + pnl - cost).numpy().astype(np.float64), "cost": cost.numpy().astype(np.float64)}


def plain_rollout(policy, data, mode):
    with torch.no_grad():
        r = rc.rollout(policy, data, mode)
    return {"offset": r["offset"].numpy().astype(np.float64), "cost": r["cost"].numpy().astype(np.float64)}


# --------------------------------------------------------------------------
# Small statistics
# --------------------------------------------------------------------------

def ranks(a, axis=-1):
    return np.argsort(np.argsort(a, axis=axis), axis=axis).astype(np.float64)


def rowwise_corr(a, b):
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.where(den > 0, (a * b).sum(axis=-1) / np.maximum(den, 1e-300), np.nan)


def spearman(x, y):
    return float(rowwise_corr(ranks(np.asarray(x)), ranks(np.asarray(y))))


def partial_spearman(x, y, z):
    rx, ry, rz = ranks(np.asarray(x)), ranks(np.asarray(y)), ranks(np.asarray(z))
    Z = np.stack([np.ones_like(rz), rz], axis=1)
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(rowwise_corr(res(rx), res(ry)))


def nearest_distance(z, protos):
    """Distance from each row of z [m, d] to its nearest prototype [K, d]."""
    return np.sqrt(((z[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


# --------------------------------------------------------------------------
# One fit
# --------------------------------------------------------------------------

def analyse_layer(policy, bank, ep, data, scaler, mk, z_mkt, z_mkt_train, knn_dec, ivol_dec, liability,
                  ticker, seed, layer, saved_offset=None):
    """X1a-X1e and the X2 arrays for one ProtoHedge fit on one split."""
    res, arrays = {"layer": layer, "liability": liability, "ticker": ticker, "seed": seed}, {}
    premium = ep.np["premium"]
    ev = rc.evaluate_policy(policy, ep, STATE, MODE)
    W, O, P, off = ev["weights"].astype(np.float64), ev["outputs"].astype(np.float64), ev["positions"], ev["offset"]
    n, T, K = W.shape
    E = off + premium
    std_e = float(E.std())
    A = rc.prototype_outputs(policy)
    s = float(O.std())
    res.update({"K": int(K), "output_std": s, "he_std": std_e, "n_episodes": int(n)})
    if saved_offset is not None:
        res["check_saved_offset_maxdiff"] = float(np.max(np.abs(off - saved_offset)))

    # identity checks of the edited mixture and the masked rollout
    states_raw = np.concatenate([ep.exogenous(STATE), np.concatenate([np.zeros((n, 1, 2)), P[:, :-1]], axis=1)], axis=2)
    flat = torch.tensor(states_raw.reshape(n * T, -1), dtype=torch.float32)
    with torch.no_grad():
        out_e, w_e = Edited(policy)(flat, return_weights=True)
    res["check_edited_identity_maxdiff"] = float(np.max(np.abs(out_e.numpy() - O.reshape(n * T, 2))))
    res["check_rollout_keep_maxdiff"] = float(np.max(np.abs(
        rollout_keep(policy, data, MODE, torch.ones(n, T))["offset"] - off)))

    Wf, Of = W.reshape(n * T, K), O.reshape(n * T, 2)
    order = np.argsort(-Wf, axis=1)

    # X1a sufficiency
    def topk_rec(k):
        idx = order[:, :k]
        w = np.take_along_axis(Wf, idx, axis=1)
        w = w / w.sum(axis=1, keepdims=True)
        return np.einsum("mk,mkd->md", w, A[idx])
    for k in (1, 3, 5):
        res[f"x1a_top{k}_weight"] = float(np.take_along_axis(Wf, order[:, :k], axis=1).sum(axis=1).mean())
        res[f"x1a_top{k}_err_over_s"] = float(np.abs(topk_rec(k) - Of).mean() / s)
    r3 = plain_rollout(Edited(policy, topk=3), data, MODE)
    res["x1a_top3_rollout_he_std"] = float((r3["offset"] + premium).std())
    res["x1a_top3_rollout_rel_change"] = abs(res["x1a_top3_rollout_he_std"] - std_e) / std_e
    entropy = -(Wf * np.log(np.maximum(Wf, 1e-30))).sum(axis=1)
    res["effective_prototypes"] = float(np.exp(entropy).mean())

    # X1b similarity in visible (input) features
    z = (states_raw.reshape(n * T, -1) - scaler.mean) / scaler.std
    protos = bank["prototypes_scaled"].astype(np.float64)
    dist = np.sqrt(((z[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2))
    top1, nearest = Wf.argmax(axis=1), dist.argmin(axis=1)
    res["x1b_top1_is_nearest"] = float(np.mean(top1 == nearest))
    res["x1b_chance"] = 1.0 / K
    res["x1b_weight_distance_spearman"] = float(np.nanmean(rowwise_corr(ranks(Wf), ranks(-dist))))
    rank_of_top1 = ranks(dist)[np.arange(n * T), top1]
    res["x1b_top1_median_distance_rank"] = float(np.median(rank_of_top1) + 1)
    if layer == "ph":                              # untrained encoder, same prototypes and states
        share, rho = [], []
        for r in range(5):
            torch.manual_seed(7000 + r)
            fresh = rc.PHPolicy(scaler, bank["prototypes_scaled"])
            fresh.eval()
            with torch.no_grad():
                wf = fresh(flat, return_weights=True)[1].numpy().astype(np.float64)
            share.append(np.mean(wf.argmax(axis=1) == nearest))
            rho.append(np.nanmean(rowwise_corr(ranks(wf), ranks(-dist))))
        res["x1b_untrained_top1_is_nearest"] = float(np.mean(share))
        res["x1b_untrained_weight_distance_spearman"] = float(np.mean(rho))
    else:
        fw = policy.feature_weights.detach().numpy().astype(np.float64)
        for name, v in zip(scaler.names, fw / fw.sum()):
            res[f"feature_weight_share__{name}"] = float(v)

    # X1c self-consistency of attached actions
    with torch.no_grad():
        out_p, w_p = policy(torch.tensor(bank["prototypes_raw"], dtype=torch.float32), return_weights=True)
    out_p, w_p = out_p.numpy().astype(np.float64), w_p.numpy().astype(np.float64)
    res["x1c_self_err_over_s"] = float(np.abs(out_p - A).mean() / s)
    res["x1c_self_weight"] = float(np.mean(np.diag(w_p)))

    # X1d necessity, local
    def drop_change(idx):
        keep = np.ones_like(Wf)
        np.put_along_axis(keep, idx, 0.0, axis=1)
        lg = np.log(np.maximum(Wf, 1e-300))                  # renormalise in log space (stable)
        lg = np.where(keep > 0, lg, -np.inf)
        lg = lg - lg.max(axis=1, keepdims=True)
        w = np.exp(lg)
        w = w / w.sum(axis=1, keepdims=True)
        return float(np.abs(w @ A - Of).mean() / s)
    if K > 3:
        res["x1d_drop_heaviest3_change"] = drop_change(order[:, :3])
        res["x1d_drop_lightest3_change"] = drop_change(order[:, -3:])
    # X1d necessity, outcome level
    usage = Wf.mean(axis=0)
    deltas = np.zeros(K)
    for k in range(K):
        rk = plain_rollout(Edited(policy, drop=[k]), data, MODE)
        deltas[k] = float((rk["offset"] + premium).std()) - std_e
    res["x1d_usage_harm_spearman"] = spearman(usage, deltas) if np.ptp(deltas) > 0 else float("nan")
    res["x1d_most_used_hurts_more"] = bool(deltas[int(usage.argmax())] > deltas[int(usage.argmin())])
    res["x1d_most_used_delta_rel"] = float(deltas[int(usage.argmax())] / std_e)
    res["x1d_least_used_delta_rel"] = float(deltas[int(usage.argmin())] / std_e)
    arrays["x1d_usage"], arrays["x1d_delta"] = usage.astype(np.float32), deltas.astype(np.float32)

    # X1e stability
    rng = np.random.default_rng(rc.stable_seed("x1e", liability, ticker, seed, layer))
    noisy = states_raw.reshape(n * T, -1).copy()
    noisy[:, mk] = noisy[:, mk] + NOISE_STD * scaler.std[mk] * rng.standard_normal((n * T, len(mk)))
    with torch.no_grad():
        out_n, w_n = policy(torch.tensor(noisy, dtype=torch.float32), return_weights=True)
    w_n = w_n.numpy().astype(np.float64)
    res["x1e_top1_unchanged"] = float(np.mean(w_n.argmax(axis=1) == top1))
    res["x1e_weight_tv"] = float(0.5 * np.abs(w_n - Wf).sum(axis=1).mean())
    res["x1e_output_change_over_s"] = float(np.abs(out_n.numpy() - Of).mean() / s)

    # X2 familiarity: distance over market features to the nearest of the fit's prototypes
    p_mkt = protos[:, mk]
    fam = nearest_distance(z_mkt.reshape(n * T, -1), p_mkt).reshape(n, T)
    fam_train = nearest_distance(z_mkt_train.reshape(-1, len(mk)), p_mkt).reshape(z_mkt_train.shape[:2])
    tau95, tau99 = float(np.quantile(fam_train, 0.95)), float(np.quantile(fam_train, 0.99))
    res.update({"x2_tau95": tau95, "x2_tau99": tau99, "x2_split_mean_fam": float(fam.mean()),
                "x2_train_mean_fam": float(fam_train.mean())})
    arrays.update({"E": E.astype(np.float32), "fam_ep": fam.mean(axis=1).astype(np.float32),
                   "fam_train_ep": fam_train.mean(axis=1).astype(np.float32)})

    # X2b fallback rule and controls (same per-fit fallback count)
    keep95 = (fam <= tau95).astype(np.float32)
    keep99 = (fam <= tau99).astype(np.float32)
    m = int(round((1 - keep95.mean()) * n * T))
    res["x2b_rate95"], res["x2b_rate99"] = float(1 - keep95.mean()), float(1 - keep99.mean())

    def keep_top(score):                                  # fall back on the m highest scores
        kp = np.ones(n * T, dtype=np.float32)
        if m > 0:
            kp[np.argsort(-score.reshape(-1), kind="stable")[:m]] = 0.0
        return kp.reshape(n, T)
    variants = {"g95": keep95, "g99": keep99, "vol": keep_top(ivol_dec), "knn": keep_top(knn_dec)}
    for name, kp in variants.items():
        r = rollout_keep(policy, data, MODE, torch.tensor(kp))
        arrays[f"off_{name}"] = r["offset"].astype(np.float32)
        res[f"x2b_{name}_he_std"] = float((r["offset"] + premium).std())
        res[f"x2b_{name}_cost"] = float(r["cost"].mean())
    rand_off = []
    rng = np.random.default_rng(rc.stable_seed("x2b_rand", liability, ticker, seed, layer))
    for _ in range(N_RANDOM):
        kp = np.ones(n * T, dtype=np.float32)
        if m > 0:
            kp[rng.choice(n * T, size=m, replace=False)] = 0.0
        rand_off.append(rollout_keep(policy, data, MODE, torch.tensor(kp.reshape(n, T)))["offset"])
    arrays["off_rand"] = np.stack(rand_off).astype(np.float32)
    arrays["off"] = off.astype(np.float32)
    res["x2b_ungated_cost"] = float(ev["metrics"]["cost_mean"])
    return res, arrays, keep95


def process_unit(job):
    liability, ticker, split, allow_test = job
    torch.set_num_threads(1)
    t0 = time.time()
    train_ep = rc.Episodes(ticker, liability, "train")
    ep = rc.Episodes(ticker, liability, split, allow_test=(split == "test" and allow_test))
    scaler = rc.Scaler(train_ep, STATE)
    mk = [scaler.names.index(m) for m in market_names(liability)]
    z_mkt = (ep.exogenous(STATE)[..., mk] - scaler.mean[mk]) / scaler.std[mk]
    z_mkt_train = (train_ep.exogenous(STATE)[..., mk] - scaler.mean[mk]) / scaler.std[mk]
    med_step = float(np.median(np.abs(np.diff(z_mkt, axis=1))))      # X1e reference: typical one-step move
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=1).fit(z_mkt_train.reshape(-1, len(mk)))
    knn_dec = knn.kneighbors(z_mkt.reshape(-1, len(mk)))[0].reshape(ep.n, rc.N_STEPS)
    ivol_dec = ep.features["ivol"]
    data = ep.tensors(STATE)
    premium = ep.np["premium"]
    liab = np.zeros((ep.n, rc.N_STEPS, 2))
    liab[:, :, 0] = ep.np["liab_delta"]
    E_liab = p2.simulate_targets(ep.world.data.market, liab)["offset"] + premium
    E_dv = p2.simulate_targets(ep.world.data.market, rc.delta_vega_targets(ep))["offset"] + premium
    saved = None
    if split == "test":
        z = np.load(phase3_fit.ROOT / STUDY / "final" / "paths" / liability / f"{ticker}.npz")
        saved = {k: z[f"offset__{k}"].astype(np.float64) for k in ("ph", "phd", "dh")}
        saved["liability_delta"] = z["offset__liability_delta"][0].astype(np.float64)
    out_dir = out_root(split) / liability / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        a = {"ticker": ticker, "liability": liability, "mode": MODE, "objective": OBJECTIVE, "state": STATE, "seed": seed}
        cell = phase3_fit.cell_dir(STUDY, a)
        arrays = {"start_date": ep.start_date.astype(str), "train_start_date": train_ep.start_date.astype(str),
                  "premium": premium.astype(np.float32), "ivol_ep": ivol_dec.mean(axis=1).astype(np.float32),
                  "knn_ep": knn_dec.mean(axis=1).astype(np.float32),
                  "train_ivol_ep": train_ep.features["ivol"].mean(axis=1).astype(np.float32),
                  "E_liab": E_liab.astype(np.float32), "E_dv": E_dv.astype(np.float32)}
        dh, _ = rc.load_policy(cell, DH_FAMILY, scaler)
        ev_dh = rc.evaluate_policy(dh, ep, STATE, MODE)
        arrays["dh__off"] = ev_dh["offset"].astype(np.float32)
        dh_check = None if saved is None else float(np.max(np.abs(ev_dh["offset"] - saved["dh"][seed])))
        for layer, fam in LAYERS.items():
            policy, _ = rc.load_policy(cell, fam, scaler)
            bank = np.load(cell / fam / "prototypes.npz", allow_pickle=False)
            res, arr, keep95 = analyse_layer(policy, bank, ep, data, scaler, mk, z_mkt, z_mkt_train, knn_dec,
                                             ivol_dec, liability, ticker, seed, layer,
                                             None if saved is None else saved[layer][seed])
            if saved is None:                                  # validation: compare with the saved training-time outcomes
                vp = np.load(cell / fam / "val_paths.npz")
                res["check_saved_offset_maxdiff"] = float(np.max(np.abs(arr["off"] - vp["offset"])))
            res["check_dh_saved_offset_maxdiff"] = dh_check
            res["x1e_reference_median_step_change"] = med_step
            if layer == "ph":                                  # secondary: the same rule applied to Deep Hedging
                r = rollout_keep(dh, data, MODE, torch.tensor(keep95))
                arrays["dh__off_g95"] = r["offset"].astype(np.float32)
                res["x2b_dh_g95_he_std"] = float((r["offset"] + premium).std())
                res["x2b_dh_he_std"] = float((ev_dh["offset"] + premium).std())
            arrays.update({f"{layer}__{k}": v for k, v in arr.items()})
            rows.append(res)
        np.savez_compressed(out_dir / f"seed_{seed}.npz", **arrays)
    rc.write_json(out_dir / "fits.json", rows)
    if saved is not None:
        liab_check = float(np.max(np.abs(E_liab - premium - saved["liability_delta"])))
        for r in rows:
            r["check_liability_delta_maxdiff"] = liab_check
        rc.write_json(out_dir / "fits.json", rows)
    return f"{time.strftime('%T')} {split} {liability} {ticker}: {len(rows)} fits in {time.time() - t0:.0f}s"


def cmd_run(args):
    check_test_allowed(args.split, args.allow_test)
    root = out_root(args.split)
    root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).read_bytes()
    rc.write_json(root / "RUN.json", {"started": time.strftime("%F %T"), "split": args.split,
                                      "script_sha256": hashlib.sha256(code).hexdigest()[:16],
                                      "protocol": "PHASE3_PROTOCOL.md: Extensions X1 and X2 (2026-11-18)"})
    jobs = [(l, t, args.split, args.allow_test) for l in LIABILITIES for t in p2.TICKERS]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(process_unit, jobs):
            print(msg, flush=True)
    rc.write_json(root / "DONE.json", {"finished": time.strftime("%F %T")})


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def load_split(split):
    root = out_root(split)
    rows, per = [], {}
    for liability in LIABILITIES:
        for ticker in p2.TICKERS:
            d = root / liability / ticker
            rows += json.load(open(d / "fits.json"))
            per[(liability, ticker)] = [dict(np.load(d / f"seed_{s}.npz")) for s in SEEDS]
    return pd.DataFrame(rows), per


def panel(df, col):
    """Equal-ticker mean of seed means, and the range across tickers."""
    t = df.groupby("ticker")[col].mean()
    return float(t.mean()), float(t.min()), float(t.max())


def calendar_stat_bootstrap(per_ticker, dates, fn, block=20, draws=2000, seed=0):
    """Point and 95% interval of mean_ticker fn(arrays, idx) under calendar-aligned block resampling."""
    rng = np.random.default_rng(seed)
    cal = np.array(sorted(set(np.concatenate([np.asarray(d) for d in dates]))))
    lookup = []
    for d in dates:
        pos = {v: i for i, v in enumerate(np.asarray(d))}
        lookup.append(np.array([pos.get(v, -1) for v in cal]))
    point = float(np.mean([fn(a, np.arange(len(d))) for a, d in zip(per_ticker, dates)]))
    out = np.zeros(draws)
    for b in range(draws):
        ci = inf.block_indices(len(cal), block, rng)
        out[b] = np.mean([fn(a, lk[ci][lk[ci] >= 0]) for a, lk in zip(per_ticker, lookup)])
    lo, hi = np.quantile(out, [0.025, 0.975])
    return point, float(lo), float(hi)


def cmd_report(args):
    df, per = load_split(args.split)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.split
    df.to_csv(SUMMARY_DIR / f"x_by_fit_{tag}.csv", index=False)
    checks = {c: float(np.nanmax(df[c].astype(float))) for c in df.columns if c.startswith("check_")}
    print("largest check differences:", {k: f"{v:.1e}" for k, v in checks.items()})

    # ---- X1 and X2b scalar summaries
    cols = ["he_std", "output_std", "effective_prototypes", "x1a_top3_weight", "x1a_top3_err_over_s",
            "x1a_top3_rollout_rel_change", "x1b_top1_is_nearest", "x1b_chance", "x1b_weight_distance_spearman",
            "x1b_top1_median_distance_rank", "x1b_untrained_top1_is_nearest", "x1b_untrained_weight_distance_spearman",
            "x1c_self_err_over_s", "x1c_self_weight", "x1d_drop_heaviest3_change", "x1d_drop_lightest3_change",
            "x1d_usage_harm_spearman", "x1d_most_used_hurts_more", "x1d_most_used_delta_rel",
            "x1d_least_used_delta_rel", "x1e_top1_unchanged", "x1e_weight_tv", "x1e_output_change_over_s",
            "x1e_reference_median_step_change",
            "x2b_rate95", "x2b_rate99"]
    summary = []
    for liability in LIABILITIES:
        for layer in LAYERS:
            sub = df[(df.liability == liability) & (df.layer == layer)].copy()
            sub["x1d_most_used_hurts_more"] = sub["x1d_most_used_hurts_more"].astype(float)
            for c in cols:
                if c in sub and sub[c].notna().any():
                    mean, lo, hi = panel(sub, c)
                    summary.append({"liability": liability, "layer": layer, "measure": c, "panel_mean": mean,
                                    "ticker_min": lo, "ticker_max": hi})
    summ = pd.DataFrame(summary)
    summ.to_csv(SUMMARY_DIR / f"x1_summary_{tag}.csv", index=False)

    # ---- reference: median one-step change of standardised market features is computed here from val/test states
    # ---- X2a correlations with calendar-block intervals
    corr_rows = []
    for liability in LIABILITIES:
        dates = [per[(liability, t)][0]["start_date"] for t in p2.TICKERS]
        for layer in LAYERS:
            def mk_fn(signal, err):
                def fn(seeds, idx):
                    vals = []
                    for z in seeds:
                        e = z[err] if err != "own" else z[f"{layer}__E"]
                        size = np.abs(e - e.mean())[idx]
                        sig = (z[f"{layer}__fam_ep"] if signal == "fam" else z[signal])[idx]
                        vals.append(spearman(sig, size))
                    return float(np.mean(vals))
                return fn

            def partial_fn(seeds, idx):
                vals = []
                for z in seeds:
                    e = z[f"{layer}__E"]
                    vals.append(partial_spearman(z[f"{layer}__fam_ep"][idx], np.abs(e - e.mean())[idx], z["ivol_ep"][idx]))
                return float(np.mean(vals))

            def dh_fn(seeds, idx):
                vals = []
                for z in seeds:
                    e = z["dh__off"].astype(np.float64) + z["premium"]
                    vals.append(spearman(z[f"{layer}__fam_ep"][idx], np.abs(e - e.mean())[idx]))
                return float(np.mean(vals))

            tests = {"fam_vs_own_error": mk_fn("fam", "own"), "fam_vs_own_error_partial_ivol": partial_fn,
                     "fam_vs_dh_error": dh_fn, "fam_vs_liability_delta_error": mk_fn("fam", "E_liab"),
                     "ivol_vs_own_error": mk_fn("ivol_ep", "own"), "fulllibrary_vs_own_error": mk_fn("knn_ep", "own")}
            for i, (name, fn) in enumerate(tests.items()):
                point, lo, hi = calendar_stat_bootstrap([per[(liability, t)] for t in p2.TICKERS], dates, fn,
                                                        block=20, draws=args.draws, seed=20261118 + i)
                corr_rows.append({"liability": liability, "layer": layer, "statistic": name, "estimate": point,
                                  "ci95_low": lo, "ci95_high": hi})
                print(f"{time.strftime('%T')} {liability} {layer} {name}: {point:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(SUMMARY_DIR / f"x2a_correlations_{tag}.csv", index=False)

    # ---- X2b contrasts (paired, calendar-aligned blocks)
    con_rows, score_rows = [], []
    for liability in LIABILITIES:
        outcomes, premiums, dates = {}, [], []
        for t in p2.TICKERS:
            seeds = per[(liability, t)]
            premiums.append(seeds[0]["premium"].astype(np.float64))
            dates.append(seeds[0]["start_date"])
            add = lambda key, arr: outcomes.setdefault(key, []).append(arr)
            for layer in LAYERS:
                add(layer, np.stack([z[f"{layer}__off"] for z in seeds]).astype(np.float64))
                for v in ("g95", "g99", "vol", "knn"):
                    add(f"{layer}_{v}", np.stack([z[f"{layer}__off_{v}"] for z in seeds]).astype(np.float64))
                add(f"{layer}_rand", np.concatenate([z[f"{layer}__off_rand"] for z in seeds]).astype(np.float64))
            add("dh", np.stack([z["dh__off"] for z in seeds]).astype(np.float64))
            add("dh_g95", np.stack([z["dh__off_g95"] for z in seeds]).astype(np.float64))
            add("liability_delta", (seeds[0]["E_liab"].astype(np.float64) - premiums[-1])[None, :])
        boot = inf.panel_bootstrap(outcomes, premiums, dates, block=20, draws=args.draws, seed=20261118, calendar=True)
        for m, vals in boot["point"].items():
            score_rows.append({"liability": liability, "arm": m, **vals})
        pairs = [(f"{L}_{v}", ref) for L in LAYERS for v, ref in
                 (("g95", L), ("g95", f"{L}_rand"), ("g95", f"{L}_vol"), ("g95", f"{L}_knn"), ("g99", L))]
        pairs += [("dh_g95", "dh"), ("ph", "dh"), ("ph_g95", "dh"), ("phd_g95", "dh")]
        for cand, ref in pairs:
            for metric in ("he_std", "he_cvar05", "offset_cvar50"):
                con_rows.append({"liability": liability, **inf.contrast(boot, cand, ref, metric)})
        print(f"{time.strftime('%T')} contrasts {liability} done", flush=True)
    pd.DataFrame(score_rows).to_csv(SUMMARY_DIR / f"x2b_scoreboard_{tag}.csv", index=False)
    con = pd.DataFrame(con_rows)
    con.to_csv(SUMMARY_DIR / f"x2b_contrasts_{tag}.csv", index=False)

    # ---- pre-declared criteria
    def get(liability, layer, measure):
        r = summ[(summ.liability == liability) & (summ.layer == layer) & (summ.measure == measure)]
        return float(r.panel_mean.iloc[0]) if len(r) else float("nan")
    verdict = {}
    for liability in LIABILITIES:
        for layer in LAYERS:
            g = lambda m: get(liability, layer, m)
            c = corr[(corr.liability == liability) & (corr.layer == layer)].set_index("statistic")
            x2b = con[(con.liability == liability) & (con.candidate == f"{layer}_g95") & (con.benchmark == layer)
                      & (con.metric == "he_std")].iloc[0]
            verdict[f"{liability}/{layer}"] = {
                "X1a": bool(g("x1a_top3_err_over_s") <= 0.10 and g("x1a_top3_rollout_rel_change") <= 0.05),
                "X1b": bool(g("x1b_top1_is_nearest") >= 0.5 and g("x1b_weight_distance_spearman") >= 0.5),
                "X1c": bool(g("x1c_self_err_over_s") <= 0.10),
                "X1d": bool(g("x1d_drop_heaviest3_change") >= 3 * g("x1d_drop_lightest3_change")
                            and g("x1d_drop_lightest3_change") <= 0.05 and g("x1d_usage_harm_spearman") >= 0.5
                            and g("x1d_most_used_hurts_more") >= 0.8),
                "X1e": bool(g("x1e_top1_unchanged") >= 0.9),
                "X2a_flags_risky_episodes": bool(c.loc["fam_vs_own_error", "ci95_low"] > 0),
                "X2a_adds_to_implied_vol": bool(c.loc["fam_vs_own_error_partial_ivol", "ci95_low"] > 0),
                "X2b_fallback_helps": bool(x2b["ci95_high"] < 0),
            }
    rc.write_json(SUMMARY_DIR / f"x_verdict_{tag}.json", {"split": tag, "checks": checks, "verdict": verdict,
                                                          "written": time.strftime("%F %T")})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(summ.pivot_table(index="measure", columns=["liability", "layer"], values="panel_mean").round(3).to_string())
    print(corr.round(3).to_string(index=False))
    print(con.round(5).to_string(index=False))
    print(json.dumps(verdict, indent=1))


# --------------------------------------------------------------------------
# X2c coverage over time, and X2d the original settings
# --------------------------------------------------------------------------

def cmd_timeline(args):
    """Monthly mean familiarity distance 2005-2024 (training, validation, test)."""
    rows = []
    for split in ("val", "test"):
        _, per = load_split(split)
        for (liability, ticker), seeds in per.items():
            for s, z in enumerate(seeds):
                for layer in LAYERS:
                    for period, d, f, v in (("train", z["train_start_date"], z[f"{layer}__fam_train_ep"], z["train_ivol_ep"]),
                                            (split, z["start_date"], z[f"{layer}__fam_ep"], z["ivol_ep"])):
                        if period == "train" and split == "test":
                            continue                      # training arrays are identical in both runs
                        rows.append(pd.DataFrame({"liability": liability, "ticker": ticker, "seed": s, "layer": layer,
                                                  "period": period, "month": pd.to_datetime(d).to_period("M").astype(str),
                                                  "fam": f, "ivol": v}))
    df = pd.concat(rows)
    monthly = (df.groupby(["liability", "layer", "period", "month", "ticker"])[["fam", "ivol"]].mean()
               .groupby(level=[0, 1, 2, 3]).agg(fam=("fam", "mean"), ivol=("ivol", "median")).reset_index())
    monthly.to_csv(SUMMARY_DIR / "x2c_monthly.csv", index=False)
    print(monthly.groupby(["liability", "layer", "period"])[["fam", "ivol"]].mean().round(3).to_string())


def cmd_orig(args):
    """X2d: what the prototype table of the original settings (arm F) shows."""
    check_test_allowed(args.split, args.allow_test)
    rows = []
    for liability in LIABILITIES:
        for ticker in p2.TICKERS:
            train_ep = rc.Episodes(ticker, liability, "train")
            ep = rc.Episodes(ticker, liability, args.split, allow_test=(args.split == "test" and args.allow_test))
            scaler = rc.Scaler(train_ep, "s0")
            unhedged = float((p2.simulate_targets(ep.world.data.market, p2.constant_targets(ep.n, 0, 0))["offset"]
                              + ep.np["premium"]).std())
            for seed in (0, 1, 2):
                a = {"ticker": ticker, "liability": liability, "mode": "trade", "objective": "cvar50_offset",
                     "state": "s0", "seed": seed}
                cell = phase3_fit.cell_dir(STUDY, a)
                policy, _ = rc.load_policy(cell, "ph", scaler)
                ev = rc.evaluate_policy(policy, ep, "s0", "trade")
                W, O, P = ev["weights"], ev["outputs"], ev["positions"]
                A = rc.prototype_outputs(policy)
                usage = W.reshape(-1, W.shape[-1]).mean(axis=0)
                prev = np.concatenate([np.zeros_like(P[:, :1]), P[:, :-1]], axis=1)
                intended = prev + O
                he_std = float((ev["offset"] + ep.np["premium"]).std())
                rows.append({"liability": liability, "ticker": ticker, "seed": seed, "K": len(A),
                             "attached_spot_buy_share": float(np.mean(A[:, 0] > 0)),
                             "attached_spot_mean": float(A[:, 0].mean()),
                             "usage_weighted_spot_buy_share": float(usage @ (A[:, 0] > 0)),
                             "usage_weighted_spot_mean": float(usage @ A[:, 0]),
                             "limits_change_share": float(np.mean(np.abs(intended - P).max(axis=2) > 1e-4)),
                             "spot_at_upper": float(np.mean(P[:, :, 0] >= 1 - 1e-6)),
                             "he_std_over_unhedged": he_std / unhedged, "failed": bool(he_std > unhedged)})
            print(f"{time.strftime('%T')} orig {args.split} {liability} {ticker}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_DIR / f"x2d_orig_{args.split}.csv", index=False)
    pd.set_option("display.width", 250)
    for liability in LIABILITIES:
        sub = df[df.liability == liability]
        print(liability, sub.drop(columns=["ticker", "seed", "liability"]).mean(numeric_only=True).round(3).to_dict())
        print("  Spearman(limits_change_share, he_std/unhedged) across fits:",
              round(spearman(sub.limits_change_share, sub.he_std_over_unhedged), 3),
              "| failed fits:", int(sub.failed.sum()), "of", len(sub))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", required=True, choices=["val", "test"])
    r.add_argument("--allow-test", dest="allow_test", action="store_true")
    r.add_argument("--workers", type=int, default=8)
    r.set_defaults(func=cmd_run)
    q = sub.add_parser("report")
    q.add_argument("--split", required=True, choices=["val", "test"])
    q.add_argument("--draws", type=int, default=2000)
    q.set_defaults(func=cmd_report)
    t = sub.add_parser("timeline")
    t.set_defaults(func=cmd_timeline)
    o = sub.add_parser("orig")
    o.add_argument("--split", required=True, choices=["val", "test"])
    o.add_argument("--allow-test", dest="allow_test", action="store_true")
    o.set_defaults(func=cmd_orig)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

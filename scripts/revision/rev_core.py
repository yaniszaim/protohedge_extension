#!/usr/bin/env python3
"""Revision core for Phase 3: episodes, policies, hedging loop, objectives, trainer.

Nothing in the original modules is changed. The historical world, costs,
position limits and inventory-step accounting are the v4 ones
(``RealWorld_Spot_ATM_Torch`` with the v4 world settings); the ProtoHedge layer
is the repository's ``ProtoHedgeAgent`` (encoder, softmax of negative squared
distance, bounded prototype outputs). What is new, and identical for Deep
Hedging and ProtoHedge:

* inputs standardised with training-episode statistics;
* three output parameterisations ("trade", "target", "residual", see ``MODES``);
* objectives that can include the liability premium and other tail levels;
* checkpoint selection on the validation value of the hedged objective only;
* full training curves and prototype source indices are saved.
"""

from __future__ import annotations

import os

# One thread per process. Besides making parallel jobs predictable, this avoids a
# crash on macOS when scikit-learn's k-means and PyTorch each start an OpenMP pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase2_diagnostics as p2  # noqa: E402  (liability deltas, exact CVaR, accounting)

PANEL = REPO / "Data" / "NEW_PANEL_DECISION_V2"
N_STEPS = 20
POS_BOUND = 1.0
TRADE_BOUND = 1.0

# What the policy output means. In every mode the position stays in [-1, 1].
#   trade    : output is the trade (original semantics), |trade| <= 1
#   target   : output is the position to hold after trading
#   residual : output is a correction added to the liability delta hedge
MODES = ("trade", "target", "residual")
# Exogenous state sets (inventory of both instruments is always appended).
STATES = {
    "s0": ["spot", "call", "time_left"],                       # as submitted
    "s1": ["spot", "call", "time_left", "ivol", "listed_log_moneyness", "listed_ttm"],
}
OBJECTIVES = {
    # name: (kind, tail level, premium included)
    "cvar50_offset": ("cvar", 0.50, False),   # the original paper's objective
    "cvar10_he": ("cvar", 0.10, True),
    "cvar25_he": ("cvar", 0.25, True),
    "var_he": ("var", None, True),            # variance of the hedging error
}


def stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class Episodes:
    """One ticker x liability x split with everything a policy or rule may observe."""

    def __init__(self, ticker: str, liability: str, split: str, allow_test: bool = False):
        if split == "test" and not allow_test:
            raise RuntimeError("test episodes are locked; pass allow_test=True only in the final evaluation")
        from deephedging.real_data_sweep_torch import world_cfg
        from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch

        self.ticker, self.liability, self.split = ticker, liability, split
        data_path = PANEL / "episodes" / f"{ticker}_training_paths.npy"
        splits = np.load(PANEL / "episodes" / f"{ticker}_chronological_splits.npz")
        self.index = np.asarray(splits[split], dtype=np.int64)
        world = RealWorld_Spot_ATM_Torch(world_cfg(
            data_path=data_path, indices=self.index, seed=1234, normalize=True, hedge_mode="step",
            position_bounds=True, trade_bounds=(-TRADE_BOUND, TRADE_BOUND),
            cumulative_bounds=(-POS_BOUND, POS_BOUND), liability_type=liability))
        self.world = world
        self.n = int(world.nSamples)
        market, per_step = world.data.market, world.data.features.per_step
        meta = pd.read_csv(PANEL / "episode_metadata" / f"{ticker}_episode_metadata.csv").set_index("episode_index")
        meta = meta.loc[self.index]
        self.start_date = meta["start_date"].to_numpy()
        self.end_date = meta["end_date"].to_numpy()

        premium, liab_delta = p2.liability_premium_and_delta(world)
        spot = np.asarray(per_step["spot"], dtype=np.float64)
        steps = np.arange(N_STEPS, dtype=np.float64)
        strike_ratio = (meta["option_strike"].to_numpy(dtype=np.float64)
                        / meta["liability_strike"].to_numpy(dtype=np.float64))
        dte0 = meta["initial_dte_days"].to_numpy(dtype=np.float64)
        feats = {
            "spot": spot,
            "call": np.asarray(per_step["price"], dtype=np.float64)[:, :, 1],
            "time_left": np.asarray(per_step["time_left"], dtype=np.float64),
            "ivol": np.asarray(per_step["ivol"], dtype=np.float64),
            "listed_log_moneyness": np.log(spot / strike_ratio[:, None]),
            # calendar days to the listed call's expiry, approximated from the
            # initial value and the trading-day count (known at decision time)
            "listed_ttm": np.maximum(dte0[:, None] - steps[None, :] * 365.0 / 252.0, 1.0) / 365.0,
        }
        if liability == "asian_call":
            feats["asian_moneyness"] = np.asarray(per_step["asian_moneyness"], dtype=np.float64)
        self.features = feats
        self.np = {
            "hedges": np.asarray(market["hedges"], dtype=np.float64),
            "cost": np.asarray(market["cost"], dtype=np.float64),
            "payoff": np.asarray(market["payoff"], dtype=np.float64).reshape(-1),
            "premium": premium, "liab_delta": liab_delta,
            "listed_delta": np.asarray(per_step["call_delta"], dtype=np.float64),
        }

    def state_names(self, state: str):
        names = list(STATES[state])
        if self.liability == "asian_call":
            names.append("asian_moneyness")
        return names

    def exogenous(self, state: str) -> np.ndarray:
        return np.stack([self.features[k] for k in self.state_names(state)], axis=2)   # [n, T, d]

    def tensors(self, state: str) -> dict:
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        return {"x": t(self.exogenous(state)), "hedges": t(self.np["hedges"]), "cost": t(self.np["cost"]),
                "payoff": t(self.np["payoff"]), "premium": t(self.np["premium"]),
                "liab_delta": t(self.np["liab_delta"])}


class Scaler:
    """Training-episode mean and std of the exogenous state; inventory is scaled by a fixed 0.5."""

    def __init__(self, train: Episodes, state: str):
        x = train.exogenous(state).reshape(-1, len(train.state_names(state)))
        self.mean = np.concatenate([x.mean(axis=0), np.zeros(2)])
        self.std = np.concatenate([np.maximum(x.std(axis=0), 1e-8), np.full(2, 0.5)])
        self.names = train.state_names(state) + ["inventory_spot", "inventory_call"]


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

class DHPolicy(nn.Module):
    """Deep Hedging network: width 20, depth 3, Softplus (the paper's size), bounded by tanh."""

    family = "dh"

    def __init__(self, scaler: Scaler, width: int = 20, depth: int = 3, init_scale: float = 0.1):
        super().__init__()
        self.register_buffer("mean", torch.tensor(scaler.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(scaler.std, dtype=torch.float32))
        layers, d = [], len(scaler.mean)
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Softplus()]
            d = width
        last = nn.Linear(d, 2)
        with torch.no_grad():                      # small initial outputs: start near a flat book
            last.weight.mul_(init_scale)
            last.bias.zero_()
        self.network = nn.Sequential(*layers, last)

    def forward(self, x, return_weights=False):
        out = torch.tanh(self.network((x - self.mean) / self.std))
        return (out, None) if return_weights else out


class PHPolicy(nn.Module):
    """ProtoHedge: the repository's ProtoHedgeAgent with outputs bounded to [-1, 1]."""

    family = "ph"

    def __init__(self, scaler: Scaler, prototypes_scaled: np.ndarray):
        super().__init__()
        from deephedging.agents_torch import ProtoHedgeAgent
        self.agent = ProtoHedgeAgent(
            input_dim=len(scaler.mean), num_prototypes=len(prototypes_scaled), output_dim=2,
            prototype_init=torch.tensor(prototypes_scaled, dtype=torch.float32),
            feature_mean=torch.tensor(scaler.mean, dtype=torch.float32),
            feature_std=torch.tensor(scaler.std, dtype=torch.float32),
            action_low=torch.full((2,), -1.0), action_high=torch.full((2,), 1.0),
            softclip_mode="tfp_exact")

    def forward(self, x, return_weights=False):
        return self.agent(x, return_weights=return_weights)


class PlainPHPolicy(nn.Module):
    """Plain-distance ProtoHedge: similarity in the standardised input space,
    weights = softmax(-sum_j w_j (x_j - p_j)^2) with trainable positive w (the authors'
    earlier layer; no encoder), outputs bounded to [-1, 1]."""

    family = "phd"

    def __init__(self, scaler: Scaler, prototypes_scaled: np.ndarray, learn_weights: bool = True,
                 init_weight: float = 1.0):
        super().__init__()
        from deephedging.softclip_torch import tfp_softclip
        self._softclip = tfp_softclip
        d = len(scaler.mean)
        self.register_buffer("mean", torch.tensor(scaler.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(scaler.std, dtype=torch.float32))
        self.register_buffer("prototypes", torch.tensor(prototypes_scaled, dtype=torch.float32))
        self.register_buffer("low", torch.full((2,), -1.0))
        self.register_buffer("high", torch.full((2,), 1.0))
        start = torch.full((d,), float(np.log(init_weight)))     # larger = sharper similarity
        if learn_weights:
            self.log_feature_weights = nn.Parameter(start)
        else:
            self.register_buffer("log_feature_weights", start)
        init = torch.empty(len(prototypes_scaled), 2)
        nn.init.normal_(init, mean=0.0, std=0.05)
        self.outputs_unbounded = nn.Parameter(init)

    @property
    def feature_weights(self):
        return torch.exp(self.log_feature_weights)

    def prototype_outputs(self):
        return self._softclip(self.outputs_unbounded, self.low, self.high)

    def similarity(self, x):
        z = (x - self.mean) / self.std
        d2 = (((z[:, None, :] - self.prototypes[None, :, :]) ** 2) * self.feature_weights).sum(dim=-1)
        return torch.softmax(-d2, dim=-1)

    def forward(self, x, return_weights=False):
        w = self.similarity(x)
        out = w @ self.prototype_outputs()
        return (out, w) if return_weights else out


def parse_family(spec: str) -> dict:
    """'dh', 'dh:w=64', 'ph', 'ph:k=25', 'phd' -> kind, options and a folder name."""
    kind, _, rest = spec.partition(":")
    opts = dict(item.split("=") for item in rest.split(",") if item)
    if kind not in ("dh", "ph", "phd") or not set(opts) <= {"w", "k"}:
        raise ValueError(f"unknown family spec {spec!r}")
    name = kind + "".join(f"_{k}{v}" for k, v in sorted(opts.items()))
    return {"kind": kind, "width": int(opts.get("w", 20)), "k": int(opts["k"]) if "k" in opts else None,
            "name": name}


def load_policy(cell_dir: Path, family: str, scaler: Scaler):
    """Rebuild a trained policy from ``<cell>/<family>/`` (weights only; no pickles)."""
    d = Path(cell_dir) / family
    meta = json.load(open(d / "complete.json"))
    kind = meta.get("kind") or ("dh" if family.startswith("dh") else "phd" if family.startswith("phd") else "ph")
    if kind == "dh":
        policy = DHPolicy(scaler, width=int(meta.get("width") or 20))
    else:
        bank = np.load(d / "prototypes.npz", allow_pickle=False)
        policy = (PHPolicy if kind == "ph" else PlainPHPolicy)(scaler, bank["prototypes_scaled"])
    policy.load_state_dict(torch.load(d / "policy.pt", weights_only=True))
    policy.eval()
    return policy, meta


def prototype_outputs(policy) -> np.ndarray:
    """Bounded output attached to each prototype, [K, 2]."""
    with torch.no_grad():
        if isinstance(policy, PHPolicy):
            return policy.agent.prototype_actions.numpy().astype(np.float64)
        return policy.prototype_outputs().numpy().astype(np.float64)


def clamp_ste(x, low, high):
    """Clamp in the forward pass, identity gradient (so a bound never switches learning off)."""
    return x + (torch.clamp(x, low, high) - x).detach()


def rollout(policy, data: dict, mode: str, return_paths: bool = False) -> dict:
    """Run a policy through the episodes. Accounting equals gym_torch (inventory-step)."""
    x, hedges, tc = data["x"], data["hedges"], data["cost"]
    n, n_steps, _ = hedges.shape
    pos = torch.zeros(n, 2)
    pnl, cost = torch.zeros(n), torch.zeros(n)
    positions, weights, raw = [], [], []
    for t in range(n_steps):
        state = torch.cat([x[:, t, :], pos], dim=1)
        if return_paths:
            out, w = policy(state, return_weights=True)
            if w is not None:
                weights.append(w[:, None, :])
            raw.append(out[:, None, :])
        else:
            out = policy(state)
        if mode == "trade":
            target = pos + out
        elif mode == "target":
            target = out
        elif mode == "residual":
            base = torch.stack([data["liab_delta"][:, t], torch.zeros(n)], dim=1)
            target = base + out
        else:
            raise ValueError(mode)
        target = clamp_ste(target, -POS_BOUND, POS_BOUND)
        trade = clamp_ste(target - pos, -TRADE_BOUND, TRADE_BOUND)
        pos = pos + trade
        pnl = pnl + (pos * hedges[:, t, :]).sum(dim=1)
        cost = cost + (trade.abs() * tc[:, t, :]).sum(dim=1)
        if return_paths:
            positions.append(pos[:, None, :])
    result = {"offset": data["payoff"] + pnl - cost, "cost": cost}
    if return_paths:
        result["positions"] = torch.cat(positions, dim=1)
        result["outputs"] = torch.cat(raw, dim=1)
        if weights:
            result["weights"] = torch.cat(weights, dim=1)
    return result


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------

class Objective(nn.Module):
    """Loss to minimise on the outcome x (offset, or offset + premium)."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.kind, self.alpha, self.with_premium = OBJECTIVES[name]
        self.aux = nn.Parameter(torch.zeros(()))     # OCE threshold y, or the mean for "var"

    def outcome(self, offset, premium):
        return offset + premium if self.with_premium else offset

    def forward(self, offset, premium):
        x = self.outcome(offset, premium)
        if self.kind == "cvar":                      # -E[min(x + y, 0) / alpha - y]
            return -(torch.clamp(x + self.aux, max=0.0) / self.alpha - self.aux).mean()
        return ((x - self.aux) ** 2).mean() * 100.0  # variance, scaled for readability

    def exact_value(self, offset: np.ndarray, premium: np.ndarray) -> float:
        """Threshold-free value, higher is better (used for validation and reporting)."""
        x = offset + premium if self.with_premium else offset
        if self.kind == "cvar":
            return float(p2.lower_cvar(x, self.alpha))
        return float(-np.std(x))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_policy(policy, objective: Objective, train: dict, val: dict, mode: str, seed: int,
                 max_epochs: int = 800, patience: int = 200, batch_size: int = 32, lr: float = 1e-3,
                 log=None) -> dict:
    """Adam, batch 32, gradient clipping as in v4. The checkpoint is the epoch with the
    best validation value of the hedged objective; initialisation is epoch -1."""
    torch.manual_seed(int(seed))
    params = list(policy.parameters()) + list(objective.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    n = train["payoff"].shape[0]

    def evaluate(data):
        with torch.no_grad():
            off = rollout(policy, data, mode)["offset"].numpy().astype(np.float64)
        return objective.exact_value(off, data["premium"].numpy().astype(np.float64))

    history = {"train": [evaluate(train)], "val": [evaluate(val)]}
    best = {"epoch": -1, "val": history["val"][0],
            "state": {k: v.detach().clone() for k, v in policy.state_dict().items()}}
    t0 = time.time()
    for epoch in range(max_epochs):
        policy.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = {k: v[idx] for k, v in train.items()}
            optimizer.zero_grad()
            loss = objective(rollout(policy, batch, mode)["offset"], batch["premium"])
            loss.backward()
            torch.nn.utils.clip_grad_value_(params, 1.0)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        policy.eval()
        history["train"].append(evaluate(train))
        history["val"].append(evaluate(val))
        if history["val"][-1] > best["val"]:
            best = {"epoch": epoch, "val": history["val"][-1],
                    "state": {k: v.detach().clone() for k, v in policy.state_dict().items()}}
        if log and epoch % 25 == 0:
            log(f"epoch {epoch:4d} train {history['train'][-1]:+.5f} val {history['val'][-1]:+.5f} "
                f"best {best['epoch']} ({time.time() - t0:.0f}s)")
        if epoch - best["epoch"] >= patience:
            break
    policy.load_state_dict(best["state"])
    policy.eval()
    return {"history": history, "best_epoch": int(best["epoch"]), "best_val": float(best["val"]),
            "epochs_run": len(history["train"]) - 1, "seconds": time.time() - t0}


# --------------------------------------------------------------------------
# Prototype extraction (original recipe, plus source indices)
# --------------------------------------------------------------------------

def extract_prototypes(source_policy, train: dict, mode: str, scaler: Scaler, k: int, seed: int) -> dict:
    """k-means on the standardised states visited by the source policy on the training
    episodes; each prototype is the real visited state nearest its cluster centre."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from threadpoolctl import threadpool_limits

    with torch.no_grad():
        positions = rollout(source_policy, train, mode, return_paths=True)["positions"].numpy()
    n, n_steps, _ = positions.shape
    prev = np.concatenate([np.zeros((n, 1, 2)), positions[:, :-1, :]], axis=1)   # inventory seen at step t
    states = np.concatenate([train["x"].numpy(), prev], axis=2).reshape(n * n_steps, -1)
    scaled = (states - scaler.mean) / scaler.std
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=int(k), random_state=int(seed), n_init="auto").fit(scaled)
    rows = []
    for c in range(int(k)):
        members = np.flatnonzero(km.labels_ == c)
        if len(members) == 0:
            continue
        j, _ = pairwise_distances_argmin_min(km.cluster_centers_[c][None, :], scaled[members])
        rows.append(int(members[j[0]]))
    rows = np.array(rows, dtype=np.int64)
    return {"prototypes_scaled": scaled[rows].astype(np.float32), "prototypes_raw": states[rows].astype(np.float32),
            "source_path": rows // n_steps, "source_step": rows % n_steps,
            "cluster_size": np.bincount(km.labels_, minlength=int(k))[km.labels_[rows]]}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_policy(policy, episodes: Episodes, state: str, mode: str) -> dict:
    data = episodes.tensors(state)
    with torch.no_grad():
        r = rollout(policy, data, mode, return_paths=True)
    offset = r["offset"].numpy().astype(np.float64)
    positions = r["positions"].numpy().astype(np.float64)
    metrics = {**p2.outcome_metrics(offset, episodes.np["premium"]),
               "offset_cvar10": float(p2.lower_cvar(offset, 0.10)),
               "he_cvar10": float(p2.lower_cvar(offset + episodes.np["premium"], 0.10)),
               "he_cvar25": float(p2.lower_cvar(offset + episodes.np["premium"], 0.25)),
               "cost_mean": float(r["cost"].numpy().mean()),
               **p2.position_metrics(positions, episodes.np["listed_delta"], episodes.np["liab_delta"])}
    if "weights" in r:
        metrics.update(p2.prototype_metrics(r["weights"].numpy()))
    return {"metrics": metrics, "offset": offset, "positions": positions,
            "weights": r["weights"].numpy() if "weights" in r else None,
            "outputs": r["outputs"].numpy()}


def liability_vega(episodes: Episodes) -> np.ndarray:
    """Model vega of the liability by step [n, T], per unit of volatility, in units of the
    initial spot (finite difference of the same pricing functions as the delta)."""
    spot, ivol = episodes.features["spot"], episodes.features["ivol"]
    bump = 0.01
    vega = np.zeros_like(spot)
    for t in range(N_STEPS):
        if episodes.liability == "european_call":
            tau = (N_STEPS - t) * p2.DT
            up, _ = p2.european_price_delta(spot[:, t], ivol[:, t] + bump, tau)
            dn, _ = p2.european_price_delta(spot[:, t], np.maximum(ivol[:, t] - bump, 1e-4), tau)
        else:
            run_sum = np.asarray(episodes.world.data.features.per_step["asian_running_average"],
                                 dtype=np.float64)[:, t] * t
            up, _ = p2.asian_price_delta(spot[:, t], ivol[:, t] + bump, run_sum, t)
            dn, _ = p2.asian_price_delta(spot[:, t], np.maximum(ivol[:, t] - bump, 1e-4), run_sum, t)
        vega[:, t] = (up - dn) / (2 * bump)
    return vega


def delta_vega_targets(episodes: Episodes) -> np.ndarray:
    """Textbook two-instrument hedge: listed calls neutralise the liability's vega, the
    share position neutralises the remaining delta. Positions are clipped to the limits."""
    listed_vega = np.asarray(episodes.world.data.features.per_step["call_vega"], dtype=np.float64)
    n_call = np.clip(liability_vega(episodes) / np.maximum(listed_vega, 1e-8), -POS_BOUND, POS_BOUND)
    n_spot = np.clip(episodes.np["liab_delta"] - n_call * episodes.np["listed_delta"], -POS_BOUND, POS_BOUND)
    return np.stack([n_spot, n_call], axis=2)


def rule_metrics(episodes: Episodes, static_from: Episodes | None = None) -> dict:
    """Rule-based references on the same episodes (liability delta, delta-vega, unhedged, constants)."""
    n, market = episodes.n, episodes.world.data.market
    tg = np.zeros((n, N_STEPS, 2))
    tg[:, :, 0] = episodes.np["liab_delta"]
    targets = {"unhedged": p2.constant_targets(n, 0, 0), "liability_delta": tg,
               "delta_vega": delta_vega_targets(episodes),
               "hold_share": p2.constant_targets(n, 1, 0)}
    if static_from is not None:
        static = p2.select_static_positions(static_from.world.data.market)
        targets["static_oce"] = p2.constant_targets(n, *static["static_oce"])
        targets["static_minvar"] = p2.constant_targets(n, *static["static_minvar"])
    out = {}
    for name, target in targets.items():
        sim = p2.simulate_targets(market, target)
        he = sim["offset"] + episodes.np["premium"]
        out[name] = {**p2.outcome_metrics(sim["offset"], episodes.np["premium"]),
                     "offset_cvar10": float(p2.lower_cvar(sim["offset"], 0.10)),
                     "he_cvar10": float(p2.lower_cvar(he, 0.10)), "he_cvar25": float(p2.lower_cvar(he, 0.25)),
                     "cost_mean": float(sim["cost"].mean())}
    return out

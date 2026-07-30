import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from deephedging.base_torch import torchCast


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _weighted_mean(weights, values):
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.sum(weights * values))


def _config_value(config, world, key, default):
    world_cfg = None

    if isinstance(config, dict):
        world_cfg = config.get("world", {})
    elif hasattr(config, "world"):
        world_cfg = getattr(config, "world")

    if world_cfg is not None:
        if isinstance(world_cfg, dict) and key in world_cfg:
            return world_cfg[key]
        if hasattr(world_cfg, "get_raw"):
            try:
                return world_cfg.get_raw(key, default)
            except Exception:
                pass

    if hasattr(world, "config"):
        world_cfg = world.config
        if isinstance(world_cfg, dict) and key in world_cfg:
            return world_cfg[key]
        if hasattr(world_cfg, "get_raw"):
            try:
                return world_cfg.get_raw(key, default)
            except Exception:
                pass
        try:
            return world_cfg(key, default)
        except Exception:
            pass

    return default


def _bin_sorted_means(sorted_x, sorted_y, n_bins):
    n = len(sorted_x)
    idx = np.linspace(0, n, n_bins + 1, endpoint=True, dtype=np.int32)
    x_out = np.zeros((n_bins,), dtype=np.float64)
    y_out = np.zeros((n_bins,), dtype=np.float64)

    for i in range(n_bins):
        lo = int(idx[i])
        hi = int(idx[i + 1])
        if hi <= lo:
            lo = min(lo, n - 1)
            hi = min(lo + 1, n)
        x_out[i] = np.mean(sorted_x[lo:hi])
        y_out[i] = np.mean(sorted_y[lo:hi])

    return x_out, y_out


def _spot_histogram(spots_t, bin_bnd):
    counts, _ = np.histogram(spots_t, bins=bin_bnd)
    total = max(int(np.sum(counts)), 1)
    return counts / float(total)


def _evaluate_gym(world, gym):
    with torch.no_grad():
        result = gym.forward(torchCast(world.torch_data), training=False, return_paths=True)
    return {k: _np(v) for k, v in result.items()}


def compare_hedges(hedge_proto, hedge_bs):
    hedge_proto = _np(hedge_proto)
    hedge_bs = _np(hedge_bs)

    plt.figure(figsize=(6, 4))
    plt.plot(hedge_proto.mean(0))
    plt.plot(hedge_bs.mean(0))
    plt.legend(["ProtoHedge", "BlackScholes"])
    plt.title("Average hedge comparison")
    plt.tight_layout()
    plt.show()


def plot_blackscholes(world, gym, config, strike=1.0, iscall=True, save_prefix=None, dpi=160, show=True):
    """
    Torch port of deephedging.plot_bs_hedge.plot_blackscholes.
    Produces the same BS comparison diagnostics as the original TF notebook.
    """

    r = _evaluate_gym(world, gym)
    spot = _np(world.details.spot_all[:, :-1])
    spot_t_final = _np(world.details.spot_all[:, -1])
    hedges = _np(world.data.market.hedges[:, :, 0])
    costs = _np(world.data.market.cost[:, :, 0])
    actions = _np(r["actions"])[:, :, 0]
    deltas = np.cumsum(actions, axis=1)
    time_left = _np(world.data.features.per_step["time_left"])
    dh_pnl = _np(r["pnl"]).reshape(-1)
    payoff = _np(r["payoff"]).reshape(-1)

    utility = _weighted_mean(world.sample_weights, _np(r["utility"]).reshape(-1))
    utility0 = _weighted_mean(world.sample_weights, _np(r["utility0"]).reshape(-1))
    dprice = utility - utility0

    dt = float(_config_value(config, world, "dt", 1.0 / 50.0))
    vol = float(_config_value(config, world, "rvol", np.mean(_np(world.data.features.per_step["ivol"]))))
    risk_measure = (
        config.get("objective", {}).get("risk_measure", "utility")
        if isinstance(config, dict)
        else "utility"
    )
    risk_aversion = (
        config.get("objective", {}).get("risk_aversion", 1.0)
        if isinstance(config, dict)
        else 1.0
    )

    bins = 20
    n_spots = spot.shape[0]
    lohi = np.quantile(spot_t_final, [0.01, 0.99])
    bin_bnd = np.linspace(lohi[0], lohi[1], bins + 1, endpoint=True)
    bin_mid = 0.5 * (bin_bnd[1:] + bin_bnd[:-1])
    delta_bins = min(100, max(10, n_spots - 1))

    fig_any, top_axes = plt.subplots(1, 5, figsize=(18, 4.5))
    ax_term_payoff, ax_terminal, ax_utility, ax_spots, ax_hedges = top_axes

    time_steps = deltas.shape[1]
    n_path_axes = time_steps * 2
    n_cols = 6
    n_rows = int(math.ceil(n_path_axes / float(n_cols)))
    fig_path, path_axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.2 * n_rows))
    path_axes = np.asarray(path_axes).reshape(-1)

    last_delta = np.zeros((n_spots,), dtype=np.float64)
    last_bs_delta = np.zeros((n_spots,), dtype=np.float64)
    model_pnl = np.zeros((n_spots,), dtype=np.float64)
    bs_pnl = np.zeros((n_spots,), dtype=np.float64)
    bs_cost_total = np.zeros((n_spots,), dtype=np.float64)

    print("Running strategies ...", end="")
    for j in range(time_steps):
        spot_t = spot[:, j]
        delta_t = deltas[:, j]
        hedges_t = hedges[:, j]
        cost_t = costs[:, j]
        t = float(j) * dt
        res_t = float(time_left[0, j])

        safe_res_t = max(res_t, 1e-12)
        safe_spot_t = np.maximum(spot_t, 1e-12)
        d1 = (np.log(safe_spot_t / strike) + 0.5 * vol * vol * safe_res_t) / math.sqrt(safe_res_t * vol * vol)
        d2 = d1 - vol * math.sqrt(safe_res_t)
        n1 = norm.cdf(d1)
        n2 = norm.cdf(d2)
        bs_price_t = spot_t * n1 - strike * n2
        if not iscall:
            bs_price_t = bs_price_t + strike - spot_t
        bs_delta_t = n1 if iscall else (n1 - 1.0)

        act_t = delta_t - last_delta
        bs_act_t = bs_delta_t - last_bs_delta
        model_pnl = act_t * hedges_t + model_pnl
        bs_pnl = bs_act_t * hedges_t + bs_pnl
        bs_cost_total = bs_cost_total + np.abs(bs_act_t) * cost_t
        last_delta = delta_t
        last_bs_delta = bs_delta_t

        sim_price_t = model_pnl + delta_t * (spot_t_final - spot_t)

        ixs = np.argsort(spot_t)
        srt_spot_t = spot_t[ixs]
        srt_delta_t = delta_t[ixs]
        srt_bs_delta_t = bs_delta_t[ixs]
        srt_hedges_t = hedges_t[ixs]
        srt_sim_price_t = sim_price_t[ixs]
        srt_bs_price_t = bs_price_t[ixs]

        bin_spot_t, bin_delta_t = _bin_sorted_means(srt_spot_t, srt_delta_t, delta_bins)
        _, bin_bs_delta_t = _bin_sorted_means(srt_spot_t, srt_bs_delta_t, delta_bins)
        _, bin_sim_price_t = _bin_sorted_means(srt_spot_t, srt_sim_price_t, delta_bins)
        _, bin_bs_price_t = _bin_sorted_means(srt_spot_t, srt_bs_price_t, delta_bins)

        frac = float(j + 1) / float(time_steps)
        ax_spots.plot(bin_mid, _spot_histogram(np.sort(spot_t), bin_bnd), color=(frac, 0.5, 0.5))
        ax_hedges.plot(
            srt_spot_t,
            srt_hedges_t,
            color=(0.7, frac, 0.7) if j < time_steps - 1 else (0.8, 1.0, 0.8),
            alpha=0.8,
        )

        ax_payoff = path_axes[2 * j]
        ax_delta = path_axes[2 * j + 1]

        ax_payoff.plot(
            bin_spot_t,
            bin_sim_price_t,
            "-" if j > 0 else "o",
            label="Model approximation",
            color=(0.0, 0.0, 1.0),
            linewidth=1.2,
        )
        ax_payoff.plot(
            bin_spot_t,
            bin_bs_price_t - dprice,
            "-" if j > 0 else "o",
            label="Black Scholes",
            color=(0.9, 0.1, 0.1),
            linewidth=1.2,
        )
        ax_payoff.set_title(f"-Payoff {t * 255.0:g} days")
        if j == 1:
            ax_payoff.legend(fontsize=8)

        ax_delta.plot(
            srt_spot_t,
            srt_delta_t,
            "-" if j > 0 else "o",
            label="model",
            color=(0.0, 0.0, 1.0),
            alpha=0.25,
            linewidth=0.8,
        )
        ax_delta.plot(
            bin_spot_t,
            bin_delta_t,
            "-" if j > 0 else "o",
            label="model (smoothed)",
            color=(0.0, 0.0, 0.6),
            linewidth=1.4,
        )
        ax_delta.plot(
            srt_spot_t,
            srt_bs_delta_t,
            "-" if j > 0 else "o",
            label="black scholes",
            color=(0.9, 0.1, 0.1),
            linewidth=1.0,
        )
        ax_delta.set_title(f"Delta {t * 255.0:g} days")
        if j == 1:
            ax_delta.legend(fontsize=8)

    print("done")

    bs_gains = payoff + bs_pnl - bs_cost_total
    with torch.no_grad():
        utility_bs_path = gym.objective.utility(torch.tensor(bs_gains, dtype=torch.float32))
    utility_bs = _weighted_mean(world.sample_weights, _np(utility_bs_path).reshape(-1))

    ixs = np.argsort(spot_t_final)
    srt_spot_t_final = spot_t_final[ixs]
    srt_gain = (model_pnl + payoff)[ixs]
    srt_bs_gain = (bs_pnl + payoff)[ixs]
    srt_dh_gain = (dh_pnl + payoff)[ixs]
    srt_payoff = payoff[ixs]
    srt_eff = model_pnl[ixs]
    srt_dh_eff = dh_pnl[ixs]
    srt_bs_eff = bs_pnl[ixs]

    bin_spot_t_final, bin_gain = _bin_sorted_means(srt_spot_t_final, srt_gain, delta_bins)
    _, bin_bs_gain = _bin_sorted_means(srt_spot_t_final, srt_bs_gain, delta_bins)
    _, bin_dh_gain = _bin_sorted_means(srt_spot_t_final, srt_dh_gain, delta_bins)

    min_y = min(np.min(bin_gain), np.min(bin_bs_gain), np.min(bin_dh_gain))
    max_y = max(np.max(bin_gain), np.max(bin_bs_gain), np.max(bin_dh_gain))
    dy = max(max_y - min_y, 1e-8)
    ax_terminal.plot(bin_spot_t_final, bin_gain, "*-", color="orange", label="hedged pnl")
    ax_terminal.plot(bin_spot_t_final, bin_bs_gain, "-", color="green", label="bs hedged pnl")
    ax_terminal.plot(bin_spot_t_final, bin_dh_gain, ":", color="black", label="hedged pnl from DH")
    ax_terminal.set_ylim(min_y - 0.25 * dy, max_y + 0.25 * dy)
    ax_terminal.set_title("Terminal Hedged Results")
    ax_terminal.legend(fontsize=8)

    _, bin_payoff = _bin_sorted_means(srt_spot_t_final, srt_payoff, delta_bins)
    _, bin_eff = _bin_sorted_means(srt_spot_t_final, srt_eff, delta_bins)
    _, bin_dh_eff = _bin_sorted_means(srt_spot_t_final, srt_dh_eff, delta_bins)
    _, bin_bs_eff = _bin_sorted_means(srt_spot_t_final, srt_bs_eff, delta_bins)

    min_y = min(np.min(bin_payoff), np.min(bin_eff), np.min(bin_dh_eff), np.min(bin_bs_eff))
    max_y = max(np.max(bin_payoff), np.max(bin_eff), np.max(bin_dh_eff), np.max(bin_bs_eff))
    dy = max(max_y - min_y, 1e-8)
    ax_term_payoff.plot(bin_spot_t_final, bin_payoff, "-", color="blue", label="payoff")
    ax_term_payoff.plot(bin_spot_t_final, -bin_eff - dprice, "*-", color="orange", label="-model hedged pnl")
    ax_term_payoff.plot(bin_spot_t_final, -bin_bs_eff - dprice, "-", color="green", label="-bs hedged pnl")
    ax_term_payoff.set_ylim(min_y - 0.25 * dy, max_y + 0.25 * dy)
    ax_term_payoff.set_title("Effective Terminal Payoffs")
    ax_term_payoff.legend(fontsize=8)

    ax_utility.bar(
        ["Unhedged", "BS", "DH"],
        [utility0, utility_bs, utility],
        color=["tab:blue", "tab:blue", "tab:blue"],
    )
    ax_utility.set_title(f"Utility {risk_measure}@{risk_aversion:g}\n(higher is better)")

    ax_spots.set_title("Spot distribution in t")
    ax_hedges.set_title("Hedge Returns")

    for ax in top_axes:
        ax.grid(True, alpha=0.25)

    for k in range(n_path_axes, len(path_axes)):
        path_axes[k].axis("off")

    fig_any.tight_layout()
    fig_path.tight_layout()

    if save_prefix is not None:
        fig_any.savefig(f"{save_prefix}_summary.png", dpi=dpi, bbox_inches="tight")
        fig_path.savefig(f"{save_prefix}_paths.png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    return {"summary_fig": fig_any, "path_fig": fig_path}

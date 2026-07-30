import math

import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _mean(weights, values):
    weights = _np(weights).reshape(-1)
    values = _np(values).reshape(-1)
    return float(np.sum(weights * values))


def _err(weights, values):
    weights = _np(weights).reshape(-1)
    values = _np(values).reshape(-1)
    mean = _mean(weights, values)
    var = float(np.sum(weights * ((values - mean) ** 2)))
    return math.sqrt(var) / math.sqrt(len(weights))


def _mean_bins(x, bins, weights=None, return_std=False):
    x = _np(x).reshape(-1)
    weights = _np(weights).reshape(-1) if weights is not None else np.full(x.shape, 1.0 / len(x))
    if len(x) <= bins:
        if return_std:
            return x, np.zeros_like(x)
        return x

    ixs = np.linspace(0, len(x), bins + 1, endpoint=True, dtype=np.int32)
    means = []
    stds = []
    for i1, i2 in zip(ixs[:-1], ixs[1:]):
        p = weights[i1:i2]
        xx = x[i1:i2]
        pp = np.sum(p)
        if pp <= 1e-12:
            means.append(0.0)
            stds.append(0.0)
        else:
            m = np.sum(p * xx) / pp
            v = np.sum(p * ((xx - m) ** 2)) / pp
            means.append(m)
            stds.append(math.sqrt(v))
    means = np.asarray(means)
    stds = np.asarray(stds)
    if return_std:
        return means, stds
    return means


def _mean_cum_bins(x, bins, weights=None):
    x = _np(x).reshape(-1)
    weights = _np(weights).reshape(-1) if weights is not None else None
    if len(x) <= bins:
        return x
    ixs = np.linspace(0, len(x), bins + 1, endpoint=True, dtype=np.int32)
    out = []
    for i in ixs[1:]:
        if weights is None:
            out.append(np.mean(x[:i]))
        else:
            out.append(np.sum((x * weights)[:i]) / np.sum(weights[:i]))
    return np.asarray(out)


def _perct_exp(x, lo, hi, weights=None):
    x = _np(x)
    if x.ndim == 2:
        return np.array([_perct_exp(x[:, i], lo, hi, weights=weights) for i in range(x.shape[1])])

    weights = _np(weights).reshape(-1) if weights is not None else None
    ixs = np.argsort(x)
    x = x[ixs]
    if weights is not None:
        weights = weights[ixs]
    ix_lo = min(math.ceil(x.shape[0] * lo), x.shape[0] - 1)
    ix_hi = max(math.floor(x.shape[0] * hi), 0)
    lo_val = np.sum((weights * x)[:ix_lo]) / np.sum(weights[:ix_lo]) if weights is not None else np.mean(x[:ix_lo])
    hi_val = np.sum((weights * x)[ix_hi:]) / np.sum(weights[ix_hi:]) if weights is not None else np.mean(x[ix_hi:])
    return np.array([lo_val, hi_val])


def _plot_loss(ax, losses, loss_errs, best_epoch, best_loss, show_epochs, title):
    epochs = len(losses["training"])
    show_epoch0 = max(0, epochs - show_epochs)
    x = np.arange(show_epoch0 + 1, epochs + 1)

    colors = {"batch": "tab:blue", "training": "tab:orange", "val": "tab:green"}
    styles = {"batch": "-", "training": "-", "val": ":"}
    alphas = {"training": 0.2, "val": 0.05}

    for key in ["batch", "training", "val"]:
        y = np.asarray(losses[key])[show_epoch0:epochs]
        ax.plot(x, y, styles.get(key, "-"), label=key, color=colors[key])
        if key in loss_errs:
            err = np.asarray(loss_errs[key])[show_epoch0:epochs]
            ax.fill_between(x, y - err, y + err, color=colors[key], alpha=alphas.get(key, 0.15))

    if best_epoch >= 0:
        ax.plot([max(best_epoch + 1, x[0])], [best_loss], "*", color="black", label="best")

    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.legend()


def _plot_utility_epoch(ax, training_util, training_util_err, val_util, best_epoch, label, title):
    epochs = len(training_util)
    x = np.arange(1, epochs + 1)
    ax.plot(x, training_util, "-", color="red", label=f"{label}, training")
    ax.plot(x, val_util, ":", color="red", label=f"{label}, val")
    ax.fill_between(
        x,
        np.asarray(training_util) - np.asarray(training_util_err),
        np.asarray(training_util) + np.asarray(training_util_err),
        color="red",
        alpha=0.2,
    )
    if best_epoch >= 0:
        ax.plot([best_epoch + 1], [training_util[best_epoch]], "*", color="black", label="best training")
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.legend()


def _plot_returns(ax, weights, gains, hedge, payoff, spot_ret, with_std, title):
    ixs = np.argsort(spot_ret)
    x = spot_ret[ixs]
    gains = gains[ixs]
    hedge = hedge[ixs]
    payoff = payoff[ixs]
    x = _mean_bins(x, bins=100, weights=weights, return_std=False)
    gains_m, gains_s = _mean_bins(gains, bins=100, weights=weights, return_std=True)
    hedge_m, hedge_s = _mean_bins(hedge, bins=100, weights=weights, return_std=True)
    payoff_m, payoff_s = _mean_bins(payoff, bins=100, weights=weights, return_std=True)

    ax.plot(x, gains_m, color="blue", label="gains")
    ax.plot(x, payoff_m, ":", color="orange", label="payoff")
    ax.plot(x, -hedge_m, color="green", label="-hedge")
    ax.plot(x, payoff_m * 0.0, ":", color="black")

    if with_std:
        ax.fill_between(x, gains_m - gains_s, gains_m + gains_s, color="blue", alpha=0.2)
        ax.fill_between(x, payoff_m - payoff_s, payoff_m + payoff_s, color="orange", alpha=0.2)
        ax.fill_between(x, -hedge_m - hedge_s, -hedge_m + hedge_s, color="green", alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Spot return")
    ax.legend()


def _plot_utility_percentile(ax, weights, utility, utility0, title):
    utility = np.sort(utility)
    utility0 = np.sort(utility0)
    u = _mean_cum_bins(utility, bins=100, weights=weights)
    u0 = _mean_cum_bins(utility0, bins=100, weights=weights)
    x = np.linspace(0.0, 1.0, len(u), endpoint=True)
    ax.plot(x, u, label="gains")
    ax.plot([x[-1]], [u[-1]], "*", color="blue")
    ax.plot(x, u0, "-", label="payoff")
    ax.plot(x, u0 * 0.0, ":", color="black")
    ax.set_title(title)
    ax.set_xlabel("Percentile")
    ax.legend()


def _plot_activity_by_step(ax, weights, actions, inst_names, title, lo=0.25, hi=0.75):
    n_steps = actions.shape[1]
    x = np.arange(1, n_steps + 1)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i in range(actions.shape[2]):
        a = actions[:, :, i]
        pc = _perct_exp(a, lo=lo, hi=hi, weights=weights)
        mean_i = np.sum(a * weights[:, None], axis=0) / np.sum(weights)
        ax.plot(x, mean_i, "-", color=colors[i % len(colors)], label=inst_names[i])
        ax.fill_between(x, pc[:, 0], pc[:, 1], color=colors[i % len(colors)], alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.legend()


def _plot_activity_by_spot_time(ax, weights, actions, spot_all, which_inst, with_std, title, slices=10):
    n_time = actions.shape[1]
    time_ixs = np.linspace(0, n_time - 1, min(slices, n_time), endpoint=True, dtype=np.int32)
    actions = actions[:, :, which_inst]

    for idx, t in enumerate(time_ixs):
        x = spot_all[:, t] / spot_all[:, 0] - 1.0
        ixs = np.argsort(x)
        x = x[ixs]
        x = _mean_bins(x, bins=100, weights=weights, return_std=False)
        act, std = _mean_bins(actions[:, t][ixs], bins=100, weights=weights, return_std=True)
        r = 2.0 * (1.0 - float(t + 1) / float(n_time))
        c1 = max(min(r - 1.0, 1.0), 0.0)
        c2 = max(min(r, 1.0), 0.0)
        color = (1.0, c1, c2)
        ax.plot(x, act, color=color, label=f"{t + 1}")
        if with_std:
            ax.fill_between(x, act - std, act + std, color=color, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Spot return")
    ax.legend()


def plot_full_diagnostics(results, save_prefix=None, dpi=160, show=True):
    history = results["history"]
    train = {k: _np(v) for k, v in results["training_result"].items()}
    val = {k: _np(v) for k, v in results["val_result"].items()}
    world = results["world"]
    val_world = results["val_world"]
    weights = _np(results["sample_weights"])
    val_weights = _np(results["val_sample_weights"])

    fig = plt.figure(figsize=(22, 34))

    ax = plt.subplot(5, 4, 1)
    _plot_loss(
        ax,
        history["losses"],
        history["losses_err"],
        history["best_epoch"],
        history["best_loss"],
        show_epochs=100,
        title="Losses (recent)",
    )

    ax = plt.subplot(5, 4, 2)
    _plot_loss(
        ax,
        history["losses"],
        history["losses_err"],
        history["best_epoch"],
        history["best_loss"],
        show_epochs=max(1, len(history["losses"]["training"])),
        title="Losses (all)",
    )

    ax = plt.subplot(5, 4, 3)
    _plot_utility_epoch(
        ax,
        history["utilities"]["training_util"],
        history["utilities"]["training_util_err"],
        history["utilities"]["val_util"],
        history["best_epoch"],
        "gains",
        "Model Gains Monetay Utility",
    )

    ax = plt.subplot(5, 4, 4)
    _plot_utility_epoch(
        ax,
        history["utilities"]["training_util0"],
        history["utilities"]["training_util0_err"],
        history["utilities"]["val_util0"],
        history["best_epoch"],
        "payoff",
        "Original Payoff Monetay Utility",
    )

    ax = plt.subplot(5, 4, 5)
    ax.plot(history["process"]["memory_rss"], label="rss", color="blue")
    ax.plot(history["process"]["memory_vms"], label="vms", color="green")
    ax.set_title("Memory usage by epoch")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Memory (GB)")
    ax.legend()

    spot_ret = world.details.spot_all[:, -1] / world.details.spot_all[:, 0] - 1.0
    val_spot_ret = val_world.details.spot_all[:, -1] / val_world.details.spot_all[:, 0] - 1.0

    adjusted_training_gains = train["gains"] - _mean(weights, train["utility"])
    adjusted_training_payoff = train["payoff"] - _mean(weights, train["utility0"])
    adjusted_training_hedge = adjusted_training_gains - adjusted_training_payoff

    adjusted_val_gains = val["gains"] - _mean(val_weights, val["utility"])
    adjusted_val_payoff = val["payoff"] - _mean(val_weights, val["utility0"])
    adjusted_val_hedge = adjusted_val_gains - adjusted_val_payoff

    ax = plt.subplot(5, 4, 6)
    _plot_returns(
        ax,
        weights,
        adjusted_training_gains,
        adjusted_training_hedge,
        adjusted_training_payoff,
        spot_ret,
        with_std=False,
        title="Returns less Utility\n(training set)",
    )

    ax = plt.subplot(5, 4, 7)
    _plot_returns(
        ax,
        weights,
        adjusted_training_gains,
        adjusted_training_hedge,
        adjusted_training_payoff,
        spot_ret,
        with_std=True,
        title="Returns less Utility (with std)\n(training set)",
    )

    ax = plt.subplot(5, 4, 8)
    _plot_utility_percentile(
        ax,
        weights,
        train["utility"],
        train["utility0"],
        "Utility by cummulative percentile\n(training set)",
    )

    ax = plt.subplot(5, 4, 9)
    _plot_returns(
        ax,
        val_weights,
        adjusted_val_gains,
        adjusted_val_hedge,
        adjusted_val_payoff,
        val_spot_ret,
        with_std=False,
        title="Returns less Utility\n(validation set)",
    )

    ax = plt.subplot(5, 4, 10)
    _plot_returns(
        ax,
        val_weights,
        adjusted_val_gains,
        adjusted_val_hedge,
        adjusted_val_payoff,
        val_spot_ret,
        with_std=True,
        title="Returns less Utility (with std)\n(validation set)",
    )

    ax = plt.subplot(5, 4, 11)
    _plot_utility_percentile(
        ax,
        val_weights,
        val["utility"],
        val["utility0"],
        "Utility by cummulative percentile\n(validation set)",
    )

    ax = plt.subplot(5, 4, 12)
    _plot_activity_by_step(
        ax,
        weights,
        train["actions"],
        results["inst_names"],
        "Action by time step\n(training set)",
    )

    ax = plt.subplot(5, 4, 13)
    _plot_activity_by_step(
        ax,
        weights,
        np.cumsum(train["actions"], axis=1),
        results["inst_names"],
        "Delta by time step\n(training set)",
    )

    ax = plt.subplot(5, 4, 14)
    _plot_activity_by_spot_time(
        ax,
        weights,
        train["actions"],
        world.details.spot_all,
        which_inst=0,
        with_std=False,
        title="Spot action by time step\n(training set)",
    )

    ax = plt.subplot(5, 4, 15)
    _plot_activity_by_spot_time(
        ax,
        weights,
        np.cumsum(train["actions"], axis=1),
        world.details.spot_all,
        which_inst=0,
        with_std=False,
        title="Spot delta by time step\n(training set)",
    )

    ax = plt.subplot(5, 4, 16)
    _plot_activity_by_spot_time(
        ax,
        weights,
        train["actions"],
        world.details.spot_all,
        which_inst=0,
        with_std=True,
        title="Spot action by time step (with std)\n(training set)",
    )

    ax = plt.subplot(5, 4, 17)
    _plot_activity_by_spot_time(
        ax,
        weights,
        np.cumsum(train["actions"], axis=1),
        world.details.spot_all,
        which_inst=0,
        with_std=True,
        title="Spot delta by time step (with std)\n(training set)",
    )

    for idx in [18, 19, 20]:
        ax = plt.subplot(5, 4, idx)
        ax.axis("off")

    plt.tight_layout()
    if save_prefix is not None:
        fig.savefig(f"{save_prefix}.png", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig

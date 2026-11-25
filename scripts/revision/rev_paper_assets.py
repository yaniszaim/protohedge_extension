#!/usr/bin/env python3
"""Tables and figures for the manuscript, generated from the final evaluation.

Reads ``<study>/final`` (or ``final_rehearsal_val`` with ``--rehearsal``) and
the training histories, and writes LaTeX fragments to
``paper/revision_2026_11/manuscript/generated`` and figures to
``manuscript/figures``. Nothing here selects or filters results: every arm,
ticker and seed in the evaluation enters the averages.

    python scripts/revision/rev_paper_assets.py --spec paper/revision_2026_11/phase3/final_spec.json [--rehearsal]

All money quantities are printed in percent of the initial underlying price
(the stored values times 100).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import phase3_fit  # noqa: E402
import phase3_final  # noqa: E402

PAPER = Path(__file__).resolve().parents[2] / "paper" / "revision_2026_11" / "manuscript"
LIABS = [("european_call", "European call"), ("asian_call", "Asian call")]
LABEL = {"unhedged": "Unhedged", "liability_delta": "Delta hedge", "delta_vega": "Delta--vega hedge",
         "static_minvar": "Static (min. variance)", "hold_share": "Hold one share",
         "spot_delta_listed": "Listed-call delta", "dh": "DH", "ph": "PH (encoder)", "phd": "PH (plain)",
         "dh_w20": "DH, width 20", "ph_k100": "PH (encoder), $K{=}100$",
         "dh_cvar10": "DH", "ph_cvar10": "PH (encoder)", "dh_cvar50": "DH", "ph_cvar50": "PH (encoder)",
         "dh_trade": "DH", "ph_trade": "PH (encoder)",
         "dh_pooled": "DH", "ph_pooled": "PH (encoder)", "phd_pooled": "PH (plain)",
         "dh_orig": "DH, width 20", "ph_orig": "PH (encoder), $K{=}100$"}
COLOR = {"dh": "#2a78d6", "ph": "#eb6834", "phd": "#1baf7a", "delta_vega": "#6b7280"}
INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"


def pct(x, digits=2, sign=False):
    return (f"{100 * x:+.{digits}f}" if sign else f"{100 * x:.{digits}f}").replace("-", "$-$")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, length=2.5, width=0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load(spec, rehearsal):
    out = phase3_final.out_dir(spec, rehearsal)
    rows = pd.read_csv(out / "rows_by_ticker_seed.csv")
    panel = pd.read_csv(out / "scoreboard_panel.csv").set_index(["liability", "arm"])
    by_ticker = pd.read_csv(out / "scoreboard_by_ticker.csv")
    contrasts = pd.read_csv(out / "contrasts.csv")
    return out, rows, panel, by_ticker, contrasts


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def table_main(panel, rows, gen):
    """Arm A and the rule-based references, both liabilities."""
    arms = ["unhedged", "liability_delta", "delta_vega", "dh", "ph", "phd"]
    lines = [r"\begin{tabular}{@{}l rrrr c rrrr@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{European call} && \multicolumn{4}{c}{Asian call} \\",
             r"\cmidrule{2-5}\cmidrule{7-10}",
             r"Strategy & Std $E$ & $\mathrm{CVaR}_{5\%}(E)$ & Mean $E$ & Cost && Std $E$ & $\mathrm{CVaR}_{5\%}(E)$ & Mean $E$ & Cost \\",
             r"\midrule"]
    for arm in arms:
        if arm == "dh":
            lines.append(r"\midrule")
        cells = []
        for liab, _ in LIABS:
            r = panel.loc[(liab, arm)]
            cells.append(" & ".join([pct(r.he_std, 3), pct(r.he_cvar05), pct(r.he_mean, 2, True), pct(r.cost_mean)]))
        lines.append(f"{LABEL[arm]} & {cells[0]} && {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_main.tex").write_text("\n".join(lines) + "\n")


def table_contrasts(contrasts, spec, gen):
    """Primary contrasts at the first block length, with Holm-adjusted p-values."""
    block = spec.get("blocks", [20])[0]
    c = contrasts[(contrasts.block == block) & contrasts.primary]
    names = {"he_std": "Std $E$", "he_cvar05": r"$\mathrm{CVaR}_{5\%}(E)$"}
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Contrast & Quantity & Difference & 95\% interval & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for liab, title in LIABS:
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for _, r in c[c.liability == liab].iterrows():
            p = "$<$0.01" if r.p_holm < 0.01 else f"{r.p_holm:.2f}"
            short = {"ph": "PH", "dh": "DH", "liability_delta": "delta"}
            lines.append(f"{short[r.candidate]} $-$ {short[r.benchmark]} & {names[r.metric]} & "
                         f"{pct(r.difference, 3, True)} & [{pct(r.ci95_low, 3, True)}, {pct(r.ci95_high, 3, True)}] & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_contrasts.tex").write_text("\n".join(lines) + "\n")


def table_arms(panel, rows, spec, gen):
    """One-factor variations of arm A: std of E relative to the delta hedge, tail, mean, failures."""
    have = {a["name"] for a in spec["arms"]}
    groups = [("Arm A: variance of $E$, correction to delta", ["dh", "ph", "phd"]),
              (r"Objective $\mathrm{CVaR}_{10\%}(E)$", ["dh_cvar10", "ph_cvar10"]),
              (r"Objective $\mathrm{CVaR}_{50\%}(O)$ (submitted)", ["dh_cvar50", "ph_cvar50"]),
              ("Output is a trade (original semantics)", ["dh_trade", "ph_trade"]),
              ("Original sizes", ["dh_w20", "ph_k100"]),
              ("One model for all ten tickers", ["dh_pooled", "ph_pooled", "phd_pooled"]),
              (r"All original settings: trade output, $\mathrm{CVaR}_{50\%}(O)$, original inputs and sizes",
               ["dh_orig", "ph_orig"])]
    fails = rows[rows.seed >= 0].groupby(["liability", "arm"]).failed.agg(["sum", "size"])
    lines = [r"\begin{tabular}{@{}l rrrrr c rrrrr@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{European call} && \multicolumn{5}{c}{Asian call} \\",
             r"\cmidrule{2-6}\cmidrule{8-12}",
             r"Variation & Std ratio & $\mathrm{CVaR}_{5\%}(E)$ & Mean $E$ & $\mathrm{CVaR}_{50\%}(O)$ & Failed && Std ratio & $\mathrm{CVaR}_{5\%}(E)$ & Mean $E$ & $\mathrm{CVaR}_{50\%}(O)$ & Failed \\",
             r"\midrule"]
    ref_cells = []
    for name in ("unhedged", "liability_delta"):
        cells = []
        for liab, _ in LIABS:
            r, ref = panel.loc[(liab, name)], panel.loc[(liab, "liability_delta")]
            cells.append(" & ".join([f"{r.he_std / ref.he_std:.2f}", pct(r.he_cvar05), pct(r.he_mean, 2, True),
                                     pct(r.offset_cvar50), "--"]))
        lines.append(f"{LABEL[name]} & {cells[0]} && {cells[1]} \\\\")
    for title, arms in groups:
        arms = [a for a in arms if a in have]
        if not arms:
            continue
        lines.append(rf"\multicolumn{{12}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for arm in arms:
            cells = []
            for liab, _ in LIABS:
                r, ref = panel.loc[(liab, arm)], panel.loc[(liab, "liability_delta")]
                f = fails.loc[(liab, arm)]
                cells.append(" & ".join([f"{r.he_std / ref.he_std:.2f}", pct(r.he_cvar05), pct(r.he_mean, 2, True),
                                         pct(r.offset_cvar50), f"{int(f['sum'])}/{int(f['size'])}"]))
            lines.append(rf"\quad {LABEL[arm]} & {cells[0]} && {cells[1]} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_arms.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def figure_tickers(rows, spec, figs):
    """Std of the hedging error by ticker, relative to the delta hedge (one shared log axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), sharey=True)
    tickers = spec["tickers"]
    offsets = {"delta_vega": -0.27, "dh": -0.09, "ph": 0.09, "phd": 0.27}
    marker = {"delta_vega": "D", "dh": "o", "ph": "s", "phd": "^"}
    for ax, (liab, title) in zip(axes, LIABS):
        d = rows[rows.liability == liab]
        ref = d[d.arm == "liability_delta"].set_index("ticker").he_std
        style(ax)
        ax.axhline(1.0, color=INK, linewidth=0.8)
        for arm in ("delta_vega", "dh", "ph", "phd"):
            g = d[d.arm == arm]
            for i, t in enumerate(tickers):
                v = g[g.ticker == t].he_std.to_numpy() / ref[t]
                x = i + offsets[arm]
                if len(v) > 1:
                    ax.plot([x, x], [v.min(), v.max()], color=COLOR[arm], linewidth=1.0, alpha=0.55)
                ax.plot([x], [v.mean()], marker[arm], color=COLOR[arm], markersize=3.6, markeredgecolor="white",
                        markeredgewidth=0.4, label=LABEL[arm].replace("--", "-") if i == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, fontsize=6.5)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_yticks([0.25, 0.5, 1, 2])
        ax.set_yticklabels(["0.25", "0.5", "1", "2"])
        ax.minorticks_off()
    axes[0].set_ylabel("Std of hedging error\n(delta hedge = 1)", fontsize=7, color=INK)
    axes[0].text(len(tickers) - 0.5, 1.03, "delta hedge", fontsize=6.5, color=INK, ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig_ticker.pdf")
    fig.savefig(figs / "fig_ticker.png", dpi=200)
    plt.close(fig)


def figure_convergence(spec, figs, grid=2000):
    """Median training curves of arm A (validation and training std of E over the delta hedge's)."""
    fams = [("dh", "dh_w128"), ("ph", "ph_k10"), ("phd", "phd_k10")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), sharey=True)
    stats = {}
    for ax, (liab, title) in zip(axes, LIABS):
        style(ax)
        ax.axhline(1.0, color=INK, linewidth=0.8)
        for arm, family in fams:
            curves = {"val": [], "train": []}
            for ticker in spec["tickers"]:
                ref = phase3_fit.references(ticker, liab)
                for seed in range(5):
                    a = {"ticker": ticker, "liability": liab, "mode": "residual", "objective": "var_he", "state": "s1",
                         "seed": seed}
                    h = json.load(open(phase3_fit.cell_dir(spec["study"], a) / family / "history.json"))
                    for split in ("val", "train"):
                        v = -np.asarray(h[split]) / ref[split]["liability_delta"]["he_std"]
                        if split == "val":                      # the checkpoint rule keeps the best so far
                            v = np.minimum.accumulate(v)
                        curves[split].append(np.concatenate([v, np.full(grid + 1 - len(v), v[-1])])[:grid + 1])
            x = np.arange(grid + 1)                              # index 0 = untrained policy
            val, train = np.median(curves["val"], axis=0), np.median(curves["train"], axis=0)
            lo, hi = np.quantile(curves["val"], [0.25, 0.75], axis=0)
            ax.fill_between(x[1:], lo[1:], hi[1:], color=COLOR[arm], alpha=0.13, linewidth=0)
            ax.plot(x[1:], val[1:], color=COLOR[arm], linewidth=1.4, label=LABEL[arm])
            ax.plot(x[1:], train[1:], color=COLOR[arm], linewidth=0.9, linestyle=(0, (3, 2)))
            stats[(liab, arm)] = {"val_final": float(val[-1]), "train_final": float(train[-1])}
        ax.set_xscale("log")
        ax.set_xlim(1, grid)
        ax.set_ylim(0, 1.3)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Epoch", fontsize=7, color=INK)
    axes[0].set_ylabel("Std of hedging error\n(delta hedge = 1)", fontsize=7, color=INK)
    axes[0].text(1900, 1.03, "delta hedge", fontsize=6.5, color=INK, va="bottom", ha="right")
    handles, labels = axes[1].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=MUTED, linewidth=1.4), plt.Line2D([], [], color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)))]
    labels += ["validation (best so far)", "training"]
    fig.legend(handles, labels, fontsize=6.5, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figs / "fig_convergence.pdf")
    fig.savefig(figs / "fig_convergence.png", dpi=200)
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------
# Training-data mechanism (stage E4; validation data)
# --------------------------------------------------------------------------

E4_FAMS = [("dh", "dh_w128"), ("ph", "ph_k10"), ("phd", "phd_k10")]
DEV = ["SPY", "TLT", "NVDA"]


def e4_rows(spec, study="E4_mechanism", seeds=(0, 1, 2)):
    """Validation std of E over the delta hedge's for every data level, family, ticker and seed."""
    rows = []
    for liab, _ in LIABS:
        for arm, family in E4_FAMS:
            for seed in seeds:
                pooled = (phase3_fit.ROOT / study / liab / "POOLED" / "residual__var_he__s1" / f"seed_{seed}"
                          / family / "complete.json")
                pooled = json.load(open(pooled)) if pooled.exists() else None
                for ticker in spec["tickers"]:
                    ref = phase3_fit.references(ticker, liab)["val"]["liability_delta"]["he_std"]
                    a = {"ticker": ticker, "liability": liab, "mode": "residual", "objective": "var_he", "state": "s1",
                         "seed": seed}
                    full = json.load(open(phase3_fit.cell_dir(spec["study"], a) / family / "complete.json"))
                    base = {"liability": liab, "arm": arm, "seed": seed, "ticker": ticker}
                    rows.append({**base, "level": "100%", "ratio": full["evaluations"]["val"]["he_std"] / ref})
                    if pooled is not None:
                        rows.append({**base, "level": "pooled",
                                     "ratio": pooled["evaluations"][f"val_{ticker}"]["he_std"] / ref})
                    for pct_ in (25, 50):
                        f = (phase3_fit.ROOT / study / liab / ticker / f"residual__var_he__s1__frac{pct_}"
                             / f"seed_{seed}" / family / "complete.json")
                        if f.exists():
                            c = json.load(open(f))
                            rows.append({**base, "level": f"{pct_}%", "ratio": c["evaluations"]["val"]["he_std"] / ref})
    return pd.DataFrame(rows)


def figure_data(spec, gen, figs):
    d = e4_rows(spec)
    d.to_csv(gen / "e4_rows.csv", index=False)
    levels = ["25%", "50%", "100%", "pooled"]
    dev = d[d.ticker.isin(DEV)].groupby(["liability", "arm", "level", "ticker"]).ratio.mean().groupby(level=[0, 1, 2]).mean()
    allt = d[d.level.isin(["100%", "pooled"])].groupby(["liability", "arm", "level", "ticker"]).ratio.mean().groupby(level=[0, 1, 2]).mean()
    dev.rename("ratio").reset_index().to_csv(gen / "e4_dev_tickers.csv", index=False)
    allt.rename("ratio").reset_index().to_csv(gen / "e4_all_tickers.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1), sharey=True)
    marker = {"dh": "o", "ph": "s", "phd": "^"}
    for ax, (liab, title) in zip(axes, LIABS):
        style(ax)
        for arm, _ in E4_FAMS:
            y = [dev.get((liab, arm, lv), np.nan) for lv in levels]
            ax.plot(range(len(levels)), y, marker[arm] + "-", color=COLOR[arm], linewidth=1.4, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5, label=LABEL[arm])
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(["25%", "50%", "all", "ten\ntickers"], fontsize=6.5)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.axhline(1.0, color=INK, linewidth=0.8)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("Training episodes used", fontsize=7, color=INK)
    axes[0].set_ylabel("Std of hedging error\n(delta hedge = 1)", fontsize=7, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figs / "fig_data.pdf")
    fig.savefig(figs / "fig_data.png", dpi=200)
    plt.close(fig)
    return dev, allt


# --------------------------------------------------------------------------
# Prototype explanations (from rev_explain.py outputs)
# --------------------------------------------------------------------------

def explanation_tables(spec, split, gen):
    """Panel-wide explanation diagnostics and the worked example (SPY, European call, seed 0)."""
    rows = []
    for liab, _ in LIABS:
        for ticker in spec["tickers"]:
            for seed in range(5):
                a = {"ticker": ticker, "liability": liab, "mode": "residual", "objective": "var_he", "state": "s1",
                     "seed": seed}
                for arm, family in E4_FAMS[1:]:
                    f = phase3_fit.cell_dir(spec["study"], a) / family / f"explanation_{split}.json"
                    if f.exists():
                        r = json.load(open(f))
                        rows.append({"arm": arm, **r["cell"], **r["fidelity"]})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d.to_csv(gen / "explanations_by_fit.csv", index=False)
    m = d.groupby("arm").mean(numeric_only=True)
    spec_rows = [("Weight on the heaviest prototype", "top1_weight", "{:.2f}"),
                 ("Weight on the three heaviest", "top3_weight", "{:.2f}"),
                 ("Effective prototypes per decision", "effective_prototypes_mean", "{:.1f}"),
                 ("Three-prototype reconstruction error", "top3_mean_abs_error", "{:.3f}"),
                 ("\\quad relative to the std of the output", None, "{:.2f}"),
                 ("Heaviest prototype is the nearest state", "top1_is_nearest_share", "{:.0%}"),
                 ("\\quad is among the three nearest", None, None),
                 ("Median distance rank of the heaviest", "top1_median_distance_rank", "{:.1f}")]
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Diagnostic & Encoder & Plain \\", r"\midrule"]
    for label, key, fmt in spec_rows:
        if key is None and "relative" in label:
            vals = [fmt.format(m.loc[a, "top3_mean_abs_error"] / m.loc[a, "output_std"]) for a in ("ph", "phd")]
        elif key is None:
            continue
        else:
            vals = [fmt.format(m.loc[a, key]).replace("%", r"\%") for a in ("ph", "phd")]
        lines.append(f"{label} & {vals[0]} & {vals[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tab_explain.tex").write_text("\n".join(lines) + "\n")

    # worked example
    a = {"ticker": "SPY", "liability": "european_call", "mode": "residual", "objective": "var_he", "state": "s1", "seed": 0}
    out = []
    for arm, family in E4_FAMS[1:]:
        f = phase3_fit.cell_dir(spec["study"], a) / family / f"explanation_{split}.json"
        if not f.exists():
            continue
        ex = json.load(open(f))["example"]
        out.append((arm, ex))
    if out:
        lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
                 r" & Weight & Spot & Days left & Impl. vol. & \multicolumn{2}{c}{Correction} \\",
                 r" & & $S/S_0$ & & & shares & calls \\", r"\midrule"]
        sg = lambda x: (f"$+${x:.3f}" if x >= 0 else f"$-${abs(x):.3f}")
        for arm, ex in out:
            st = ex["state"]
            lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{LABEL[arm]}}}: step 10 of the episode starting {ex['episode_start_date']}}} \\")
            lines.append(f"Current state & & {st['spot']:.3f} & {st['time_left'] * 252:.0f} & {st['ivol']:.3f} & "
                         f"{sg(ex['output'][0])} & {sg(ex['output'][1])} \\\\")
            for r in ex["top"]:
                ps = r["state"]
                lines.append(f"Prototype: {r['source_start_date']}, step {r['source_step']} & {r['weight']:.2f} & "
                             f"{ps['spot']:.3f} & {ps['time_left'] * 252:.0f} & {ps['ivol']:.3f} & "
                             f"{sg(r['output_spot'])} & {sg(r['output_call'])} \\\\")
            lines.append(f"Three-prototype reconstruction & & & & & {sg(ex['top_reconstruction'][0])} & "
                         f"{sg(ex['top_reconstruction'][1])} \\\\")
            lines.append(r"\addlinespace")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (gen / "tab_example.tex").write_text("\n".join(lines) + "\n")
        rc.write_json(gen / "worked_example.json", {arm: ex for arm, ex in out})
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--rehearsal", action="store_true")
    args = p.parse_args()
    spec = json.load(open(args.spec))
    suffix = "_rehearsal_val" if args.rehearsal else ""
    gen, figs = PAPER / f"generated{suffix}", PAPER / f"figures{suffix}"
    gen.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    out, rows, panel, by_ticker, contrasts = load(spec, args.rehearsal)
    table_main(panel, rows, gen)
    table_contrasts(contrasts, spec, gen)
    table_arms(panel, rows, spec, gen)
    figure_tickers(rows, spec, figs)
    stats = figure_convergence(spec, figs)
    rc.write_json(gen / "convergence_stats.json", {f"{k[0]}__{k[1]}": v for k, v in stats.items()})
    if (phase3_fit.ROOT / "E4_mechanism").exists():
        dev, allt = figure_data(spec, gen, figs)
        print(dev.unstack().round(3).to_string())
        print(allt.unstack().round(3).to_string())
    m = explanation_tables(spec, "val" if args.rehearsal else "test", gen)
    if m is not None:
        print(m.T.round(3).to_string())
    print("wrote", gen, "and", figs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tables and figures for the IDA 2027 manuscript (Springer LNCS), generated from stored results.

Reads the main test evaluation (`phase3/final_results`, the run tree's `final/rows_by_ticker_seed.csv`),
the explanation tests (`phase3/extensions`, the run tree's `extensions_X/events`), and the simulated
reproduction rows. For Figure 2 it re-runs the frozen SPY arm-A fits on the validation episodes
(2019-2021); nothing is trained and the test episodes are not opened. Nothing is selected or filtered:
every ticker, seed and fit of each arm enters the averages.

    python scripts/revision/rev_ida_assets.py

Writes LaTeX fragments to `paper/revision_2026_11/manuscript_ida/generated/` and figures to
`paper/revision_2026_11/manuscript_ida/figures/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import phase3_fit  # noqa: E402
import rev_events as ev  # noqa: E402
import rev_faithful as rf  # noqa: E402

REV = rc.REPO / "paper" / "revision_2026_11"
FINAL = REV / "phase3" / "final_results"
EXT = REV / "phase3" / "extensions"
RUNS = phase3_fit.ROOT / "M1_main"
OUT = REV / "manuscript_ida"
GEN, FIGS = OUT / "generated", OUT / "figures"
LIABS = [("european_call", "European call"), ("asian_call", "Asian call")]
TICKERS = ["AAPL", "IWM", "NVDA", "QQQ", "SPY", "TLT", "XLE", "XLF", "XLK", "XLV"]
COLOR = {"dh": "#2a78d6", "ph": "#eb6834", "phd": "#1baf7a", "delta_vega": "#6b7280"}
INK, MUTED, GRID, SHADE = "#1f2933", "#6b7280", "#e5e7eb", "#eef0f3"
LABEL = {"delta_vega": "Delta-vega hedge", "dh": "DH", "ph": "PH, released encoder", "phd": "PH, plain distance"}
TEXTWIDTH_IN = 4.80                                    # LNCS text width, 12.2 cm
# The original paper's Table 2 (Faloughi, Guo and Luk 2025): gains over unhedged on CVaR@50%.
PAPER_SIM = {"bs": (0.04570, 0.04547, -0.00023, -0.0040), "sv": (0.03440, 0.03410, -0.00030, -0.0033)}


def num(x, d=5):
    return f"{x:+.{d}f}".replace("-", "$-$")


def pct(x, d=0, sign=False):
    s = f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"
    return s.replace("-", "$-$") + r"\%"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7, length=2.5, width=0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Table 1: the original metric, simulation and the three real-data setups
# --------------------------------------------------------------------------

def table_original_metric(panel, contrasts, posthoc):
    rep = pd.read_csv(REV / "phase1_reproduction" / "five_replicate_tf" / "reproduction_rows.csv")
    rep = rep[(rep.arm == "ph") & (rep.rep == "paper") & (rep.set == "paper_test")].set_index("env")
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}lrrlrrr@{}}", r"\toprule",
             r"Setting & DH gain & PH gain & PH $-$ DH [95\% interval] & $|U_{\mathrm{DH}}|$ & Rel. & Removed \\",
             r" & & & & & & DH/PH, \% \\", r"\midrule",
             r"\multicolumn{7}{@{}l}{\emph{Simulated markets (authors' code and 5,000-path test set)}} \\"]
    for env, name in (("bs", "BS"), ("sv", "SV")):
        d, p, g, r = PAPER_SIM[env]
        lines.append(rf"\quad {name}, paper & {num(d)} & {num(p)} & {num(g)} & -- & {pct(r, 2, True)} & -- \\")
        x = rep.loc[env]
        lines.append(rf"\quad {name}, rerun & {num(x.dh_vs_unhedged)} & {num(x.ph_vs_unhedged)} & "
                     rf"{num(x.gap)} & {abs(x.dh):.5f} & {pct(x.rel_gap, 2, True)} & -- \\")
    # (label, DH arm, PH arm, source of the interval): "pre" = pre-declared contrast of the main
    # evaluation, "post" = computed afterwards (extension X2b), "sec" = pre-declared secondary contrast
    setups = [("Original settings", "dh_orig", "ph_orig", "pre"),
              ("Equal tuning", "dh_cvar50", "ph_cvar50", "pre"),
              ("Main, encoder", "dh", "ph", "post"),
              ("Main, plain", "dh", "phd", "pre")]
    for liab, title in LIABS:
        u, dl = panel.loc[(liab, "unhedged")], panel.loc[(liab, "liability_delta")]
        lines += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{Historical data, {title} "
                               rf"(delta hedge: gain {num(dl.offset_cvar50 - u.offset_cvar50)}, "
                               rf"{pct(1 - dl.he_std / u.he_std)} removed)}}}} \\"]
        for name, a_dh, a_ph, src in setups:
            d, p = panel.loc[(liab, a_dh)], panel.loc[(liab, a_ph)]
            if src == "post":
                c = posthoc[(posthoc.liability == liab) & (posthoc.candidate == "ph") & (posthoc.benchmark == "dh")
                            & (posthoc.metric == "offset_cvar50")].iloc[0]
            else:
                c = contrasts[(contrasts.liability == liab) & (contrasts.block == 20) & (contrasts.candidate == a_ph)
                              & (contrasts.benchmark == a_dh) & (contrasts.metric == "offset_cvar50")].iloc[0]
            mark = r"$^{\dagger}$" if src == "post" else ""
            removed = " / ".join(f"{100 * (1 - r.he_std / u.he_std):.0f}".replace("-", "$-$") for r in (d, p))
            lines.append(rf"\quad {name} & {num(d.offset_cvar50 - u.offset_cvar50)} & {num(p.offset_cvar50 - u.offset_cvar50)} & "
                         rf"{num(c.difference)} [{num(c.ci95_low)}, {num(c.ci95_high)}]{mark} & "
                         rf"{abs(d.offset_cvar50):.5f} & "
                         rf"{pct(c.difference / abs(d.offset_cvar50), 1, True)} & {removed} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_original_metric.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Table 2: hedging quality in the main setup
# --------------------------------------------------------------------------

def table_hedging(panel):
    arms = [("unhedged", "Unhedged"), ("liability_delta", "Delta hedge"), ("delta_vega", "Delta--vega hedge"),
            ("dh", "DH"), ("ph", "PH, released encoder"), ("phd", "PH, plain distance")]
    lines = [r"\begin{tabular}{@{}lrrrrcrrrr@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{European call} && \multicolumn{4}{c}{Asian call} \\",
             r"\cmidrule{2-5}\cmidrule{7-10}",
             r"Strategy & Removed & Std $E$ & CVaR$_{5}$ & Cost && Removed & Std $E$ & CVaR$_{5}$ & Cost \\",
             r"\midrule"]
    for arm, name in arms:
        if arm == "dh":
            lines.append(r"\midrule")
        cells = []
        for liab, _ in LIABS:
            r, u = panel.loc[(liab, arm)], panel.loc[(liab, "unhedged")]
            cells.append(" & ".join([pct(1 - r.he_std / u.he_std, 1), f"{100 * r.he_std:.2f}",
                                     f"{100 * r.he_cvar05:.2f}".replace("-", "$-$"), f"{100 * r.cost_mean:.2f}"]))
        lines.append(f"{name} & {cells[0]} && {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_hedging.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Table 3: the explanation audit
# --------------------------------------------------------------------------

def crisis_recall_levels():
    """Weight on stress-window prototypes, stressed vs other test decisions (equal-ticker, seed means)."""
    rows = []
    for liab, _ in LIABS:
        for layer in ("ph", "phd"):
            per_ticker = []
            for t in TICKERS:
                z = np.load(RUNS / "extensions_X" / "events" / liab / t / "arrays.npz")
                vals = []
                for s in range(5):
                    w = z[f"{layer}__{s}__s_named"]
                    if w.max() == 0:
                        continue
                    vals.append((w[z["stressed_iv"]].mean(), w[~z["stressed_iv"]].mean()))
                if vals:
                    per_ticker.append(np.mean(vals, axis=0))
            m = np.mean(per_ticker, axis=0)
            rows.append({"liability": liab, "layer": layer, "stressed": m[0], "other": m[1],
                         "tickers": len(per_ticker)})
    df = pd.DataFrame(rows)
    df.to_csv(EXT / "x3_recall_levels.csv", index=False)
    return df.set_index(["liability", "layer"])


def table_audit():
    x1 = pd.read_csv(EXT / "x1_summary_test.csv").set_index(["liability", "layer", "measure"]).panel_mean
    x3a = pd.read_csv(EXT / "x3a_representation.csv").set_index(["liability", "layer"])
    sb = pd.read_csv(EXT / "x2b_scoreboard_test.csv").set_index(["liability", "arm"])
    v1 = json.load(open(EXT / "x_verdict_test.json"))["verdict"]
    v3 = json.load(open(EXT / "x3_verdict.json"))["verdict"]
    rec = crisis_recall_levels()
    mark = lambda ok: r" \textsc{p}" if ok else r" \textsc{f}"

    def cells(layer, fn, verdict):
        return " & ".join(fn(liab, layer) + mark(verdict(liab, layer)) for liab, _ in LIABS)

    g = lambda liab, layer, m: float(x1.loc[(liab, layer, m)])
    V1 = lambda test: (lambda liab, layer: v1[f"{liab}/{layer}"][test])
    V3 = lambda test: (lambda liab, layer: v3[f"{liab}/{layer}"][test])
    # X1d has two parts (protocol: local removal, and whole-hedge deletion); the stored verdict
    # requires both, so each part's own pass mark is re-applied here from the stored measures.
    local_ok = lambda l, y: (g(l, y, "x1d_drop_heaviest3_change") >= 3 * g(l, y, "x1d_drop_lightest3_change")
                             and g(l, y, "x1d_drop_lightest3_change") <= 0.05)
    deletion_ok = lambda l, y: (g(l, y, "x1d_usage_harm_spearman") >= 0.5 and g(l, y, "x1d_most_used_hurts_more") >= 0.8)
    for l, _ in LIABS:                           # the split must reproduce the stored combined verdict
        for y in ("ph", "phd"):
            assert v1[f"{l}/{y}"]["X1d"] == (local_ok(l, y) and deletion_ok(l, y))
    known = r"$^{\ddagger}$"
    groups = [
        ("Influence", [
            ("Stability", "Heaviest prototype unchanged under 0.1 sd noise", r"$\ge$90\%",
             lambda l, y: pct(g(l, y, "x1e_top1_unchanged")), V1("X1e")),
            ("Local removal", "Output change, 3 heaviest / 3 lightest removed (sd)",
             r"$\ge$3$\times$, $\le$0.05",
             lambda l, y: f"{g(l, y, 'x1d_drop_heaviest3_change'):.2f}/{g(l, y, 'x1d_drop_lightest3_change'):.2f}",
             local_ok),
            ("Deletion", "Corr.\\ of use with std$(E)$ rise on deletion",
             r"$\ge$0.5", lambda l, y: f"{g(l, y, 'x1d_usage_harm_spearman'):.2f}", deletion_ok)]),
        ("Visible similarity", [
            ("Nearest" + known, "Heaviest is nearest of 10 (chance 10\\%)",
             r"$\ge$50\%", lambda l, y: pct(g(l, y, "x1b_top1_is_nearest")), V1("X1b"))]),
        ("Crisis correspondence", [
            ("Represent.", "Prototypes in stress windows (9.4\\% of days)", r"$\ge$14\%",
             lambda l, y: pct(float(x3a.loc[(l, y), "prototype_share_named"])), V3("X3a")),
            ("Recall", "Their weight, calm $\\to$ stressed test days", "rise",
             lambda l, y: f"{100 * rec.loc[(l, y), 'other']:.0f}$\\to${100 * rec.loc[(l, y), 'stressed']:.0f}\\%", V3("X3b"))]),
        ("Reading and use", [
            ("Sufficiency" + known, "std$(E)$ change, top 3 prototypes only", r"$\le$5\%",
             lambda l, y: pct(g(l, y, "x1a_top3_rollout_rel_change")), V1("X1a")),
            ("Self-consist.", "Output at own state vs attached output (sd)",
             r"$\le$10\%", lambda l, y: pct(g(l, y, "x1c_self_err_over_s")), V1("X1c")),
            ("Fallback", "std$(E)$ change, delta hedge far from bank", "$<$0",
             lambda l, y: pct(sb.loc[(l, f"{y}_g95"), "he_std"] / sb.loc[(l, y), "he_std"] - 1, 0, True),
             V1("X2b_fallback_helps"))]),
    ]
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}p{3.45cm}"
             r">{\raggedright\arraybackslash}p{0.8cm}cccc@{}}",
             r"\toprule",
             r" & & & \multicolumn{2}{c}{Released encoder} & \multicolumn{2}{c}{Plain distance} \\",
             r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Test & Measure & Pass mark & European & Asian & European & Asian \\"]
    for head, rows in groups:
        lines += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{head}}}}} \\"]
        for name, q, passmark, fn, verdict in rows:
            lines.append(f"{name} & {q} & {passmark} & {cells('ph', fn, verdict)} & {cells('phd', fn, verdict)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_audit.tex").write_text("\n".join(lines) + "\n")
    facts = {"untrained_encoder_top1_is_nearest": {l: g(l, "ph", "x1b_untrained_top1_is_nearest") for l, _ in LIABS},
             "crisis_recall": rec.reset_index().to_dict(orient="records"),
             "plain_usage_harm_spearman": {l: g(l, "phd", "x1d_usage_harm_spearman") for l, _ in LIABS}}
    (GEN / "audit_facts.json").write_text(json.dumps(facts, indent=1, default=float))


# --------------------------------------------------------------------------
# Figure 1: spread by ticker; Figure 2: the March 2020 case
# --------------------------------------------------------------------------

def figure_summary(panel):
    """Figure 1: the two headline results. Left, remaining spread std(E) by strategy; right, how often the
    heaviest prototype is the nearest one in the standardised state. European = circle, Asian = triangle."""
    x1 = pd.read_csv(EXT / "x1_summary_test.csv").set_index(["liability", "layer", "measure"]).panel_mean
    shape = {"european_call": "o", "asian_call": "^"}
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 1.95), gridspec_kw={"width_ratios": [1.2, 1]})
    left = [("liability_delta", "Delta hedge", MUTED), ("delta_vega", "Delta–vega hedge", COLOR["delta_vega"]),
            ("phd", "PH, plain distance", COLOR["phd"]), ("ph", "PH, released encoder", COLOR["ph"]),
            ("dh", "DH", COLOR["dh"])]
    style(axl)
    axl.grid(axis="y", visible=False)
    axl.grid(axis="x", color=GRID, linewidth=0.6)
    for i, (arm, name, col) in enumerate(left):
        for liab, _ in LIABS:
            v = 100 * panel.loc[(liab, arm)].he_std
            hollow = arm in ("liability_delta", "delta_vega")
            axl.plot([v], [i], shape[liab], color=col, markersize=4.6, markeredgewidth=0.9,
                     markerfacecolor="white" if hollow else col, markeredgecolor=col)
    axl.set_yticks(range(len(left)))
    axl.set_yticklabels([n for _, n, _ in left], fontsize=6.8)
    axl.set_xlim(0.15, 0.75)
    axl.set_ylim(-0.7, len(left) - 0.4)
    axl.set_xlabel("Std of hedging error, % of $S_0$ (lower is better)", fontsize=6.8, color=INK)
    axl.set_title("(a) Spread left after hedging", fontsize=7.2, color=INK, loc="left")
    right = [("ph", "untrained", "Encoder, untrained", COLOR["ph"], True),
             ("phd", None, "PH, plain distance", COLOR["phd"], False),
             ("ph", None, "PH, released encoder", COLOR["ph"], False)]
    style(axr)
    axr.grid(axis="y", visible=False)
    axr.grid(axis="x", color=GRID, linewidth=0.6)
    axr.axvline(10, color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)))
    axr.text(11.5, -0.62, "chance", fontsize=5.6, color=MUTED, va="bottom")
    for i, (layer, ref, name, col, hollow) in enumerate(right):
        for liab, _ in LIABS:
            m = "x1b_untrained_top1_is_nearest" if ref else "x1b_top1_is_nearest"
            v = 100 * float(x1.loc[(liab, layer, m)])
            axr.plot([v], [i], shape[liab], color=col, markersize=4.6, markeredgewidth=0.9,
                     markerfacecolor="white" if hollow else col, markeredgecolor=col)
            axr.text(v, i + (0.33 if liab == "european_call" else -0.36), f"{v:.0f}", fontsize=5.4, color=INK,
                     ha="center", va="center")
    axr.set_yticks(range(len(right)))
    axr.set_yticklabels([n for *_, n, _, _ in [(r[0], r[1], r[2], r[3], r[4]) for r in right]], fontsize=6.8)
    axr.set_xlim(0, 100)
    axr.set_ylim(-0.7, len(right) - 0.4)
    axr.set_xlabel("Share of decisions (%)", fontsize=6.8, color=INK)
    axr.set_title("(b) Cited prototype is the nearest", fontsize=7.2, color=INK, loc="right")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=INK, markersize=4, label="European call"),
               Line2D([], [], marker="^", linestyle="", color=INK, markersize=4, label="Asian call")]
    fig.legend(handles=handles, fontsize=6.5, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    fig.savefig(FIGS / "fig_summary.pdf")
    fig.savefig(FIGS / "fig_summary.png", dpi=220)
    plt.close(fig)


def figure_tickers(rows):
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 2.45), sharex=True, sharey=True)
    offsets = {"delta_vega": -0.27, "dh": -0.09, "ph": 0.09, "phd": 0.27}
    marker = {"delta_vega": "D", "dh": "o", "ph": "s", "phd": "^"}
    for ax, (liab, title) in zip(axes, LIABS):
        d = rows[rows.liability == liab]
        ref = d[d.arm == "liability_delta"].set_index("ticker").he_std
        style(ax)
        ax.axhline(1.0, color=INK, linewidth=0.8)
        for arm in ("delta_vega", "dh", "ph", "phd"):
            gdf = d[d.arm == arm]
            for i, t in enumerate(TICKERS):
                v = gdf[gdf.ticker == t].he_std.to_numpy() / ref[t]
                x = i + offsets[arm]
                if len(v) > 1:
                    ax.plot([x, x], [v.min(), v.max()], color=COLOR[arm], linewidth=1.0, alpha=0.6)
                ax.plot([x], [v.mean()], marker[arm], color=COLOR[arm], markersize=4.2, markeredgecolor="white",
                        markeredgewidth=0.5, label=LABEL[arm] if i == 0 else None)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2])
        ax.set_yticklabels(["0.25", "0.5", "1", "2"])
        ax.minorticks_off()
        ax.set_ylim(0.18, 2.6)
        ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=8, color=INK, ha="left", va="bottom")
        ax.text(len(TICKERS) - 0.55, 1.04, "delta hedge", fontsize=6.5, color=INK, ha="right", va="bottom")
    axes[1].set_xticks(range(len(TICKERS)))
    axes[1].set_xticklabels(TICKERS, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color=MUTED, linewidth=1.0))
    labels.append("min–max over 5 seeds")
    fig.legend(handles, labels, fontsize=6.4, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=0.9)
    fig.supylabel("Spread relative to delta hedge", fontsize=7, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.6)
    fig.savefig(FIGS / "fig_tickers.pdf")
    fig.savefig(FIGS / "fig_tickers.png", dpi=220)
    plt.close(fig)


def covid_series():
    """Daily weight on 2008-09 crisis prototypes for SPY (European), validation 2019-2021, every seed and layer."""
    tr = rc.Episodes("SPY", "european_call", "train")
    va = rc.Episodes("SPY", "european_call", "val")
    sc = rc.Scaler(tr, "s1")
    cp = pd.read_csv(rc.PANEL / "contract_paths" / "SPY_contract_paths.csv", usecols=["episode_index", "step", "date"])
    date_of = {(int(a), int(b)): d for a, b, d in cp.itertuples(index=False)}
    dd = ev.decision_dates(cp, va.index).ravel()
    ivol = pd.Series(va.features["ivol"].ravel(), index=dd).groupby(level=0).median()
    series = {}
    for seed in range(5):
        a = {"ticker": "SPY", "liability": "european_call", "mode": "residual", "objective": "var_he", "state": "s1",
             "seed": seed}
        cell = phase3_fit.cell_dir("M1_main", a)
        for layer, fam in rf.LAYERS.items():
            pol, _ = rc.load_policy(cell, fam, sc)
            bank = np.load(cell / fam / "prototypes.npz", allow_pickle=False)
            pdates = np.array([date_of[(int(x), int(y))] for x, y in zip(bank["source_episode_index"], bank["source_step"])])
            crisis = (pdates >= "2008-09-15") & (pdates <= "2009-03-31")
            W = rc.evaluate_policy(pol, va, "s1", "residual")["weights"]
            s = W[..., crisis].sum(axis=2).ravel()
            series[(layer, seed)] = pd.Series(s, index=dd).groupby(level=0).mean()
    df = pd.DataFrame({f"{layer}_{seed}": v for (layer, seed), v in series.items()})
    df["ivol_median"] = ivol
    df.index = pd.to_datetime(df.index)
    df.sort_index().to_csv(EXT / "x3_covid_daily_SPY.csv")
    return df.sort_index()


def figure_covid(df):
    """Full validation series (left) and a zoom on Feb-Apr 2020 (right). Both layers get the same treatment:
    one thin line per model (5 seeds each) and a bold mean."""
    fig, axes = plt.subplots(3, 2, figsize=(TEXTWIDTH_IN, 2.55), sharey="row",
                             gridspec_kw={"height_ratios": [0.8, 1, 1], "width_ratios": [2.3, 1]})
    lo, hi = pd.Timestamp("2020-02-24"), pd.Timestamp("2020-03-31")
    zlo, zhi = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")
    for ax in axes.ravel():
        style(ax)
        ax.axvspan(lo, hi, color=SHADE, zorder=0)
    for col, (a, b) in enumerate([(df.index.min(), df.index.max()), (zlo, zhi)]):
        d = df.loc[a:b]
        axes[0, col].plot(d.index, d["ivol_median"], color=INK, linewidth=0.9)
        for row, layer, name in ((1, "phd", "PH, plain distance"), (2, "ph", "PH, released encoder")):
            ax = axes[row, col]
            cols = [c for c in df.columns if c.startswith(layer + "_")]
            for c in cols:
                ax.plot(d.index, d[c], color=COLOR[layer], linewidth=0.55, alpha=0.55)
            ax.plot(d.index, d[cols].mean(axis=1), color=COLOR[layer], linewidth=1.4)
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0", "50%", "100%"])
            if col == 0:
                ax.set_title(f"{name}: 5 models (thin) and their mean (bold)", fontsize=6.6, color=INK,
                             loc="left", pad=2)
        for ax in axes[:, col]:
            ax.set_xlim(a, b)
    axes[0, 0].set_ylabel("Implied vol.", fontsize=6.8, color=INK)
    axes[1, 0].set_ylabel("Weight on\n2008–09 days", fontsize=6.8, color=INK)
    axes[2, 0].set_ylabel("Weight on\n2008–09 days", fontsize=6.8, color=INK)
    axes[0, 1].set_title("Zoom: Feb–Apr 2020", fontsize=6.8, color=INK, loc="left")
    axes[0, 0].set_title("SPY validation period, 2019–2021 (crash shaded)", fontsize=6.8, color=INK, loc="left")
    for ax in axes[:2, :].ravel():
        ax.tick_params(labelbottom=False)
    axes[2, 0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2, 1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for ax in axes[2]:
        for lab in ax.get_xticklabels():
            lab.set_fontsize(6.5)
    for ax in axes.ravel():
        ax.tick_params(labelsize=6.5)
    fig.tight_layout(h_pad=0.3, w_pad=0.6)
    fig.savefig(FIGS / "fig_covid.pdf")
    fig.savefig(FIGS / "fig_covid.png", dpi=220)
    plt.close(fig)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(FINAL / "scoreboard_panel.csv").set_index(["liability", "arm"])
    contrasts = pd.read_csv(FINAL / "contrasts.csv")
    posthoc = pd.read_csv(EXT / "x2b_contrasts_test.csv")
    rows = pd.read_csv(RUNS / "final" / "rows_by_ticker_seed.csv")
    table_original_metric(panel, contrasts, posthoc)
    table_hedging(panel)
    table_audit()
    figure_summary(panel)
    figure_tickers(rows)
    cached = EXT / "x3_covid_daily_SPY.csv"          # written by covid_series(); pass --recompute to re-run the fits
    if cached.exists() and "--recompute" not in sys.argv:
        df = pd.read_csv(cached, index_col=0, parse_dates=True).sort_index()
    else:
        df = covid_series()
    figure_covid(df)
    print("wrote", sorted(p.name for p in GEN.iterdir()), sorted(p.name for p in FIGS.iterdir()))


if __name__ == "__main__":
    main()

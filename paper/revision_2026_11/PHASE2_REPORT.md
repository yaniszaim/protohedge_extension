# Phase 2 report: what the completed historical results actually show

Date: 2026-11-13. No model was trained for this report. Every number comes
from re-evaluating saved models and rule-based hedges on the same episodes.

- Code: `scripts/revision/phase2_diagnostics.py`, `phase2_split_check.py`, `phase2_figures.py`
- Raw outputs: `.deephedging_real_runs/revision_v1/phase2_diagnostics/`
- Tables and figures: `paper/revision_2026_11/phase2/`
- The v4 run tree and the checkpoint-sensitivity tree were read only.

## 1. Is the re-evaluation trustworthy?

| Check | Result |
|---|---|
| 186 re-evaluated model/seed test rows against the saved `sweep_metrics.csv` | largest difference 7e-6 (float precision) |
| The submitted table's four ProtoHedge minus Deep Hedging utility differences, rebuilt from saved weights | European +0.001009 and +0.000514, Asian +0.001859 and +0.004015: identical to the manuscript |
| Liability pricing (Black-Scholes; Levy approximation for the Asian) against 400,000-path Monte Carlo | prices agree to 0.00004; Asian delta agrees with a finite difference |
| Premium-inclusive mean hedging error of the liability delta hedge | +0.0007 (European), +0.0002 (Asian): the model premium is close to fair |

"As submitted" below means the manuscript's own provenance: the v4 fits, with
the 36 fits that were retrained under unpenalised validation checkpointing
replacing their v4 versions (33 European, 3 Asian).

## 2. Scoreboard (test episodes 2022-2024, ten tickers equal weight, seeds averaged)

All values are fractions of the initial spot. "CVaR@50%" is the paper's
training objective on the premium-excluded offset (higher is better). Hedging
error adds the liability's model premium at inception.

**European call**

| Strategy | CVaR@50% (objective) | Std of hedging error | CVaR@5% of hedging error | Mean hedging error |
|---|---|---|---|---|
| Unhedged | -0.0617 | 0.0399 | -0.1075 | -0.0042 |
| Spot-Delta, listed call's delta (the paper's baseline) | -0.0406 | 0.0129 | -0.0417 | -0.0027 |
| **Liability delta hedge** | **-0.0341** | **0.0066** | **-0.0148** | +0.0007 |
| Deep Hedging, as submitted | -0.0439 | 0.0328 | -0.0821 | +0.0060 |
| ProtoHedge, validation-mean, as submitted | -0.0405 | 0.0272 | -0.0655 | +0.0058 |
| ProtoHedge, validation-tail, as submitted | -0.0399 | 0.0232 | -0.0556 | +0.0040 |
| ProtoHedge, fixed original-style (K=100) | -0.0439 | 0.0286 | -0.0742 | +0.0030 |
| Buy one share and hold | -0.0437 | 0.0354 | -0.0961 | +0.0055 |
| Best constant position (chosen on training data, CVaR@50%) | -0.0414 | 0.0276 | -0.0733 | +0.0048 |
| Lowest-variance constant position (training data) | -0.0404 | 0.0114 | -0.0351 | -0.0034 |

**Asian call**

| Strategy | CVaR@50% (objective) | Std of hedging error | CVaR@5% of hedging error | Mean hedging error |
|---|---|---|---|---|
| Unhedged | -0.0364 | 0.0238 | -0.0638 | -0.0020 |
| Spot-Delta, listed call's delta (the paper's baseline) | -0.0352 | 0.0229 | -0.0453 | -0.0006 |
| **Liability delta hedge** | **-0.0210** | **0.0044** | **-0.0113** | +0.0002 |
| Deep Hedging, as submitted | -0.0335 | 0.0244 | -0.0640 | +0.0003 |
| ProtoHedge, validation-mean, as submitted | -0.0315 | 0.0266 | -0.0690 | +0.0033 |
| ProtoHedge, validation-tail, as submitted | -0.0295 | 0.0222 | -0.0595 | +0.0022 |
| ProtoHedge, fixed original-style (K=100) | -0.0330 | 0.0274 | -0.0717 | +0.0023 |
| Buy one share and hold | -0.0477 | 0.0496 | -0.1082 | +0.0076 |
| Best constant position (chosen on training data, CVaR@50%) | -0.0305 | 0.0230 | -0.0524 | +0.0039 |
| Lowest-variance constant position (training data) | -0.0285 | 0.0146 | -0.0375 | -0.0010 |

Paired block-bootstrap intervals for every contrast are in
`phase2/contrasts_bootstrap.csv` (2,000 draws, 20-episode blocks within
ticker, the submitted paper's method).

## 3. Findings

**F1. A delta hedge of the liability itself beats every learned model on every
metric, in all ten tickers, including the models' own training objective.**
- CVaR@50%, European: Deep Hedging trails it by 0.0098 [0.0053, 0.0149];
  tail-selected ProtoHedge by 0.0058 [0.0041, 0.0078] (ahead in 1 of 10 tickers).
- Std of hedging error is 3.5 times (ProtoHedge) to 5 times (Deep Hedging)
  larger than the delta hedge for the European call, and 5 times larger for
  both on the Asian call. Per ticker the ProtoHedge ratio runs from 2.1 to 6.8.
- The submitted paper had no such baseline.

**F2. The paper's "Spot-Delta" baseline hedges the wrong option.** It uses
the vendor delta of the listed call (mean 0.52), not the delta of the
liability. For the Asian call the correct delta averages 0.28 and decays to
zero, so "Spot-Delta" barely reduces dispersion there (0.0229 against 0.0238
unhedged). The paper's Asian claim that ProtoHedge beats Spot-Delta is a
comparison with a mis-specified hedge.

**F3. Deep Hedging, as submitted, is mostly a static long position.**
- European: the spot position is at its +1 limit in 87% of path-steps; 19 of
  30 fits are either selected at initialisation or the first epoch (8) or do
  at least 90% of their trading at the first step (11).
- Its hedge delta is negatively correlated with the liability delta (-0.52).
- It is worse than the best constant position on its own objective
  (-0.0025 [-0.0038, -0.0013], 0 of 10 tickers) and indistinguishable from
  buying one share (-0.0002 [-0.0029, +0.0024]).
- Asian: no better than unhedged on dispersion or CVaR@5%.

**F4. The cause is identified: initial output scale plus hard clipping, not
only short training.**
- The Deep Hedging network outputs a trade per step. With raw, unstandardised
  inputs an untrained network outputs a nearly constant trade, which
  accumulates to a position limit within a few steps. Measured on SPY, 68% to
  92% of position entries are already at a limit at initialisation in five of
  six seed-by-liability cases, and the gradient norm there is 5 to 15 times
  smaller than in the sixth case.
- The same seed gives the same initial weights for every ticker, so the
  outcome follows the seed, not the data. Asian seed 2345 ends with spot at +1
  in all ten tickers.
- European seed 2345 is the exception that confirms it: its untrained network
  happens to hold about 0.53 shares and 0.35 calls, away from the limits, and
  the v4 checkpoint rule selected that **untrained initialisation** in 9 of
  10 tickers. It is an accidental static hedge. Those are the fits the
  manuscript retrained with unpenalised checkpointing; once trained, 8 of the
  12 retrained fits end at the limits (F8).
- ProtoHedge starts from small bounded prototype actions and smooth clipping,
  so it never starts at a limit. The comparison in the paper is confounded by
  this difference.

**F5. ProtoHedge trades dynamically but is not a hedge either.**
- Its mean spot position (0.44 to 0.48, European) is near the liability delta,
  but the correlation with the liability delta across path-steps is only 0.43
  (European) and 0.13 (Asian).
- On the training objective it is not distinguishable from the best constant
  position: +0.0016 [-0.0014, +0.0048] (European), +0.0010 [-0.0014, +0.0034] (Asian).
- It has lower dispersion than that constant position for the European call,
  but 2 times the dispersion of the lowest-variance constant position.

**F6. With one fixed original-style configuration, ProtoHedge ties Deep
Hedging.** The fixed K=100, Deep-Hedging-state, unweighted model (declared
before evaluation) against Deep Hedging as submitted: frozen-threshold utility
-0.0017 (European, better in 4 of 10 tickers) and -0.0002 (Asian, 6 of 10);
empirical CVaR@50% -0.0000 [-0.0053, +0.0058] and +0.0006 [-0.0021, +0.0033].
The submitted advantage comes from selecting among 16 configurations, as
Reviewer 3 suspected.

**F7. Why learning fails differs by liability** (`phase2/split_check_panel.csv`).

| Training episodes, CVaR@50% | Liability delta | Buy and hold | Deep Hedging | ProtoHedge (tail) |
|---|---|---|---|---|
| European | -0.0302 | -0.0313 | -0.0301 | -0.0320 |
| Asian | -0.0187 | -0.0344 | -0.0252 | -0.0215 |

- European: on 2005-2018 data (mean 20-day return +0.97%) the objective cannot
  tell hedging from being long. All four sit within 0.002. Out of sample the
  hedge holds up (-0.0341) and the long position does not (-0.0439). This is a
  task-design problem: CVaR@50% with the premium excluded is too close to
  risk-neutral to identify a hedge.
- Asian: the delta hedge is clearly better **in sample**, and neither network
  finds it. This is an optimisation failure.

**F8. Overfitting is visible in the 12 retrained Deep Hedging fits**, the only
ones with saved training curves. Validation loss bottoms out early (median
epoch 132; four fits by epoch 7) while training loss keeps falling, and 8 of
12 finish with every validation path-step at a position limit. The v4 runs
saved no curves, so convergence cannot be shown for the other 228 fits.

**F9. ProtoHedge decisions rest on few prototypes.** Effective number of
prototypes per decision is 3.5 to 6.4; the top prototype carries 52% to 65% of
the weight and the top three 81% to 89%. A top-three decomposition of a
historical trade is therefore feasible and faithful, which is what Reviewers 1
and 3 asked for.

**F10. The repository's empirical CVaR mishandles ties.**
`outcome_metrics.empirical_lower_tail_cvar` averages every value at or below
the quantile. When many outcomes tie at the quantile (a share held against a
short call ties on every in-the-money path) it returns the mean instead of the
tail mean. The submitted numbers are barely affected because learned policies
rarely tie exactly, but any rule-based baseline needs the exact form used here.

## 4. What this means for Phase 3

1. The comparison the paper needs is three-way: Deep Hedging, ProtoHedge, and
   the liability delta hedge, on dispersion and tail metrics as well as utility.
2. Deep Hedging must be re-parameterised before anything else (standardised
   inputs, small initial trades, smooth bounds). Gate: on training and
   validation data it must reach the liability delta hedge's objective value
   on the Asian call and its dispersion on both.
3. The objective must be able to identify a hedge. CVaR@50% stays for
   continuity with the original paper; a tail-averse objective on the
   premium-inclusive hedging error is added and chosen on training and
   validation data only.
4. With about 100 independent training paths per ticker, learned policies may
   not beat the delta hedge. A policy that learns a correction to the delta
   hedge is the standard remedy and is a fair option for both model families.
5. The honest headline may be that neither learned model beats delta hedging
   here, and that ProtoHedge is the more robust of the two under scarce data.
   That is still a paper, and a more defensible one.

## 5. Limits of this report

- The test period 2022-2024 has been examined many times; nothing here is
  confirmatory.
- The liability delta hedge uses the listed call's implied volatility, which
  the networks do not receive directly (they see the call price).
- The bootstrap resamples within ticker and ignores common shocks across
  tickers, so panel intervals are too narrow (problem P14).
- The Asian delta is a moment-matching approximation with zero rates.

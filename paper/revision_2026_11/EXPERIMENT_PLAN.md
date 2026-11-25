# Experiment plan, version 2 (after Phases 0-2)

Updated 2026-11-13. Target: IEEE IJCNN 2027, deadline 2027-01-31. Version 1
was the plan agreed on 2026-11-11; Phases 0-2 are done or running, and Phase 3
below is revised in the light of `PHASE2_REPORT.md`.

## Revised research question

The submitted paper asked whether ProtoHedge matches Deep Hedging on
historical data. Phase 2 shows that, as trained, neither hedges as well as a
delta hedge of the liability. The revised question:

> On historical option data with realistic sample sizes, how close can a
> learned hedging policy get to a delta hedge of the liability, does the
> prototype policy get as close as an unconstrained network when both are
> trained properly and tuned equally, and what does the prototype policy's
> explanation of a real trade look like?

Any outcome is reportable, including "neither learned model beats the delta
hedge" and "ProtoHedge is worse than Deep Hedging".

## Status of Phases 0-2

| Phase | Item | Status |
|---|---|---|
| 0 | Revision folders, progress log, reviewer matrix, baseline tests (51 run, 50 pass, 1 CUDA skip) and audit | done |
| 1A | Original paper reproduced with the authors' code (two independent sets of runs) | done; plain-distance arm still running in the sibling project |
| 1B | PyTorch parity on the simulated worlds | data identity passes; six 800-epoch chains running |
| 2 | Diagnostics on saved models, baselines, premium, scoreboard, bootstrap, split check | done |

## Phase 3. Fix the problems, one at a time

Rules for every step: new output directory under
`.deephedging_real_runs/revision_v1/`; protocol file written and hashed before
the run; full training curves saved; atomic writes and stage-level resume; no
selection on test data; 2022-2024 labelled as previously examined; every
restart reported, failures included.

### 3.0 Gate from Phase 1B (no new compute)
Proceed if the PyTorch Deep Hedging matches TensorFlow within 1% and the
median ProtoHedge gap is no worse than -1% (G2, G3). If not, find the port
difference first.

### 3.1 Task and parameterisation pilot (problems P1, P1a, P8, P4)
Development set fixed now: SPY, TLT, NVDA (index, bond, single stock);
training and validation episodes only.

Factors, all pre-declared, each applied identically to both model families:
1. **Policy parameterisation**: (a) as submitted; (b) standardised inputs,
   smooth bounds, small initial trades; (c) = (b) with the policy outputting a
   correction to the liability delta hedge.
2. **Objective**: CVaR@50% on the premium-excluded offset (original); CVaR@10%
   on the premium-inclusive hedging error.
3. **State**: original; plus liability moneyness, listed call moneyness and
   time to expiry, implied volatility.

Size: 3 tickers x 2 liabilities x 2 families x (3 x 2 x 2 = 12 cells) x 2
seeds = 288 fits at one fixed small configuration per family. Estimated 120
CPU-hours at September timings (10 min per Deep Hedging fit, 35 per
ProtoHedge fit); to be re-measured in a 12-fit pilot.

Output: protocol v2 fixes one parameterisation and state for the main study
and keeps both objectives.

### 3.2 Deep Hedging adequacy gate (P3, P4, P5, P1b, P21)
On the development tickers, validation data:
- **Gate A (optimisation):** on training data, Deep Hedging reaches at least
  the liability delta hedge's objective value for both liabilities.
- **Gate B (generalisation):** validation dispersion of hedging error within
  1.5 times the delta hedge's under the tail-averse objective.
- Checkpoint rule for both families: validation hedged utility only.
- If Gate B fails for both families under every parameterisation, that is the
  finding; the main study still runs and reports it.

### 3.3 Fair comparison (P6, P7, P19, P22)
- Primary: one fixed configuration per family, chosen before training (Deep
  Hedging width 20 depth 3; ProtoHedge original-style, Deep-Hedging-state
  prototypes, unweighted, K fixed from 3.1 validation).
- Tuned: four pre-declared configurations per family (ProtoHedge K in {10,
  25, 50, 100}; Deep Hedging four width and weight-decay settings), one
  configuration per family chosen on pooled validation for all tickers.
- Ten tickers x 2 liabilities x 2 objectives x 5 seeds. Rule-based baselines
  from Phase 2 in every table.
- Secondary, labelled: the expanded ProtoHedge family (Spot-Delta-sourced
  prototypes, fixed feature weights).
- Size: primary 400 fits, tuned adds up to 1,200. About 500 to 700 CPU-hours;
  the tuned arm is cut to the development tickers if compute is short.

### 3.4 Mechanism experiments (P9, P13)
1. Training-set size: learning curves at 25%, 50%, 100% of each ticker's
   training episodes, and one pooled ten-ticker model per family. Tests
   whether the prototype bottleneck helps when data are scarce.
2. Regimes: performance and prototype usage by realised-volatility tercile and
   by calendar year of the test period.

### 3.5 Interpretability (P10)
- Store each prototype's source ticker, date and episode at extraction.
- One worked historical trade, chosen by a rule fixed in advance (median-loss
  test episode of SPY, step 10): state, top three prototypes with their dates
  and attached actions, weights, the reconstructed action, the residual, and
  the position-limit adjustment.
- Panel-wide diagnostics: top-k reconstruction error for k = 1, 3, 5;
  effective number of prototypes; share of decisions altered by limits;
  prototype stability across seeds.

### 3.6 Inference (P14, P17)
- Calendar-aligned block bootstrap across tickers; block lengths 10, 20, 40.
- Exact tail mean for every CVaR.
- Pre-declared primary contrasts (ProtoHedge vs Deep Hedging, each vs the
  liability delta hedge; utility and dispersion) with a Holm adjustment.
- Optional, if 2025 data can be obtained: untouched confirmation period.

## Phase 4. Rewrite

Structure around the three-way comparison; episode diagram; metric-purpose
table; one main results table; convergence figure; worked decomposition;
accurate description of the original method and its reproduction; related
work additions; limitations. Coauthor draft by 2027-01-15.

## Timeline

| Dates | Work |
|---|---|
| Nov 13-16 | Phase 1B finishes; 3.0 gate; implement 3.1 options with unit tests |
| Nov 17-27 | 3.1 pilot and 3.2 gates; lock protocol v2 |
| Nov 28-Dec 18 | 3.3 main runs; 3.5 in parallel |
| Dec 19-Jan 5 | 3.4, 3.6, analysis |
| Jan 5-15 | Rewrite; coauthor draft |
| Jan 16-31 | Revisions, IJCNN formatting and page check, submission by the authors |

## Decisions needed from the user

1. Compute. The laptop ran at a load average above 200 on 2026-11-13 with
   three projects sharing it. Phase 3 needs roughly 700 to 900 CPU-hours. Is
   the cloud VM available, and with what spending limit?
2. Scope. The revised question and the three-way comparison change the
   paper's likely conclusion. Confirm before 3.1 is locked.
3. Data. Can 2025 OptionMetrics and CRSP data be pulled for an untouched
   confirmation period?

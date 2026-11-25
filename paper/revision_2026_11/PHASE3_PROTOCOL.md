# Phase 3 protocol

Living document: each stage is written before its runs start and is not
edited afterwards except to add results pointers. Code: `scripts/revision/
rev_core.py`, `phase3_fit.py`; tests: `test_rev_core.py` (9 pass).

Rules for all stages
- Training and validation episodes only. Test episodes (2022-2024) are locked
  in code (`rev_core.Episodes`) until the final evaluation, which runs once.
- Both model families get the same inputs, standardisation, output
  parameterisation, objective, optimiser, epoch budget and checkpoint rule.
- Checkpoint: best validation value of the hedged objective (threshold-free,
  exact tail mean); the untrained network is epoch -1.
- Optimiser as in v4: Adam 1e-3, batch 32, gradient value and norm clipping at
  1. No action or inventory penalties.
- ProtoHedge: the repository's layer (encoder, softmax of negative squared
  distance, bounded prototype outputs, exact soft clip); prototypes by k-means
  plus nearest real state on the states the same cell's Deep Hedging agent
  visits on the training episodes (the original recipe); source episode and
  date stored.
- Seeds are hashed from ticker, liability, cell and family, so no two fits
  share initial weights.
- Every fit is reported; failures are counted, not dropped.

## Stage E1: exploration on SPY (written 2026-11-13, before the runs)

Purpose: find out which output parameterisation and objective let a network
learn a hedge at all. Exploratory, one ticker, one seed; nothing from E1 is a
result of the paper.

Design (`phase3/E1_spec.json`): SPY; European and Asian; 3 parameterisations
(trade, target, residual) x 3 objectives (CVaR@50% of the premium-excluded
offset, i.e. the original; CVaR@10% of the hedging error; variance of the
hedging error); original state; K = 100; at most 400 epochs, patience 100.
18 cells, 36 fits.

Questions, with the reading fixed in advance
- Q1 (gate A, optimisation): under which parameterisations does Deep Hedging
  reach the liability delta hedge's objective value on the training episodes?
- Q2 (gate B, generalisation): under which is its validation hedging-error
  std within 1.5 times the liability delta hedge's?
- Q3: with the parameterisation repaired, does the original objective still
  produce a directional position (mean net delta, correlation of the hedge
  delta with the liability delta)?
- Q4: where does ProtoHedge stand on the same three questions?

What E1 decides: which parameterisations and objectives go into the
three-ticker pilot (stage E2). A parameterisation is dropped only if it fails
gate A for both liabilities under both hedging-error objectives.

### E1 results (2026-11-13; 36 fits, none failed; `phase3/E1/rows.csv`)

SPY, validation episodes, one seed. References: liability delta hedge std of
hedging error 0.0071 (European) and 0.0055 (Asian); delta-vega hedge 0.0022
and 0.0036; unhedged 0.0245 and 0.0164.

| Objective | Family | European: std of hedging error (trade / target / residual) | Asian (trade / target / residual) |
|---|---|---|---|
| Variance of hedging error | Deep Hedging | 0.0032 / 0.0032 / 0.0031 | 0.0032 / 0.0030 / 0.0028 |
| Variance of hedging error | ProtoHedge | 0.0074 / 0.0081 / 0.0048 | 0.0046 / 0.0048 / 0.0053 |
| CVaR@10% of hedging error | Deep Hedging | 0.0062 / 0.0064 / 0.0080 | 0.0058 / 0.0094 / 0.0064 |
| CVaR@10% of hedging error | ProtoHedge | 0.0105 / 0.0101 / 0.0060 | 0.0082 / 0.0065 / 0.0052 |
| CVaR@50% of offset (original) | Deep Hedging | 0.0229 / 0.0316 / 0.0097 | 0.0204 / 0.0227 / 0.0069 |
| CVaR@50% of offset (original) | ProtoHedge | 0.0248 / 0.0198 / 0.0074 | 0.0167 / 0.0236 / 0.0063 |

- Q1 (gate A): under both hedging-error objectives Deep Hedging reaches the
  liability delta hedge's training objective in every parameterisation
  (one marginal miss: Asian, trade, CVaR@10%). No parameterisation is dropped.
- Q2 (gate B): Deep Hedging passes in 11 of 12 hedging-error cells (miss:
  Asian, target, CVaR@10%); ProtoHedge passes in all 12.
- Q3: with the original objective and the trade or target parameterisation,
  both families return to directional positions (std about 3 to 4.5 times the
  delta hedge's; hedge-delta correlation with the liability delta 0.37 to 0.55, and
  -0.71 for Deep Hedging target). In the residual parameterisation Deep
  Hedging keeps its untrained start, i.e. the plain delta hedge: under the
  original objective nothing found on training data beats the delta hedge on
  validation.
- Q4: under the variance objective ProtoHedge trails Deep Hedging in all six
  cells (std 1.4 to 2.5 times Deep Hedging's) but matches or beats the
  liability delta hedge in four of six. Under CVaR@10% the two are mixed and
  the training-validation gap is large (ProtoHedge European target: -0.006 in
  training, -0.020 in validation), as expected for a 10% tail on about 100
  independent paths.
- Deep Hedging under the variance objective uses the listed call heavily
  (0.6 to 0.8 calls) and comes close to the delta-vega hedge.
- Interpretability note: in the trade parameterisation the heaviest prototype
  of a decision can be a very different market state (the worked example's
  top prototype is at step 18 with spot 0.93 for a state at step 10 with spot
  1.03). In the residual parameterisation the top prototypes resemble the
  state. A state-similarity diagnostic is added to stage 3.5.

## Stage E2: three-ticker pilot (written 2026-11-13, before the runs)

Purpose: fix protocol v2 for the main study, with more than one ticker and
seed. Training and validation only.

Design: SPY, TLT, NVDA; both liabilities; seeds 0 and 1; K = 100; at most 800
epochs, patience 150 (E1's cap of 400 was reached by one fit).
- E2a (`phase3/E2a_spec.json`): 3 parameterisations x 3 objectives, original
  state: 108 cells.
- E2b (`phase3/E2b_spec.json`): richer state (adds implied volatility, the
  listed call's log-moneyness and time to expiry), variance objective, 3
  parameterisations: 36 cells.

Decisions, with rules fixed now
- D1 (state): the richer state replaces the original one in the main study
  only if it improves the mean validation objective of both families over
  the 36 matched cells; otherwise the original state is kept.
- D2 (objectives): the main study reports all three. The original objective
  is kept for continuity with the original paper whatever it shows.
- D3 (parameterisation): treated as a hyperparameter chosen on validation
  data, with the same three options for both families. No option is removed
  unless it fails gate A for Deep Hedging on at least 5 of the 6
  ticker-liability pairs under both hedging-error objectives.
- D4 (Deep Hedging adequacy, problem P3): met if, under the variance
  objective, Deep Hedging's validation std of hedging error is within 1.5
  times the liability delta hedge's on at least 5 of 6 ticker-liability pairs
  in at least one parameterisation, both seeds averaged.
- Reported, not decided here: ProtoHedge against Deep Hedging and against the
  two rule-based hedges; training failures (a fit whose validation objective
  is worse than the unhedged position's counts as a failure).

### E2 results (2026-11-14; 288 fits, no crash; `phase3/E2/rows.csv`)

Validation episodes; SPY, TLT, NVDA; both liabilities; two seeds. Numbers are
the std of the hedging error divided by the liability delta hedge's, averaged
over the six ticker-liability pairs (below 1 = better than the delta hedge).

| Objective | State | Family | trade | target | residual |
|---|---|---|---|---|---|
| Variance of hedging error | original | Deep Hedging | 0.53 | 0.56 | 0.54 |
| Variance of hedging error | original | ProtoHedge | 0.78 | 0.99 | 0.78 |
| Variance of hedging error | richer | Deep Hedging | 0.45 | 0.41 | 0.39 |
| Variance of hedging error | richer | ProtoHedge | 0.86 | 0.93 | 0.75 |
| CVaR@10% of hedging error | original | Deep Hedging | 1.87 | 1.90 | 1.18 |
| CVaR@10% of hedging error | original | ProtoHedge | 1.86 | 1.61 | 0.96 |
| CVaR@50% of offset (original) | original | Deep Hedging | 3.63 | 3.64 | 1.91 |
| CVaR@50% of offset (original) | original | ProtoHedge | 4.20 | 3.58 | 1.78 |

Decisions under the rules fixed before the runs
- **D1 (state): the richer state is adopted.** It improves the mean validation
  objective of both families over the 36 matched cells (Deep Hedging 0.00442
  to 0.00351, better in 81% of cells; ProtoHedge 0.00709 to 0.00687, better
  in 56%).
- **D2 (objectives): all three are kept.**
- **D3 (parameterisation): none is removed.** Deep Hedging passes gate A on
  all six pairs in every cell but one (target, original objective: five of six).
- **D4 (Deep Hedging adequacy): met.** Under the variance objective Deep
  Hedging's validation std is about half the liability delta hedge's on all
  six pairs in all three parameterisations, with no failure. Problems P3 and
  P4 are resolved under the new protocol.

Reported
- ProtoHedge beats the liability delta hedge on average under the variance
  objective (0.75 to 0.99) and trails Deep Hedging by a factor of about 1.5
  (original state) to 2 (richer state). Under the variance objective its
  training-validation gap is 4 to 15 times Deep Hedging's although it has
  fewer parameters.
- Against the delta-vega hedge (richer state, variance objective): Deep
  Hedging 0.81 to 0.98 of its std, ProtoHedge 1.7 to 2.0.
- The original objective fails as a hedging objective for both families:
  in the trade and target forms 24 of 48 fits end with more dispersion than
  the unhedged position, while validation CVaR@50% improves through
  directional exposure (mean hedging error +0.008 to +0.012). In the residual
  form three Deep Hedging fits keep the untrained delta hedge.
- CVaR@10% trains, but with large training-validation gaps (ProtoHedge target:
  -0.008 in training, -0.021 in validation); 4 of 72 fits end with more
  dispersion than the unhedged position (Deep Hedging 3, ProtoHedge 1).
- Mean validation rank of the three parameterisations (1 = best), per family
  and objective, in the state that will be used: residual is first for both
  families under all three objectives (variance, richer state: Deep Hedging
  1.58 / 2.00 / 2.42 for residual / target / trade, ProtoHedge 1.75 / 2.08 /
  2.17).
- Seed-to-seed spread of the validation std (variance objective): 6% for Deep
  Hedging, 13.5% for ProtoHedge.
- Throughput on the shared machine: 144 cells in 22.7 hours.

Change to the plan, made here before any further run: instead of carrying
every parameterisation into the ten-ticker study, the configuration of each
family (parameterisation and capacity) is chosen on the three development
tickers' validation data and then frozen for all ten tickers. This gives both
families the same tuning budget and a single fixed configuration each, which
are the two alternatives Reviewer 3 asked for, at less than half the compute.

## Stage E3: capacity pilot (written 2026-11-14, before the runs)

Purpose: give both families the same capacity search.
Design (`phase3/E3_spec.json`): development tickers, both liabilities, seeds 0
and 1, variance objective, richer state, residual parameterisation (first by
mean validation rank for both families). Four capacities per family: Deep
Hedging width 8, 20, 64, 128 at depth 3; ProtoHedge K = 10, 25, 50, 100.
ProtoHedge prototypes always come from the width-20 Deep Hedging agent of the
same cell. The width-20 and K = 100 fits are the E2 fits (copied; seeds are
deterministic). The plain-distance ProtoHedge is trained once per cell at
K = 100 as an untuned arm.

Rule: for each family, the capacity with the best mean validation rank over
the 12 ticker-liability-seed cells; ties go to the original size (width 20,
K = 100). The chosen capacity is used for every objective in the main study.

### E3 results (2026-11-14; 108 fits, no failure; `phase3/E3/rows.csv`)

Validation, 12 cells (three tickers, two liabilities, two seeds), variance
objective, richer state, residual parameterisation. "Ratio" is the std of the
hedging error over the liability delta hedge's.

| Family | Capacity | Mean rank (1 = best of 4) | Ratio to delta hedge | Ratio to delta-vega hedge |
|---|---|---|---|---|
| Deep Hedging | width 8 | 3.00 | 0.382 | 0.82 |
| Deep Hedging | width 20 | 3.00 | 0.385 | 0.81 |
| Deep Hedging | width 64 | 2.08 | 0.367 | 0.77 |
| Deep Hedging | width 128 | 1.92 | 0.350 | 0.74 |
| ProtoHedge | K = 10 | 1.75 | 0.578 | 1.28 |
| ProtoHedge | K = 25 | 2.33 | 0.660 | 1.54 |
| ProtoHedge | K = 50 | 2.58 | 0.672 | 1.57 |
| ProtoHedge | K = 100 | 3.33 | 0.750 | 1.71 |
| ProtoHedge, plain distance | K = 100 | (single) | 0.756 | 1.76 |

Both optima are at the edge of their grids (largest network, fewest
prototypes).

### Amendment E3b (written 2026-11-14, before its runs)

Because both optima sit at a grid edge, one further level is added for each
family, symmetrically: Deep Hedging width 256 and ProtoHedge K = 5. The rule
is unchanged and is applied to the five levels of each family; ties still go
to the original size. Separately, since the number of prototypes matters this
much, the plain-distance ProtoHedge receives the same five K levels instead
of staying untuned at K = 100; this is decided before any plain-distance
result at another K has been seen. Spec: `phase3/E3b_spec.json`.

### E3b results and capacity decision (2026-11-14; 72 further fits, no failure; `phase3/E3/rows_with_E3b.csv`)

Validation, same 12 cells, five levels per family. Ranks are within family
(1 = best of 5).

| Family | Capacity | Mean rank | Ratio to delta hedge | Ratio to delta-vega hedge | Best in (of 12 cells) |
|---|---|---|---|---|---|
| Deep Hedging | width 8 | 3.83 | 0.382 | 0.82 | 2 |
| Deep Hedging | width 20 | 3.75 | 0.385 | 0.81 | 2 |
| Deep Hedging | width 64 | 2.58 | 0.367 | 0.77 | 2 |
| Deep Hedging | width 128 | 2.42 | 0.350 | 0.74 | 4 |
| Deep Hedging | width 256 | 2.42 | 0.352 | 0.75 | 2 |
| ProtoHedge | K = 5 | 2.58 | 0.650 | 1.51 | 3 |
| ProtoHedge | K = 10 | 2.25 | 0.578 | 1.28 | 5 |
| ProtoHedge | K = 25 | 2.83 | 0.660 | 1.54 | 2 |
| ProtoHedge | K = 50 | 3.25 | 0.672 | 1.57 | 2 |
| ProtoHedge | K = 100 | 4.08 | 0.750 | 1.71 | 0 |
| ProtoHedge, plain distance | K = 5 | 3.58 | 0.598 | 1.29 | 1 |
| ProtoHedge, plain distance | K = 10 | 1.67 | 0.524 | 1.16 | 4 |
| ProtoHedge, plain distance | K = 25 | 2.17 | 0.530 | 1.17 | 6 |
| ProtoHedge, plain distance | K = 50 | 3.58 | 0.658 | 1.53 | 0 |
| ProtoHedge, plain distance | K = 100 | 4.00 | 0.756 | 1.76 | 1 |

Decision by the rule
- ProtoHedge: K = 10. Plain-distance ProtoHedge: K = 10. Both optima are now
  inside the grid.
- Deep Hedging: widths 128 and 256 have the same mean rank (2.417). The rule's
  tie-break names only the original size, which is not one of the two, so the
  rule does not decide this case. Resolved here, before any further run, by
  taking the size nearer the original: width 128. Its mean ratio is also the
  lower of the two (0.350 against 0.352) and it is best in more cells (4
  against 2). The difference between the two is far below the seed spread, so
  the choice does not matter for the comparison; the grid is flat from width
  64 upward.

Observations recorded with the decision
- With ten prototypes the plain-distance layer is ahead of the encoder layer
  on validation (0.524 against 0.578 of the delta hedge's std). At K = 100 the
  two were equal (0.756 and 0.750).
- Validation over training std: Deep Hedging 1.2 at every width; ProtoHedge
  1.7 to 2.0; plain-distance ProtoHedge 1.35 at K = 10 and 2.1 at K = 100.
- The 800-epoch cap was binding for the prototype families and not for Deep
  Hedging: 9 of 12 plain-distance K = 10 fits and 4 of 12 ProtoHedge K = 10
  fits were still improving when stopped (best validation std fell by 2.4% and
  1.8% on average between epochs 600 and 800, by up to 7% and 10% in single
  fits); no Deep Hedging width-128 fit ran past epoch 388. The capacity
  choice was therefore made under a cap that slightly disadvantaged the
  prototype families. The main study raises the cap for every family (below).

## Main study, protocol v2 (written 2026-11-14, before its runs)

Everything below is fixed before any ten-ticker fit of this study exists and
before the test episodes are opened. Study folder: `phase3/M1_main`. Specs:
`phase3/M1_A_spec.json` to `M1_E_spec.json`; final evaluation:
`phase3/final_spec.json`.

Fixed for every arm
- Ten tickers, both liabilities, richer state (`s1`), the rules at the top of
  this document.
- Frozen configurations, chosen on the development tickers' validation data:
  Deep Hedging width 128, depth 3; ProtoHedge K = 10; plain-distance
  ProtoHedge K = 10; residual parameterisation (except arm D).
- Prototypes for both ProtoHedge forms come from a width-20 Deep Hedging agent
  trained in the same cell, as in the pilots (the original recipe with the
  original network size). That agent is reported as a secondary arm.
- Epoch cap 2000 for every family (raised from 800 because the cap was
  binding for the prototype families), patience 150. The number of fits that
  reach the cap is reported.
- No pilot fit is reused: every fit of this study is trained under this
  protocol, including the development tickers.
- A fit "fails" if its hedging-error std exceeds the unhedged position's on
  the evaluated split. Failures are counted and kept in every average.

Arms (cells = tickers x liabilities x seeds)

| Arm | Purpose | Parameterisation | Objective | Families | Seeds | Cells |
|---|---|---|---|---|---|---|
| A (primary) | Best configuration of each family | residual | variance of hedging error | Deep Hedging, ProtoHedge, plain-distance ProtoHedge (+ width-20 Deep Hedging) | 0-4 | 100 |
| B | Tail objective | residual | CVaR@10% of hedging error | Deep Hedging, ProtoHedge | 0-2 | 60 |
| C | The submitted paper's objective | residual | CVaR@50% of offset | Deep Hedging, ProtoHedge | 0-2 | 60 |
| D | The original output semantics | trade | variance of hedging error | Deep Hedging, ProtoHedge | 0-2 | 60 |
| E | The original sizes | residual | variance of hedging error | ProtoHedge K = 100 (with arm A's width-20 Deep Hedging) | 0-2 | 60 |

Arms B to E change one thing each relative to arm A. Run order: A, B, C, D, E.

Final evaluation (once, after all arms finish; `phase3_final.py`)
- Test episodes 2022-2024, all ten tickers, every seed; equal-ticker averages
  of seed-averaged metrics. This test period was examined in the submitted
  paper and in Phase 2; it is a held-out period for the new models, not
  untouched confirmation data, and the paper will say so.
- Primary metrics: std of the hedging error and CVaR@5% of the hedging error.
  Reported alongside: CVaR@10%, CVaR@50% of the offset (the submitted paper's
  metric), mean hedging error, cost, position diagnostics, failures, seed
  dispersion, per-ticker results for every ticker.
- Primary contrasts, arm A, per liability, on both primary metrics (12 tests,
  Holm-adjusted): ProtoHedge against Deep Hedging; Deep Hedging against the
  liability delta hedge; ProtoHedge against the liability delta hedge.
  "ProtoHedge" in the primary contrasts is the released encoder layer.
- Secondary (reported with unadjusted intervals and labelled as such):
  plain-distance ProtoHedge against ProtoHedge, Deep Hedging and the delta
  hedge; every arm-A family against the delta-vega hedge; ProtoHedge against
  Deep Hedging within arms B, C and D; original sizes (K = 100 against width
  20, and each against its tuned size).
- Intervals: calendar-aligned circular block bootstrap across tickers, 2000
  draws, block 20 trading days (10 and 40 as sensitivity).
- Whatever the outcome, all arms, tickers and seeds are reported. A result of
  parity, or of ProtoHedge behind Deep Hedging or behind the delta hedges, is
  reported as such.

Estimated cost on the shared machine: about 1,000 fits, two to four days with
ten workers. The queue resumes: finished fits are skipped on restart.

## Stage E4: training-data mechanism (written 2026-11-15, before its runs and before any main-study result was read)

Question (problems P9, P13): each ticker gives only about a hundred
independent training paths. Does the gap between the families depend on how
much data they are trained on, and does training one model on all ten tickers
help either of them?

Design (`rev_mechanism.py`; specs `phase3/E4_subset_spec.json`,
`phase3/E4_pooled_spec.json`; study folder `phase3/E4_mechanism`)
- Same configuration as arm A: residual parameterisation, variance objective,
  richer state, Deep Hedging width 128, ProtoHedge K = 10, plain-distance
  ProtoHedge K = 10, prototypes from a width-20 Deep Hedging agent trained on
  the same data.
- Subset: development tickers (SPY, TLT, NVDA), both liabilities, seeds 0-2,
  the most recent 25% and 50% of the training episodes; validation episodes
  unchanged. Epoch cap 2000, patience 150. The 100% point is arm A.
- Pooled: one model per liability and seed (0-2) trained on the training
  episodes of all ten tickers together, with pooled standardisation and
  pooled validation episodes for the checkpoint; scored on each ticker
  separately. Same optimiser and batch size (32), so one pooled epoch has
  about ten times the gradient steps of a single-ticker epoch; cap 200 epochs,
  patience 30.
- Reported on validation data (the same split for every point of the curve):
  std of the hedging error over the liability delta hedge's, per family, at
  25%, 50%, 100% and pooled. No configuration is chosen from these runs.

Amendment to the final evaluation, declared here: the pooled models are also
evaluated once on the test episodes of each ticker, in the same run as the
main arms, as secondary arms (`dh_pooled`, `ph_pooled`, `phd_pooled`), with
secondary contrasts against the per-ticker models of the same family and
against the liability delta hedge. The primary contrasts are unchanged.

## Amendment: arm F, the original paper's settings together (written 2026-11-15, before its runs; test episodes unopened)

Arms B to E each change one thing relative to arm A. None of them runs all of
the original paper's settings at once, which is the direct answer to "what
happens when ProtoHedge is run as published on historical data". Arm F does:
output is a trade, objective CVaR@50% of the premium-excluded outcome, the
original state (`s0`: spot, listed call price, time left, positions), Deep
Hedging width 20, ProtoHedge K = 100 with the released encoder layer; ten
tickers, both liabilities, seeds 0-2 (60 cells). Training mechanics are those
of every other arm (standardised inputs, smooth bounds, straight-through
limits, same optimiser, cap and checkpoint rule), so that a difference from
the submitted paper's runs is not a training artefact. Spec
`phase3/M1_F_spec.json`. In the final evaluation these are secondary arms
`dh_orig` and `ph_orig`, with secondary contrasts against each other, the
liability delta hedge and the unhedged position. Primary contrasts unchanged.

## Final evaluation: record (written 2026-11-15 20:05:54, immediately before opening the test episodes)

All training is finished: main study arms A-F (1,120 fits) and stage E4 (168
fits), no failed job. Integrity reports: `phase3/M1_verification_ABCDE.json`,
`phase3/M1_verification_F.json` (every fit passes). The pipeline was rehearsed
on validation data with this exact spec (`M1_main/final_rehearsal_val`).
Spec: `phase3/final_spec.json` (copy kept as `final_spec_as_opened.json`),
hash c0d7d68eaa9f4005: 16 arms, 3 primary contrasts x 2 liabilities x 2 metrics, 34
secondary contrasts, blocks 20/10/40, 2000 draws. Known before opening and
reported as limits: the pooled ProtoHedge fits reached their 200-epoch cap in
3 of 6 (encoder) and 5 of 6 (plain) cases; 13 of 100 plain-distance arm-A fits
reached the 2000-epoch cap.

## Extensions X1 (are the explanations faithful?) and X2 (are they useful?), declared 2026-11-18 before any of these analyses were run

No model is trained or changed. Every analysis below re-runs the frozen
arm-A fits (residual output, variance objective, state s1, K = 10;
`ph_k10` = released encoder layer, `phd_k10` = plain-distance layer; 10
tickers, 2 liabilities, seeds 0-4, 100 fits per layer) on the validation
episodes (2019-2021) and the test episodes (2022-2024). The test episodes
were opened once on 2026-11-15 for the main evaluation; these analyses are
exploratory, designed after the main results were known, and are labelled
so. Validation is computed first as a pipeline check and reported too.

Already known before this was written (computed in the main evaluation and
reported to the user): the top-k weight and top-k reconstruction error, the
number of effective prototypes, and the share of decisions whose heaviest
prototype is the nearest in input space (test: 0.28 encoder, 0.76 plain).
The thresholds of X1a(i) and X1b(i) were therefore set with those numbers in
view. Every other quantity below is new.

Notation. A decision is one episode and step; its state is the market
features at that step plus the inventory held before trading. Its
explanation is the prototype weights w (K numbers summing to one), the
prototypes (real training states with dates) and the output a_k attached to
each; the decision is exactly sum_k w_k a_k. Distances are Euclidean in the
standardised state (training mean and std; inventory scaled by 0.5). The
output scale s is the std of the policy's outputs over all decisions of the
split, both instruments pooled.

**X1a Sufficiency.** (i) Mean |output - top-3 reconstruction| / s; (ii) the
policy acts on its top-3 reconstruction at every decision and the episodes
are re-run: |std(E_top3) - std(E)| / std(E). Pass: (i) <= 0.10 and (ii) <= 0.05.

**X1b Similarity in visible features.** (i) Share of decisions whose
heaviest prototype is the nearest in input space (chance 1/K); (ii) mean over
decisions of the Spearman correlation, across prototypes, between weight and
minus input distance. Pass: (i) >= 0.5 and (ii) >= 0.5. Reference, not a
criterion: the same statistics with an untrained encoder (five random
initialisations, same prototypes and states).

**X1c Self-consistency of attached actions.** The policy evaluated at each
prototype's own state p_k: mean over k of |pi(p_k) - a_k| / s, and the weight
prototype k receives at its own state. Pass: mean error <= 0.10.

**X1d Necessity.** (i) Local: at each decision remove the three heaviest
prototypes and renormalise; mean output change / s, against removing the
three lightest. Pass: heaviest-three change >= 3 x lightest-three change and
lightest-three change <= 0.05. (ii) Outcome level: for each prototype k,
delete it from the policy (weight zero, renormalise), re-run the episodes,
D_k = std(E without k) - std(E); usage u_k = mean weight over decisions.
Pass: mean over fits of the Spearman correlation between u_k and D_k >= 0.5,
and deleting the most-used prototype raises std(E) more than deleting the
least-used in >= 80% of fits.

**X1e Stability.** Gaussian noise with std 0.1 (standardised units) added to
the market features of every decision, inventory unchanged, one draw with a
fixed seed: (i) share of decisions whose heaviest prototype is unchanged;
(ii) mean total-variation distance between weight vectors; (iii) mean output
change / s. Pass: (i) >= 0.9. For reference the median one-step change of
the standardised market features is reported.

**Familiarity distance (X2).** For each decision, the standardised distance
over market features only (spot, listed call price, implied volatility,
listed log-moneyness, and the Asian moneyness for the Asian call; not the two
time features, not inventory) to the nearest prototype of the fit, i.e. to
the closest of the ten reference days its explanations can name. Episode
familiarity distance = mean over its decisions. The same distance is computed
for every training decision; it does not depend on the policy.

**X2a Warning signal.** Per fit, Spearman correlation across episodes between
familiarity distance and error size |E_i - mean(E)|; seed mean per ticker,
equal-ticker panel mean, 95% calendar-aligned block-bootstrap interval
(block 20, 2,000 draws). Also: the partial Spearman correlation given the
episode's mean implied volatility; the same correlation for Deep Hedging
(`dh_w128`, same cell and seed) and for the liability delta hedge; and two
comparator signals for ProtoHedge's errors, the episode's mean implied
volatility and the distance to the nearest of all training decisions (full
library). Pass "flags risky episodes": correlation > 0 with the interval above
0. Pass "adds to implied volatility": partial correlation > 0 with the
interval above 0.

**X2b Fallback rule.** When a decision's familiarity distance exceeds tau, the
95th percentile of the training decisions' distances for that fit, its
correction is set to zero, so the decision falls back to the liability delta
hedge. Sensitivity: the 99th percentile. Controls with the same per-fit
fallback rate: random decisions (20 repeats), the highest implied volatility
decisions, the decisions farthest from the full training library. Measures:
std(E) (primary), CVaR5(E), CVaR50(O), cost; paired calendar-aligned block
bootstrap of gated minus ungated and gated minus each control. Pass "fallback
helps": gated std(E) below ungated with the interval below 0. Secondary: the
same rule applied to Deep Hedging, using the encoder bank of the same cell.

**X2c Coverage over time.** Monthly mean episode familiarity distance,
2005-2024, averaged over fits, with the median implied volatility.
Descriptive.

**X2d Reading the failure of the original settings.** Arm F ProtoHedge
(trade output, K = 100, seeds 0-2): usage-weighted share and mean of the
attached spot trades, share of decisions where the position limits change
the intended position, and the Spearman correlation across fits between that
share and std(E) over the unhedged std. Descriptive.

Reporting. Equal-ticker means of seed means, each layer separately, with the
range across tickers; validation and test. All tests are reported whatever
the outcome; no multiplicity correction (exploratory). A test supports a
claim in the paper only if it passes on the test period; a failed test is
reported as a limitation. "The explanations are faithful" needs X1a, X1c,
X1d and X1e; "the similarity is faithful" needs X1b; "a useful warning signal"
needs X2a's first condition; "the fallback rule helps" needs X2b.

## Extension X3: do prototypes correspond to real market events? (declared 2026-11-19, before the analysis was run)

The original paper says a ProtoHedge decision could be explained by its
similarity to a prototype such as a "2008-style financial crisis". X3 tests
this on the frozen arm-A fits (both layers, 10 tickers, 2 liabilities, seeds
0-4), test episodes (already opened; exploratory). Every ticker is analysed
and reported. Disclosed before writing: the prototype dates of SPY and QQQ
(European call, seeds 0-1, both layers) were listed while checking
feasibility; they include states from September 2008 to April 2009 and
August-September 2011. Training ends in 2018, so no 2020 state can be a
prototype.

Dates. A prototype's date is the trading day of its source state (episode
and step, from `contract_paths`); a decision's date likewise.

Stress prototypes. Primary: the prototype's date falls in one of these
market-wide stress windows of the training period, fixed from the public
record (VIX spikes): 2008-09-15 to 2009-03-31 (financial crisis), 2010-05-06
to 2010-06-30 (flash crash, euro crisis), 2011-08-01 to 2011-10-31 (US
downgrade, euro crisis), 2015-08-17 to 2015-09-30 (China devaluation),
2018-02-02 to 2018-02-28 (volatility shock), 2018-10-01 to 2018-12-31 (Q4
selloff). Secondary: the prototype's implied volatility is at or above the
95th percentile of that ticker's training decisions.

Stressed test decisions. Primary: implied volatility at or above the 90th
percentile of that ticker's test decisions. Secondary: the date falls in
2022-01-21 to 2022-03-15, 2022-05-05 to 2022-05-20, 2022-06-10 to 2022-06-24,
2022-09-23 to 2022-10-21, 2023-03-09 to 2023-03-24 (bank failures) or
2024-08-01 to 2024-08-09 (yen carry unwind).

**X3a Representation.** Share of prototypes that are stress prototypes,
against the share of training decisions in the stress windows; share of
banks with at least one. Pass: prototype share >= 1.5 x the training share.

**X3b Invocation.** Per fit with at least one stress prototype: weight on
stress prototypes, mean over stressed test decisions minus mean over the
others; and the share of decisions whose heaviest prototype is a stress
prototype, stressed minus others. Seed mean per ticker, equal-ticker panel
mean, 95% calendar-aligned block-bootstrap interval (block 20, 2,000 draws).
Pass: weight difference > 0 with the interval above 0.

**X3c Large trades.** Decisions in the top 5% of each fit's absolute trade
(both instruments): share whose heaviest prototype is a stress prototype,
minus the same share among the other decisions. Pass: > 0 with the interval
above 0.

Worked examples, rule fixed now: SPY and QQQ (named by the user), European
call, seed 0, both layers. The prototype list with dates and implied
volatility, and the explanation (top three prototypes with dates and weights)
of the test decision with the highest implied volatility.

A pass supports, for that layer and liability, "the explanations invoke
historical stress episodes when markets are stressed". The encoder layer's
similarity is not visible-feature similarity (X1b), and the report says so.

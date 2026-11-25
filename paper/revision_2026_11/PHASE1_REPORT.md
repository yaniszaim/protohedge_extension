# Phase 1 report: reproducing the original ProtoHedge paper

Date: 2026-11-13. Protocol: `PHASE1_PROTOCOL.md`.

Status: complete. Part A (authors' TensorFlow code) reproduces the paper's
central claim; Part B (PyTorch parity) passes all three gates. One replicate of
one extra arm in the sibling project had not finished.

## Part A. Authors' unmodified TensorFlow code

Sources, used read-only (copies in `phase1_reproduction/`):
- five replicates per world, `deephedging moveable prototypes/protohedge_repro/`
- three restarts per world, `ProtoRL/results/x7/`

`protohedge-main/` in this repository is byte-identical to the authors' final
public commit `0776fd2`, the same code both reproductions import.

### What reproduces

| Check | Paper | Reproduction | Verdict |
|---|---|---|---|
| Simulated test data (unhedged CVaR@50%) | BS -0.10387, SV -0.12462 | -0.103868, -0.124612; BS payoff mean equal to the authors' notebook to 9 digits | identical |
| Deep Hedging gain over unhedged | BS +0.04570, SV +0.03440 | +0.04566, +0.03438 (authors' configuration); five-replicate mean +0.04566, +0.03433 | reproduced |
| ProtoHedge minus Deep Hedging, authors' configuration and test set | BS -0.00023, SV -0.00030 | -0.00020 (SE 0.00009), -0.00021 (SE 0.00038) | reproduced |
| Same gap on a 100,000-path test set, runs that train | BS -0.40%, SV -0.33% | BS -0.41%, -0.28%, -0.25%; SV -0.01%, -0.61%, -0.45%, -0.27%, -1.35% | reproduced |
| ProtoRL X7, runs that train (10 fresh test worlds) | same | BS -0.22%, -0.30%; SV +0.03%, -0.15% | reproduced |

So the paper's central claim, near-parity on CVaR@50% in both simulated
worlds, is reproduced by two independent sets of runs.

### What does not reproduce, or needs a caveat

1. **ProtoHedge does not always train.** With the released encoder layer,
   2 of 5 Black-Scholes replicates stall at a gap of -23% and -30% (ProtoRL: 1
   of 3 in each world, -22% and -15%). Deep Hedging trains in every run
   (between-replicate SD of utility 0.00006 in BS, 0.00011 in SV; ProtoHedge
   0.0084 in BS with the stalls). The paper reports one run per method.
2. **The authors' seeds do not fix the initial weights.** The code seeds
   TensorFlow, but Keras 2.13 draws initialiser seeds from Python's `random`,
   which is never seeded. Their published runs cannot be regenerated from their
   seeds, and every restart is an independent draw.
3. **The feature-weight variant does not reproduce the table.** The
   trainable-feature-weight layer logged in the authors' notebooks trails Deep
   Hedging by 3% to 40% in BS and 4% to 21% in SV in our replicates.
4. **The table's rows come from different layer versions.** The notebook logs
   record 100 trainable weights for the BS run (plain distance, no encoder),
   103 and 1,005 for the feature-weight runs; the released layer has an
   encoder. A plain-distance arm was added to the five-replicate reproduction
   on 2026-11-13 for this reason and is still running.
5. **Table 1 of the paper says drift 0; the code uses 0.1.** The data-identity
   check above passes only with the code default of 0.1.
6. **The "BS" row of Table 2 is not a delta hedge.** The value implied for it
   (-0.05163) equals the authors' plotting helper applied to the unhedged
   payoff. The actual Black-Scholes delta hedge with costs scores -0.0580 on
   the paper's test set, level with Deep Hedging (-0.0582).
7. **ProtoHedge has heavier tails than Deep Hedging in every run that
   trains.** CVaR@5% is 2% to 3% worse in BS and 4% to 13% worse in SV. The
   paper reports CVaR@50% only.

### Consequences for the revision

- The original result can be cited as reproduced, with the training-failure
  rate and the one-run caveat stated.
- Any comparison on historical data needs several restarts per configuration
  and must report failures, not drop them.
- Tail metrics belong in the historical evaluation because the original
  method already shows a tail deficit in simulation.
- In the simulated Black-Scholes world a plain delta hedge matches Deep
  Hedging. The historical paper needs the same baseline (Phase 2, finding F1).

## Part B. PyTorch parity (problem P15): all three gates pass

Six chains (BS and SV; replicates `paper`, `r1`, `r2`), 800 epochs, finished
2026-11-13. Rows: `phase1_pytorch_parity/pytorch_parity_rows.csv`.

| Gate | Black-Scholes | Stochastic volatility | Verdict |
|---|---|---|---|
| G1 data identity (unhedged CVaR@50%, authors' test set) | -0.103868 in both frameworks | -0.124612 in both | pass |
| G2 Deep Hedging utility, 100,000-path test set | PyTorch -0.05820, TensorFlow -0.05823 (0.05% apart) | PyTorch -0.09012, TensorFlow -0.09026 (0.15% apart) | pass |
| G3 ProtoHedge relative gap to Deep Hedging, by replicate | -0.08%, -0.19%, -0.02% (median -0.08%) | -3.52%, -0.30%, -0.22% (median -0.30%) | pass |
| Stalls (gap worse than -3%) | 0 of 3 (TensorFlow 2 of 5) | 1 of 3 (TensorFlow 0 of 5; ProtoRL 1 of 3) | reported |

Reading:
- The PyTorch port reproduces the authors' Deep Hedging to within the
  TensorFlow seed-to-seed spread and reproduces the paper's ProtoHedge
  near-parity on five of six restarts.
- ProtoHedge stalls occasionally in both frameworks. Any historical comparison
  must use several restarts and report the failures.
- ProtoHedge selected very early checkpoints in three PyTorch runs (epochs 20,
  36, 61) under the reference rule of lowest training loss, with no loss of
  test utility.
- Limits: this validates the agents, objective, trainer and simulated world.
  The historical world, cost model, position limits and inventory-step
  accounting have no TensorFlow counterpart and are covered only by the unit
  tests and the Phase 2 re-evaluation.

## Addendum: plain-distance arm of the five-replicate reproduction

The sibling project's plain-distance Black-Scholes arm (no encoder, the layer
the authors' notebook log points to) had finished four of five replicates on
2026-11-13: gaps to Deep Hedging of -0.00008, +0.00004, -0.00004 and +0.00006
on the 100,000-path test set, with no stall. The fifth replicate had not
finished. This is the closest match to the paper's Black-Scholes row.

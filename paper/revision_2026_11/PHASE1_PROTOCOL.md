# Phase 1 protocol: original-paper reproduction and PyTorch parity

Written 2026-11-13, before any full PyTorch parity run. Two-epoch pilots
(`pilot2` profile) were run first, only to measure runtime and to check that
the simulated data are identical; their results are not used.

## Part A. Reproduction of the original paper (authors' TensorFlow code)

No new runs. Two independent reproductions with the authors' unmodified code
already exist in sibling projects and are used read-only:

| Source | Design | Location |
|---|---|---|
| Five-replicate reproduction | `paper` replicate (authors' defaults) plus `r1`-`r4` (new data and TensorFlow seeds); Black-Scholes K=100 and stochastic volatility K=500; 800 epochs; released encoder layer, plus feature-weight and plain-distance arms; criteria R0-R4 fixed in advance | `deephedging moveable prototypes/protohedge_repro/` and its `.deephedging_real_runs/protohedge_repro_tf/paper800/` |
| ProtoRL X7 | 3 restarts per world, K=100/500 and K=16, test on 10 fresh worlds of 50,000 paths; gate X7-0 fixed in advance | `ProtoRL/results/x7/SUMMARY.md`, `ProtoRL/docs/decision_log.md` |

Copies of the summary files are kept in `phase1_reproduction/` with their
source paths. The conclusions drawn from them are in `PHASE1_REPORT.md`.

## Part B. PyTorch parity (problem P15)

Question: does this repository's PyTorch port, which produced every historical
result, behave like the authors' TensorFlow code at the paper's scale?

Script: `scripts/revision/phase1_pytorch_parity.py`. Outputs:
`.deephedging_real_runs/revision_v1/phase1_pytorch_parity/paper800/`.

| Item | Setting |
|---|---|
| Worlds | `SimpleWorld_Spot_ATM` (PyTorch port of the authors' world), 20 steps, dt 0.02, drift 0.1 (the authors' default, as in their notebooks); Black-Scholes and stochastic volatility |
| Training / validation paths | 10,000 / 1,000 |
| Replicates | `paper` (world seed 2312414312), `r1` (1101), `r2` (1102): the same training data as the TensorFlow replicates of the same name. Initial weights cannot be matched across frameworks, so each replicate is an independent restart |
| Deep Hedging | width 20, depth 3, Softplus, features price / delta / time_left |
| ProtoHedge | K=100 (BS), K=500 (SV); prototypes by k-means (`random_state=0`) plus nearest real state on the PyTorch Deep Hedging agent's training states; unweighted distance; the same agent settings as the historical v4 runs |
| Optimizer | Adam 1e-3, batch 32, 800 epochs; SV adds clipvalue 1 and global clipnorm 1 (as the authors' SV notebook cells) |
| Checkpoint | lowest training loss (the reference trainer's rule); no action or inventory penalties |
| Test sets | the authors' (BS seed 1, SV seed 42; 5,000 paths) and the reproduction's large sets (seeds 20261111 / 20261112; 100,000 paths) |
| Metric | mean per-path utility at the trained OCE intercept; relative gap (PH - DH)/|DH| |

Gates, fixed now:

- **G1 data identity.** Unhedged CVaR@50% on the authors' test sets equals the
  TensorFlow value to 6 decimals (BS -0.103868, SV -0.124612).
- **G2 Deep Hedging.** Mean PyTorch Deep Hedging utility on the large test set
  is within 1% of the TensorFlow five-replicate mean, in each world.
- **G3 ProtoHedge.** Median PyTorch relative gap across replicates is no worse
  than -1% in each world (the paper reports -0.40% and -0.33%; the TensorFlow
  runs that train give -0.02% to -1.3%).
- **Stalls** (relative gap worse than -3%) are counted and reported next to
  the TensorFlow stall rate; they do not by themselves fail G3, because the
  authors' code stalls too.

All replicates are reported. If G2 or G3 fails, the port is examined before
any Phase 3 historical training. A pass does not validate the historical
world, cost model or accounting, which have no TensorFlow counterpart.

Runtime measured in the pilot (one thread, machine shared with 8 other jobs):
Deep Hedging 3.3 s/epoch (about 45 min); ProtoHedge 5.5 s/epoch at K=100
(about 75 min) and 15.5 s/epoch at K=500 (about 3.5 h). Six chains, about
19 CPU-hours in total.

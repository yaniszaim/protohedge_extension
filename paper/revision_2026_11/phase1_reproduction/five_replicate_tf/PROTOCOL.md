# ProtoHedge reproduction protocol (written before any full run, 2026-11-11)

Goal: check the simulated results of Faloughi, Guo & Luk, "ProtoHedge:
Interpretable Hedging with Market Prototypes" (ICAIF 2025) with the authors'
own TensorFlow code (`protohedge-main/`, imported unmodified).

## What is reproduced

| Item | Paper (Table 1/2, notebooks) | This run |
|---|---|---|
| Worlds | `SimpleWorld_Spot_ATM`, 20 steps, dt 0.02; BS: `black_scholes=True`; SV: defaults | same code, same config |
| Training / validation paths | 10,000 / 1,000 (`world.clone(samples=nSamples//10)`) | same |
| Objective | CVaR@50% (`utility="cvar"`, `lmbda=1`) | same |
| Vanilla DH | depth 3, softplus, width 20, no initial delta | same |
| ProtoHedge | `agent_type="protopnet"`, features price/delta/time_left, released `ClusteredProtoLayer` (Dense 32 ReLU -> Dense D encoder, softmax of -squared distance, soft-clipped actions) | same |
| Prototypes | k-means (`random_state=0`, `n_init="auto"`) on standardized vanilla-DH states of the training paths, nearest real state per cluster | same, re-extracted from our own vanilla DH |
| K | 100 (BS), 500 (SV) | same |
| Optimizer | Adam lr 0.001, batch 32, 800 epochs; SV cells add clipvalue 1 and global_clipnorm 1 | same |
| Checkpoint | best full-training-set loss over epochs (reference trainer) | same |
| Test set | BS `world.clone(seed=1, samples=5000)`, SV `world.clone(seed=42, samples=5000)` | same, plus an independent 100,000-path test set per world |
| Metric | mean per-path utility at the trained OCE intercept (higher is better) | same, plus empirical CVaR50 of gains |

Replicates: `paper` keeps every default of the reference code (world seed
2312414312, TensorFlow seed 423423423), i.e. the authors' configuration.
`r1`..`r4` change the training-data seed and the TensorFlow seed, so the whole
pipeline (vanilla agent, prototype bank, ProtoHedge) is redrawn.  The paper
reports one trained model per method; the replicates measure how much the
gap moves between equally valid runs.

Paper values to check (Table 2):
BS: DH vs unhedged +0.04570, PH vs unhedged +0.04547, PH vs DH -0.00023 (-0.40%), PH vs BS -0.00677.
SV: DH vs unhedged +0.03440, PH vs unhedged +0.03410, PH vs DH -0.00030 (-0.33%).

## Criteria (fixed before the full runs)

- R0 data identity: the unhedged CVaR50 on the paper test sets equals the
  authors' notebook outputs (BS -0.10387, SV -0.12462).  Checked in the smoke
  test: BS -0.103868, SV -0.124612, BS payoff mean -0.051934138 (notebook
  -0.051934138), 2,604/5,000 zero payoffs (notebook 2,604).
- R1 vanilla DH: DH minus unhedged on the paper test set within +-5% of the
  paper value, for the `paper` replicate and for the replicate mean.
- R2 headline gap, per world:
  - confirmed if the replicate-mean relative gap (PH - DH)/|DH| on the
    100,000-path test set is no worse than -0.5%, and the `paper`
    replicate's gap on the paper test set is within 2 paired standard errors
    of the paper's gap;
  - partially confirmed if the relative gap is between -0.5% and -1%, or
    only one of the two conditions holds;
  - not confirmed if PH trails DH by more than 1%, or ProtoHedge fails to train.
- R3 stability ("repeated runs ... nearly identical utility"): reported
  descriptively as the between-replicate SD of DH and PH utility on the
  common 100,000-path test set, next to the gap.
- R4 the BS row: check whether the paper's "BS" value equals
  `oce_utility("cvar", 1, payoff)` of the unhedged payoff (what
  `plot_bs_hedge.plot_blackscholes` computes) and report the actual
  Black-Scholes delta hedge CVaR50 with costs.

No result is dropped; all replicates are reported.

## Addendum (2026-11-13, written after the MLP and weighted arms finished)

The notebook logs record the trainable-weight count of every ProtoHedge
run, which identifies the layer behind each Table 2 row:

| Run in the notebooks | Logged weights | Layer |
|---|---|---|
| BS K=100, trained 2025-05-31, loaded in `proto-trainer.ipynb` 2025-06-19 (best training loss 0.161502) | 100 | plain distance `softmax(-||x - p||^2)`, no encoder, no feature weights |
| BS K=100, "NEW WEIGHTED SIMILARITY", 2025-07-24 (best 0.163174; test utility -0.05977) | 103 | trainable per-feature weights |
| SV K=500, loaded 2025-07-24 (best 0.213738; test utility -0.09042) | 1005 | trainable per-feature weights |
| released `layers.py` | K*nInst + 227 (BS) | MLP encoder |

The weighted BS run's test utility (-0.05977, a -2.8% gap) is inconsistent
with Table 2's BS gap (-0.4%); the plain-distance BS run (training loss
0.0001 above vanilla DH) is consistent with it.  So a third arm,
`ph_plain` (`repro_variants.py --variant plain`), is added for BS with the
same banks, vanilla agents, trainer configuration, test sets and criteria
R0-R4.  The arm was added after the other two arms had finished (the reason
is the notebook logs above); it replaces neither of them, and nothing else
in this protocol changes.

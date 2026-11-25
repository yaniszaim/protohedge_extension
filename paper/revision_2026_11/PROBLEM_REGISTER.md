# Problem register

Updated 2026-11-14 after Phases 0-2 and Phase 3 stages E1-E2. One row per problem. "Evidence" points to
`PHASE1_REPORT.md` (P1-x), `PHASE2_REPORT.md` (F-x) or the handoff audit.
Status: **confirmed** = shown from code or saved artifacts; **open** = still
to be measured. Reviewer IDs follow the handoff issue register.

## A. Task validity

| ID | Problem | Source | Evidence | Status | Phase 3 fix |
|---|---|---|---|---|---|
| P1 | The learned policies take direction instead of hedging. A delta hedge of the liability beats every learned model on every metric in all ten tickers, including the training objective | new | F1, F3, F5 | confirmed | 3.1 objective and parameterisation pilot; three-way comparison |
| P1a | CVaR@50% with the premium excluded cannot tell a hedge from a long position on 2005-2018 training data (European: all within 0.002 in sample) | new | F7; confirmed by experiment in E1 and E2: with everything else repaired, 24 of 48 trade/target fits under this objective end with more dispersion than the unhedged position | confirmed | 3.1: keep CVaR@50% for continuity, add a tail-averse objective on premium-inclusive hedging error |
| P1b | For the Asian call the delta hedge is better in sample and neither network finds it | new | F7 | confirmed | 3.2 optimisation gate |
| P2 | No baseline hedges the actual liability; no trivial baselines | new | F1, F2 | confirmed; baselines now implemented | keep liability delta, buy-and-hold, constant-position baselines in every table |
| P8 | The policy state omits the hedge option's strike, maturity and implied volatility, and the liability's implied volatility | new | code; measured in E2: adding them lowers validation dispersion by 21% for Deep Hedging and 3% for ProtoHedge | confirmed; **richer state adopted** (rule D1) | done |
| P9 | About 80-120 independent training paths per ticker; validation loss bottoms out early while training loss keeps falling | new | F8 | confirmed | 3.4 pooled training and learning curves; 3.1 delta-plus-correction policy |

## B. Baseline adequacy

| ID | Problem | Source | Evidence | Status | Phase 3 fix |
|---|---|---|---|---|---|
| P3 | Deep Hedging is mostly a static corner position: 19 of 30 European fits as submitted are untrained or static; spot at the +1 limit in 87% of path-steps | R3-TRAIN, understated | F3 | confirmed; **fixed in the new protocol** (pilot E2, rule D4: validation dispersion about half the delta hedge's on all six development pairs, no training failure) | carried into the main study |
| P4 | Root cause: per-step trade outputs with unstandardised inputs start at the position limits; hard clipping kills the gradient; outcome follows the seed, not the data. ProtoHedge has standardisation, small initial actions and smooth clipping | handoff plus new | F4 | confirmed; **fixed** (`rev_core.py`: shared standardisation, bounded outputs starting near zero, limits that keep the gradient, seeds unique per fit) | done |
| P5 | Checkpoint score is not the reported utility (adds the unhedged-utility term and position penalties); under it an untrained network was selected in 9 of 10 European tickers for one seed | R3-CKPT | F4; `trainer_torch.py`, `objectives_torch.py` | confirmed | 3.2: select on validation hedged utility only, for both families |
| P18 | "Spot-Delta" uses the listed call's delta, which is the wrong delta for the Asian call and an approximate one for the European; its band variant was tuned on mean offset | new | F2; `real_data_sweep_torch.py` (`tuned_baseline_metric`) | confirmed | relabel as "listed-call delta"; liability delta hedge becomes the main rule-based baseline |
| P21 | v4 saved no training curves, so convergence cannot be shown for 228 of 240 reported fits | R3-TRAIN | F8 | confirmed | save full histories in every new run |

## C. Fairness of the ProtoHedge versus Deep Hedging comparison

| ID | Problem | Source | Evidence | Status | Phase 3 fix |
|---|---|---|---|---|---|
| P6 | 16 ProtoHedge configurations against 1 Deep Hedging; 25 of 40 selected ProtoHedge models use extensions | R3-BUDGET, R3-FAMILY | handoff audit | confirmed; fixed in the new protocol: five capacity levels and three parameterisations searched for each family on three development tickers' validation data, one frozen configuration each for all ten (E2, E3, E3b) | main study, protocol v2 |
| P19 | With one fixed original-style configuration ProtoHedge ties Deep Hedging (frozen utility -0.0017 European, -0.0002 Asian); the submitted advantage is a selection effect | R3-BUDGET | F6 | confirmed | 3.3: fixed original-style ProtoHedge is the primary model; expanded family is a labelled secondary analysis |

## D. Provenance and reporting

| ID | Problem | Source | Evidence | Status | Phase 3/4 fix |
|---|---|---|---|---|---|
| P7 | Headline table mixes v4 fits with 36 retrained fits | EXTRA-HYBRID | reproduced exactly in Phase 2 (33 European, 3 Asian fits replaced) | confirmed | one uniform protocol feeds every table |
| P16 | Manuscript says the checkpoint sensitivity "requires no retraining"; the fits record about 592 s of training each. Title differs between TeX and submitted PDF | EXTRA-PROV | `complete.json` in the sensitivity tree | confirmed | correct in rewrite |
| P17 | `empirical_lower_tail_cvar` mishandles ties at the quantile | new | F10 | confirmed | use the exact tail mean everywhere; add a unit test |
| P20 | Original paper: 2 of 5 BS restarts stall; seeds do not fix initial weights; Table 1 drift differs from the code; the "BS" row is not a delta hedge; ProtoHedge has heavier tails in every run | new | P1 report, Part A | confirmed | describe the original accurately; several restarts and failure counts in our runs |

## E. Inference

| ID | Problem | Source | Evidence | Status | Phase 3 fix |
|---|---|---|---|---|---|
| P14 | Bootstrap resamples within ticker and ignores common shocks; no multiplicity control; 2022-2024 has been examined repeatedly | handoff | code; measured: calendar-aligned intervals are a median 1.5 times wider (up to 2.0). The submitted European tail-selected advantage becomes [-0.003, +0.012] (`phase2/bootstrap_scheme_comparison.csv`) | confirmed; method implemented (`rev_inference.py`) | 3.6 calendar-aligned blocks, block-length sensitivity, pre-declared contrasts; label the test period as previously examined |
| P15 | The PyTorch port was never validated against the TensorFlow code at paper scale | handoff | all three parity gates pass (P1 report, Part B): Deep Hedging within 0.05-0.15% of TensorFlow; ProtoHedge near-parity on 5 of 6 restarts, 1 stall | **resolved 2026-11-13** for agents, objective, trainer and simulated world; the historical world has no TensorFlow counterpart | none |
| P22 | Three seeds, and seeds are not independent across tickers (same initial weights) | new | F4 | confirmed; fixed in the new protocol (seeds hashed per ticker and family; five seeds in the primary arm) | report per-seed dispersion in the final evaluation |

## F. Interpretability

| ID | Problem | Source | Evidence | Status | Phase 3 fix |
|---|---|---|---|---|---|
| P10 | No worked prototype decomposition of a historical trade and no fidelity diagnostic; prototype files do not record their source episode | R1-PH, R3-EXPL | `prototype_extraction_torch.py` | confirmed; tool written (`rev_explain.py`: worked example by a fixed rule, top-k reconstruction error, similarity); at K = 10 a three-prototype reconstruction has mean abs error 0.012 against an output std of 0.30 (pilot, validation). 2026-11-18: pre-declared faithfulness and usefulness audit X1/X2 run on the final models (`rev_faithful.py`). Stable and necessary: yes. Similarity faithful: plain layer yes, released encoder no. Top-3 and attached-action readings: approximate. Fallback rule: hurts. Details in `PROGRESS_LOG.md` | 3.5 on the final models; done |

| P23 | The encoder makes "similar past state" unreliable: in pilot models the heaviest prototype is the nearest prototype in input space in only 5-21% of decisions (median distance rank 11 to 50 of 100; 78 in one trade-form model), although the top three to five prototypes reproduce the output closely | new | E1, E2 (`rev_explain.py`) | confirmed on pilot models, validation data; at the chosen K = 10 the heaviest prototype is the nearest in input space in 32% of decisions for the encoder layer and 71% for the plain-distance layer | 3.5: report faithfulness and similarity separately; plain-distance ProtoHedge (K = 10) is in the primary arm |

## G. Presentation

| ID | Problem | Source | Status | Phase 4 fix |
|---|---|---|---|---|
| P11 | Data section, episode construction, three CVaR notions and Figure 1 are unclear | R1-DATA, R1-READ, R3-METRIC, R3-FIG | confirmed | episode diagram, metric-purpose table, redrawn figure |
| P12 | Original ProtoHedge read as the authors' own method; QLBS and risk-averse RL references missing | R1-ATTR, R2-REF | confirmed | explicit attribution; add references |
| P13 | Limited novelty; "what is the new insight?" | R2-NOV, R3 | open | 3.4; reframed question (see `EXPERIMENT_PLAN.md`) |

## Resolved or downgraded

- "Reproduce the original paper exactly": the central claim reproduces (P1
  report). Exact regeneration is impossible because the authors' seeds do not
  fix initial weights; this is documented, not a defect of our work.

## Status after the test evaluation (2026-11-15)

Resolved by the new protocol, with test-period evidence and manuscript text:
P1, P1a, P1b, P2, P3, P4, P5, P6, P7, P8, P10, P11, P12, P14, P17, P18, P19,
P21, P22, P23. P9 and P13: addressed by stage E4 and the reframed three-way
question (pooled training removes most of ProtoHedge's deficit). Open: P16
(wording of the old manuscript, superseded by the rewrite), the status of the
test period (held out for the new models, previously examined; stated in the
paper), and an untouched confirmation period (needs 2025 data and the user's
decision).

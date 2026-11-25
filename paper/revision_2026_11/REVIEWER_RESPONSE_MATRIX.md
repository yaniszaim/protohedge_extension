# Reviewer-response matrix (SSCI/CIFEr 2027 reviews, revision for IJCNN 2027)

Updated 2026-11-15 (evening, after the single test evaluation). Status: **diagnosed** = cause established from existing
artifacts; **planned** = fix specified; **trained** = the new protocol is
implemented and its models exist, with validation-period evidence, test
evaluation pending; **done** = test-period evidence and manuscript text
exist. Rows marked **trained** below now have test-period evidence in
`phase3/final_results/` and text in `manuscript/main.tex` (draft v2,
Sections VI-VIII); they are **done** pending coauthor review. The validation
figures quoted in individual rows were written before the test evaluation;
the test-period figures are in `PROGRESS_LOG.md` (2026-11-15 evening).

| Concern | Reviewer | Problem IDs | What we found (Phases 1-2) | Change planned | Evidence to produce | Manuscript location | Status |
|---|---|---|---|---|---|---|---|
| 16 ProtoHedge configurations against 1 Deep Hedging | R3 | P6, P19 | A single fixed original-style ProtoHedge ties Deep Hedging; the reported gain is a selection effect | Fixed original-style ProtoHedge as primary model; four pre-declared configurations per family as the tuned comparison | Done in the new protocol: both families got the same search (three parameterisations, five sizes) on three development tickers' validation data, then one frozen configuration each for all ten tickers (E2, E3, E3b). The original sizes (K = 100, width 20) are a labelled secondary arm | Method, Results | trained |
| 25 of 40 selected ProtoHedge models use extensions | R3 | P6 | Confirmed | Expanded family reported only as a labelled secondary analysis | The extensions are dropped. Only the released encoder layer (primary) and the authors' earlier plain-distance layer (secondary, labelled) are trained | Method | trained |
| Deep Hedging possibly under-trained (initialisation selected in 12 of 60 fits) | R3 | P3, P4, P21 | Worse than stated: 19 of 30 European fits untrained or static; cause is the output parameterisation and hard clipping | Re-parameterise and standardise; save curves; gate against the liability delta hedge | Fixed: validation std of the hedging error is 0.30 (European) and 0.40 (Asian) of the liability delta hedge's, below it in all ten tickers and for all 100 fits; no fit stopped by the epoch cap; full training curves saved for every fit. Convergence figure still to draw | Method, Results | trained |
| Checkpoint selection | R3 | P5, P7, P16 | Score differs from reported utility; headline mixes provenances; "no retraining" sentence is false | One rule (validation hedged utility) for both families; one protocol for all tables | One rule for every fit (best validation value of the trained objective, untrained network eligible), checked on all 760 finished fits by `rev_verify.py`; one protocol for every table | Method | trained |
| Budget-matched tuned Deep Hedging or a single fixed ProtoHedge | R3 | P6, P19 | Fixed ProtoHedge result already available from saved models | Both will be reported | Both exist: equal-budget tuned configurations (arm A) and the single fixed original sizes (arm E, running) | Results | trained |
| Purpose of the three CVaR notions | R3 | P11, P17 | One of the three is computed with a tie-handling flaw | Metric-purpose table; exact tail mean | Phase 4 | Evaluation | planned |
| Figure 1: Asian tick overlap, different scales | R3 | P11 | - | Redraw with a shared scale and the delta-hedge reference | Phase 4 | Results | planned |
| Prototype decomposition of a historical trade | R3, R1 | P10 | Feasible: top three prototypes carry 81-89% of the decision weight | Worked example chosen by a declared rule, with a panel-wide fidelity diagnostic | Tool written and run on pilot models (`rev_explain.py`): at K = 10 three prototypes carry 93% of the weight and reproduce the output with mean abs error 0.012 (output std 0.30). To run on the final models | Interpretability section | trained |
| Data and Section II.B unclear | R1 | P11 | - | Episode diagram; explicit list of what the policy observes and trades | Phase 4 | Data | planned |
| Prototypes and case-based reasoning unexplained | R1 | P10, P12 | - | Short self-contained description of the original method and of one decision | Phase 4, 3.5 | Background | planned |
| "Remains unconvinced" that the method is interpretable | R1 | P10 | Pre-declared faithfulness and usefulness tests X1/X2 (2026-11-18, `PHASE3_PROTOCOL.md`; results `phase3/extensions/`, `PROGRESS_LOG.md`) | Report the audit as it came out | Test period: explanations stable (X1e pass) and they name the prototypes that matter (X1d pass, except plain European at the outcome level); the released encoder's similarity does not match visible similarity (X1b fail: 19% / 37% vs 65% / 75% untrained), the plain layer's does (73% / 78%); top-3 and attached-action readings are approximate (X1a, X1c fail); familiarity distance flags harder episodes (X2a pass, mostly via volatility); fallback to the delta hedge hurts (X2b fail). X3 (2026-11-19): prototypes over-represent past stress periods (20-24% vs 9%) and in stressed test markets receive more weight (plain 13% to 27%, encoder 17% to 20%; X3a and X3b pass); the large-trade version fails (X3c) | Interpretability section (not yet written) | evidence done, text pending the user's decision |
| Sections hard to read | R1 | P11 | - | Restructure around one question and one table | Phase 4 | all | planned |
| ProtoHedge misread as the authors' own method | R1 | P12 | - | Attribute to Faloughi, Guo and Luk in abstract, introduction and method | Phase 4 | Abstract, Introduction | planned |
| Limited novelty | R2, R3 | P13, P1 | The study as submitted could not support a strong claim: neither learned model hedges as well as a delta hedge | Reframe: three-way comparison with a properly trained Deep Hedging, plus two mechanism experiments (data scarcity, regimes) | Three-way comparison trained; mechanism runs (data fraction, pooled ten-ticker training) queued as stage E4 | Introduction, Results | trained |
| Missing references (QLBS; risk-averse RL for hedging) | R2 | P12 | Vittori et al. is already cited by the original paper | Add Halperin (QLBS) and Vittori, Trapletti and Restelli, with related work | Phase 4 | Related work | planned |

## Concerns no reviewer raised, which a finance reviewer would

| Concern | Problem IDs | Found | Change planned | Status |
|---|---|---|---|---|
| No hedging-error dispersion against a delta hedge | P1, P2 | Delta hedge has 3.5 to 5 times lower dispersion than the learned models | Three-way comparison in the main table; on validation data both learned families are now below the delta hedge's dispersion in all ten tickers | trained |
| "Spot-Delta" hedges the wrong option for the Asian call | P18 | Confirmed | Liability delta hedge and delta-vega hedge as rule-based baselines; the listed-call delta kept as a labelled reference | trained |
| Objective cannot identify a hedge | P1a | Confirmed in sample | Variance of the hedging error as the primary objective; CVaR@10% of the hedging error and the submitted CVaR@50% as arms. Under the submitted objective 32 of 60 Deep Hedging and 15 of 60 ProtoHedge fits end with more dispersion than no hedge (validation) | trained |
| Cross-ticker dependence in the bootstrap | P14 | Confirmed from code | Calendar-aligned blocks, Holm adjustment over 12 pre-declared contrasts; implemented and rehearsed on validation data | trained |
| Original paper's own limits (training failures, one run, tail deficit) | P20 | Confirmed by two reproductions | State them when describing the original | diagnosed |

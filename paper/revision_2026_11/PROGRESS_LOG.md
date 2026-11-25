# Progress log: ProtoHedge empirical paper, revision for IJCNN 2027

Newest entry first. Each entry: what was done, what was found, where it is,
what is running. Companion files in this folder: `EXPERIMENT_PLAN.md`,
`PROBLEM_REGISTER.md`, `REVIEWER_RESPONSE_MATRIX.md`, `PHASE1_PROTOCOL.md`,
`PHASE1_REPORT.md`, `PHASE2_REPORT.md`.

Standing rules: completed historical outputs (`submission_rerun_scientific_v4`,
`checkpoint_selection_sensitivity_v1`) and `protohedge-main/` are never
modified; new runs go under `.deephedging_real_runs/revision_v1/`; licensed
data and run trees are not committed; nothing is selected on test data;
2022-2024 is a previously examined test period.

---

## 2026-11-22 (later): full IDA draft written (LNCS, 12 pages)

At the user's request, a complete submission-ready draft. The user will rewrite it, and also wants a
version to link from their CV. Files:
- `manuscript_ida/main.tex`, `manuscript_ida/references.bib`, compiled to `manuscript_ida/build/main.pdf`
  (12 pages including references, no overfull boxes);
- three tables (`generated/tab_original_metric.tex`, `tab_hedging.tex`, `tab_audit.tex`) and two
  figures (`figures/fig_tickers.pdf`, `figures/fig_covid.pdf`), all regenerated from the result files by
  `scripts/revision/rev_ida_assets.py`. For Figure 2 the script re-runs the frozen SPY fits on
  validation; no training, test marker unchanged. Daily series: `phase3/extensions/x3_covid_daily_SPY.csv`.
  Equal-ticker crisis-recall levels: `phase3/extensions/x3_recall_levels.csv`.
- Five references were added after checking them on arXiv and the ACM catalogue: Hoffmann et al. 2021,
  Alvarez-Melis and Jaakkola 2018, Nauta et al. 2023, Adebayo et al. 2018, Holm 1979; plus
  Doshi-Velez and Kim 2017 and a Claude entry for the AI-use disclosure.

Corrections found while drafting, now in the draft and in `WRITING_BRIEF_IDA.md`:
1. On the original metric, PH is "never detectably behind" DH. In the main setup it is slightly ahead,
   detectably for the Asian call (post-hoc interval +0.00041 to +0.00086), so "no detectable difference
   in any setup" was wrong.
2. The original settings fail because of the objective. Setup (ii), correction outputs with the original
   objective, removes at most 41% of the risk; trade outputs with the variance objective remove 92/88%
   (DH) and 87/82% (PH). The raw-input limit saturation belongs to the earlier, rejected runs; our arm F
   used standardised inputs, and its spot position was at the limit in only 11-12% of decisions.
3. Crisis recall computed with equal ticker weights: plain layer 13% to 26% (European), 15% to 28%
   (Asian); encoder 17% to 20%, 20% to 26%. The earlier pooled figures were 13-27% and 16-30%.
4. The encoder's heaviest prototype is the nearest state in 19% / 37% of decisions: "usually not", not
   "rarely".

The draft's methods section discloses AI use, including AI-drafted text. IDA's policy permits LLM use for
editing, polishing and coding with disclosure, so a submitted version should be rewritten by the authors,
or the PC asked, and the disclosure kept accurate.

---

## 2026-11-22: target changed to IDA 2027; brief converted

The user dropped ESANN: its call says applications are not a central focus.
The new target is IDA 2027 (Intelligent Data Analysis, Pisa, 21-23 April
2027). Abstract registration 28 Nov 2026 and paper 5 Dec 2026 (23:59 AoE, CMT,
Regular Paper Track); decision 22 Jan 2027, so IJCNN (31 Jan) stays open as
a fallback. Rules from ida2027.eu: LNCS, 10-12 pages including references and
appendices; single-blind, at least three reviewers, judged on novelty,
technical quality, potential impact and clarity; LLM use for editing,
polishing and coding allowed if disclosed in the methods section; in-person
attendance strongly encouraged; arXiv allowed. The brief is now
`WRITING_BRIEF_IDA.md`, with a 12-page section plan, two proposed figures
(Figure B, the March 2020 case, not yet generated), the AI-use disclosure
note, and a plan to 5 Dec.

---

## 2026-11-19 (evening): target ESANN 2027 (25 Nov), writing brief

The user will submit to ESANN 2027 (deadline extended to 25 Nov 2026;
decisions 22 Jan 2027, so IJCNN 2027 on 31 Jan stays open as a fallback). Rules
read on esann.org today: 6 pages including figures and references, esannV2
template, A4, single-blind review, no supplementary material, no AI policy
found. The user will write the paper; `WRITING_BRIEF_ESANN.md` holds the
thesis, claims with their evidence, a 6-page section plan, every number with
its source file, the reviewer comments (fixed and still to address in the
text), the disclosure checklist and a day plan. Correction found while
compiling it: 296 of 300 main-setup fits beat the delta hedge on spread, not
all 300. The exceptions are PH on XLK European (seeds 1, 4), NVDA Asian (seed
2) and XLE Asian (seed 0); every ticker's seed average beats it.

---

## 2026-11-19 (later): case study, the March 2020 crash explained by 2008 prototypes

Illustration requested by the user (not a pre-declared test). Rule stated
before looking: SPY, European call, all five seeds and both layers, the
crash window 2020-02-24 to 2020-03-31 against calm 2019, every fit reported.
2020 lies in the validation period: no model trained on it, but stopping
epochs were chosen on 2019-2021, so the crash-period performance figures are
slightly optimistic. Files: `phase3/extensions/x3_case_covid2020_SPY.csv`,
`x3_case_covid2020_SPY_prototype.json`.

- Famous days among prototypes (all tickers): 29 Sep 2008 (House rejects
  TARP, S&P 500 -8.8%) is a prototype in 17 SPY banks; 6 May 2010 (flash
  crash) in 3 QQQ and 1 XLK banks; 5 Feb 2018 (volatility shock) in 7 XLK
  banks. No prototype falls on the 2012 or 2016 election days or Brexit.
- Plain layer, all five models: 49-71% of the weight during the crash goes
  to 2008-09 crisis days, against 0.0-0.2% in calm 2019. Where 29 Sep 2008 is
  a prototype (seeds 0, 2, 4) it is the heaviest one (54%, 50%, 49%); in the
  other two the heaviest is 6 Mar 2009 (53%) or 4 Mar 2009 (60%), the days
  before the March 2009 bottom.
- Released encoder: 0-25% on 2008-09 days during the crash. Its heaviest
  prototypes are mostly calm days, and seed 0 puts more weight on crisis days
  in 2019 (68%) than in the crash (21%).
- The 29 Sep 2008 prototype: day 13 of a hedge started 10 Sep 2008, SPY at
  0.90 of its start, implied volatility 48%, listed call 11% out of the
  money. Plain-layer attached output (seed 0): -0.05 shares, +0.36 listed
  calls relative to the delta hedge.
- What the models did on 9, 12 and 16 Mar 2020: liability delta 0.09. PH held
  0.01-0.10 shares and 0.21-0.61 listed calls (mean 0.44 plain, 0.42 encoder),
  DH 0.00-0.06 shares and 0.69-0.79 calls; the delta hedge holds 0.09 shares
  and no calls.
- Mean |hedging error| of the 20 hedges running on 16 Mar 2020, % of the
  starting price: unhedged 3.6, delta hedge 2.7, PH plain 1.5, PH encoder 1.1,
  DH 0.5 (calm 2019: PH encoder 0.21, plain 0.32).

---

## 2026-11-19: extension X3 (do prototypes correspond to real market events?)

The user asked to test the original paper's claim that a decision could be
explained by similarity to a "2008-style financial crisis" prototype.
Protocol written first: `PHASE3_PROTOCOL.md`, "Extension X3"; it discloses
that the SPY and QQQ prototype dates were listed while checking feasibility.
Code: `scripts/revision/rev_events.py` (new; no training; test marker
unchanged). Summaries: `phase3/extensions/x3a_representation.csv`,
`x3bc_invocation.csv`, `x3_verdict.json`, `x3_prototype_dates.csv`.

Test period, all 10 tickers; encoder / plain layer (European; Asian):
- X3a PASS (all four): 20% / 22% (24% / 23%) of prototypes are days from the
  declared training stress windows, against 9.4% of training decisions
  (2.2-2.6 times); 92-98% of banks contain at least one; by implied
  volatility, 16-18% against 5%.
- X3b PASS (all four): weight on stress prototypes is higher on stressed
  test days. Encoder +0.026 [0.018, 0.038] (+0.059 [0.043, 0.083]), i.e. 17%
  to 20% (20% to 26%); plain +0.137 [0.116, 0.166] (+0.128 [0.103, 0.162]),
  i.e. 13% to 27% (16% to 30%). The named test-window definition of stress
  agrees.
- X3c FAIL (all four): the largest trades are not more often explained by
  stress prototypes (encoder -0.009, plain -0.087 European; about 0 Asian).
  Diagnosis after the fact: 94-100% of the largest trades are the opening
  trade of an episode, so the test did not isolate defensive trades.
- Pre-declared worked examples: the highest implied-volatility test
  decisions of SPY and QQQ are step 19, two days before the listed option's
  expiry (2022-07-27, 2022-12-12), i.e. noisy near-expiry quotes. The rule was
  poorly chosen and these examples should not be used as illustrations.
- Illustration chosen after the test, by event (5 Aug 2024, yen carry
  unwind), seed 0, European; whatever came out is reported, in
  `phase3/extensions/x3_illustration_2024-08-05.json`. QQQ encoder: 0.67 of
  the weight on 16 Feb 2018 (declared volatility-shock window). QQQ plain: 0.70
  on 3 Jul 2008 (outside the windows). SPY plain: 0.85 on 12 Mar 2008 (the
  week of Bear Stearns, outside the windows). SPY encoder: 0.43 on 12 Mar 2018
  and 0.33 on 6 Feb 2015 (calm), with 0.14 on 27 Jan 2009.

Reading: prototypes over-represent past stress periods, and in stressed
markets ProtoHedge leans on them: clearly with the plain layer, modestly with
the released encoder. The specific "large defensive trade" version is not
shown.

---

## 2026-11-18 (night): extensions X1 (faithfulness) and X2 (usefulness) run; results mixed

The user asked for extensions 1 and 2. Protocol written first:
`PHASE3_PROTOCOL.md`, "Extensions X1 ... and X2 ...", with pass criteria. It
discloses that the nearest-state share and top-k numbers were known
beforehand. Code: `scripts/revision/rev_faithful.py` (new; no existing module
or result changed, no training). Per-fit outputs:
`.deephedging_real_runs/revision_v1/phase3/M1_main/extensions_X/{val,test}/`.
Summaries: `phase3/extensions/` (`x1_summary_*.csv`, `x2a_correlations_*.csv`,
`x2b_contrasts_*.csv`, `x2b_scoreboard_*.csv`, `x_verdict_*.json`,
`x2c_monthly.csv`, `x2d_orig_*.csv`). Checks: re-runs reproduce the saved
outcomes exactly (largest difference 0 for policies, 1.8e-7 for the edited
mixture identity); `TEST_OPENED.json` unchanged.

**Test period, arm A, encoder / plain layer (European; Asian).** Verdicts
follow the pre-declared rules.
- X1a sufficiency FAIL (all four): top-3 output error 2.7% / 7.1% of the
  output scale (7.9% / 11.9%), but acting on the top 3 changes std(E) by 7% /
  63% (15% / 35%); limit 5%.
- X1b similarity: encoder FAIL, plain PASS. Heaviest prototype is the
  nearest one in 19% / 73% (37% / 78%) of decisions (chance 10%);
  weight-distance rank correlation 0.30 / 0.87 (0.32 / 0.86). An untrained
  encoder with the same prototypes scores 65% (75%), rank correlation 0.75
  (0.82), so training the encoder moves similarity away from the visible
  features. Every ticker agrees: encoder 12-25% (31-45%), plain 60-82% (71-85%).
- X1c self-consistency FAIL (all four): at a prototype's own state the
  policy's output differs from its attached output by 14% / 18% (23% / 28%) of
  the output scale; self-weight 0.59-0.70; limit 10%.
- X1d necessity: local PASS everywhere (removing the heaviest three changes
  the output by 35-76% of the output scale, the lightest three by 0.1-1%).
  Outcome level: encoder PASS (usage-harm rank correlation 0.69 (0.73); the
  most-used deletion hurts more in 98% (100%) of fits); plain European FAIL
  (0.46, threshold 0.5), plain Asian PASS (0.52).
- X1e stability PASS (all four): the heaviest prototype is unchanged under
  0.1-std noise in 91-94% of decisions; median one-step move 0.20 (0.14) std.
- X2a warning signal: "flags risky episodes" PASS (all four): rank
  correlation of familiarity distance with error size 0.14 [0.09, 0.19] / 0.21
  [0.14, 0.26] (0.25 [0.20, 0.28] / 0.22 [0.18, 0.26]). "Adds to implied
  volatility": encoder European FAIL (0.00 [-0.04, 0.04]); others PASS but small
  (0.06-0.11). Implied volatility alone correlates 0.21-0.26. The full
  training library gives the same as the 10 prototypes (0.15-0.25).
  Familiarity also flags Deep Hedging's errors (0.15; 0.24).
- X2b fallback rule FAIL (all four): falling back to the delta hedge on the
  9-10% least familiar decisions raises std(E) by 0.00069 / 0.00084 (0.00043 /
  0.00035), intervals above 0, and worsens CVaR5 and CVaR50. It is no better
  than random fallback at the same rate. The 99th-percentile rule also hurts
  slightly. Deep Hedging with the same rule is also worse.
- X2c: the least familiar months are 2020-03, 2008-09 to 2009-03; 2022 is
  above 2023-24. Monthly familiarity tracks median implied volatility (r = 0.93-0.94).
- X2d (arm F, original settings) inconclusive: prototypes are not stuck at
  the limits (spot at +1 in 11-12% of decisions). Rank correlation between the
  limit-binding share and the std ratio across fits is 0.58 (0.48).

Validation (2019-2021) shows the same verdicts, except plain European X1d,
which fails on both periods, and encoder Asian "adds to implied volatility",
which passes on both. Reading: the explanations are stable and name the
prototypes that matter. With the released encoder they do not reflect visible
similarity; the plain-distance layer's do. "Top three prototypes" and "the
attached action" are approximations, not what the hedge does. The prototype
library gives a crisis indicator that mostly tracks volatility; using it to
switch to the delta hedge hurts.

---

## 2026-11-18: venue fixed as IJCNN 2027; extensions under discussion

The user chose IEEE IJCNN 2027 (PAKDD and TMLR ruled out). Rules read on
ijcnn.org today: regular papers due 31 Jan 2027 (time zone not stated),
submissions open 1 Dec 2026, notification 15 Mar 2027; 4 to 6 pages including
references and appendices, up to 4 extra pages at $100 each, charged on the
larger of the initial and final length; double-blind, preprints allowed;
AI-generated text must be disclosed in the acknowledgements and the AI system
cited in the sections that use it. Special sessions are announced 15 Dec 2026.

The user asked which extensions would add novelty. Options were proposed in
the conversation; none chosen or started yet. Any extension will be
pre-declared here, designed on validation data only, and evaluated once on
2022-2024, labelled exploratory.

---

## 2026-11-16: original-metric view, earlier runs re-read, PAKDD check (no training, no test re-run)

Nothing was trained and the test evaluation was not re-run. All numbers below
come from files that already existed.

**User decisions today.** Framing: positive but qualified (ProtoHedge carries
over to real data; near-parity with Deep Hedging on the original utility; a
measured cost in hedging error). **Do not rewrite the manuscript until the
user says so.** The user is considering PAKDD 2027 (deadline 20 Nov 2026).

**The original paper's metric (CVaR@50% utility of the outcome), test period.**
ProtoHedge minus Deep Hedging, equal-ticker mean (European / Asian):
- the original paper's settings (arm F): +0.00008 [-0.00665, +0.00737] /
  +0.00045 [-0.00385, +0.00523];
- revised setup trained on that utility (arm C): -0.00137 [-0.00366, +0.00140]
  / -0.00112 [-0.00394, +0.00212];
- main arm (trained on variance): +0.00056 / +0.00065 (no interval; the
  contrast was not pre-declared).
No detectable gap in any setup; single tickers range from -44% to +20% of
Deep Hedging's utility. In arm F both models are below the liability delta
hedge on this utility (-0.0065 and -0.0064 European, intervals exclude zero).
Source: `phase3/final_results/scoreboard_by_ticker.csv`, `contrasts.csv`.

**Spread of the outcome in the simulated worlds** (new, from the saved
100,000-path test outcomes of the PyTorch reproduction,
`phase1_pytorch_parity/paper800/*/*/{dh,ph}/large_test_paths.npz`):
ProtoHedge over Deep Hedging 1.017, 1.004, 1.027 (Black-Scholes); 1.025, 1.006
and 1.175 in the stalled run (stochastic volatility). On the test period the
ratio is 1.13 / 1.00 (arm F), 0.82 / 0.76 (arm C), 1.72 / 1.33 (main arm).
Share of the unhedged spread removed, Deep Hedging / ProtoHedge: arm F 29% /
20% and -14% / -14%; arm C 28% / 41% and -15% / 13%; main arm 93% / 89% and
89% / 85% (liability delta hedge 83% and 81%).

**The run behind the submitted paper (`submission_rerun_scientific_v4`), its
own `sweep_metrics.csv`, column `liability_offset_std`, test split.**
European: unhedged 0.0400, Spot-Delta 0.0142, Deep Hedging 0.0338, ProtoHedge
0.0267 (validation-mean) and 0.0223 (validation-tail). Asian: 0.0239, 0.0234,
0.0246, 0.0260, 0.0222. So ProtoHedge's spread was narrower than Deep
Hedging's there (10 of 10 tickers European, 8 of 10 Asian, validation-tail),
and neither model removed much risk. The column existed; the submitted paper
did not report it. The "ProtoHedge wider than Deep Hedging" result appears
only after Deep Hedging is trained properly.

**Compute, old against new.** Old run: 1,020 fits (51 per ticker and
liability), 800 fixed epochs, one fit at a time, 31 Jul to 22 Aug 2026 (about
528 hours). New: 1,796 fits including pilots, 498.5 processor-hours, 6 to 10
at a time (main study 1,120 fits in about 23 hours of wall time), early
stopping (mean 414 epochs, cap 2,000), same panel and splits, 5 seeds in the
main arm (old: 3).

**PAKDD 2027** (read from pakdd2027.org on 2026-11-16; re-check before
relying on it): deadline 20 Nov 2026 23:59 AoE, notification 26 Feb 2027;
Springer LNCS, 12 pages including references and appendices, double-blind,
abstract up to 200 words, CMT; in person only (Wellington, 29 Jun to 2 Jul
2027). Applied Data Science track: not assessed on methodological novelty,
values investigated failures, "merely applying an existing method to a
real-world dataset is insufficient", restricted data does not disqualify.
Generative AI: use must be disclosed in the submission form and tools "may
only be employed for grammatical error correction". No submission elsewhere
during the review period, which covers IJCNN's deadline. IJCNN 2027: call
says six pages (draft v2 is seven); one official page gives 15 Jan, another
31 Jan.

**Reviewer-response matrix.** The rows still marked "planned" (metric table,
figure, data description, prototype description, attribution, references) are
in draft v2; statuses there were not edited today.

---

## 2026-11-15 evening: TEST EVALUATION DONE (once), manuscript results written (draft v2)

**Test episodes opened once**, 20:05, spec hash c0d7d68eaa9f4005
(`phase3/final_spec_as_opened.json`; record in `PHASE3_PROTOCOL.md`, "Final
evaluation: record"). Do not re-run `phase3_final.py evaluate` with a changed
spec: it refuses without `--acknowledge-reopen`, and any reopening must be
reported in the paper. Summary files: `phase3/final_results/`
(scoreboards, contrasts with intervals, seed dispersion). Full outputs:
`.deephedging_real_runs/revision_v1/phase3/M1_main/final/`.

**Test period 2022-2024, ten tickers; std of the hedging error over the
liability delta hedge's (European / Asian).**
- Deep Hedging 0.40 / 0.61; ProtoHedge (encoder) 0.69 / 0.81; plain-distance
  ProtoHedge 0.56 / 0.69; delta-vega hedge 0.44 / 0.68. No failed fit among
  the 300 arm-A fits; all three families beat the delta hedge in all ten
  tickers for both liabilities.
- Primary contrasts (Holm over 12): ProtoHedge behind Deep Hedging on std and
  on CVaR@5% for both liabilities (all adjusted p 0.006), and behind in all
  ten tickers. Deep Hedging and ProtoHedge both below the delta hedge on std
  (p 0.006). CVaR@5% against the delta hedge: better for the European call
  (Deep Hedging p 0.006, ProtoHedge p 0.03), not significant for the Asian
  call (p 0.42 and 0.53): the learned hedges pay 0.25-0.39% in costs.
- Deep Hedging ahead of the delta-vega hedge in 9 of 10 tickers for each
  liability; ProtoHedge in 0 (European) and 2 (Asian).
- Pooled ten-ticker models: Deep Hedging 0.37 / 0.50, ProtoHedge 0.47 / 0.60,
  plain-distance 0.55 / 0.66. Pooling removes most of ProtoHedge's deficit.
- All original settings together (arm F): parity on the paper's own utility
  (difference +0.008 and +0.045 in percent of price, intervals include zero),
  but both have 4.3 to 6.1 times the delta hedge's dispersion, both are below
  the delta hedge on that utility, and 54 of 120 fits are more dispersed than
  no hedge. NOTE: on validation data ProtoHedge had looked 16-28% behind on
  this utility; the test period shows parity. The user was told both.
- Objective variations within the main configuration: CVaR@50% of the offset
  3.6 to 6.2 times the delta hedge's dispersion, 38 of 120 failures; CVaR@10%
  of the hedging error 1.4 to 2.0 times, no failures. Trade output: Deep
  Hedging 0.49 / 0.66, ProtoHedge 0.76 / 0.95. Original sizes: 0.43 / 0.64
  and 0.86 / 0.90.
- Data mechanism (validation, development tickers): with 25% of the training
  episodes ProtoHedge is no better than the delta hedge (0.93 / 1.03), Deep
  Hedging 0.42 / 0.65.
- Explanations (test, 100 fits per layer, K = 10): top three prototypes carry
  92% (encoder) and 84% (plain) of the weight; three-prototype reconstruction
  error 4% and 9% of the output std; heaviest prototype is the nearest state
  in 28% (encoder) and 76% (plain) of decisions.

**Stage E4** finished 19:47 (168 fits, no failure; pooled ProtoHedge reached
its 200-epoch cap in 3 of 6 encoder and 5 of 6 plain fits, stated in the
paper).

**Manuscript draft v2** (`manuscript/main.tex`, 7 pages, compiles): abstract,
contributions, results (main table, contrasts, per-ticker figure, variations
table, convergence and data figures, explanation tables, worked example),
limitations and reproducibility are written from the test results. Tables
and figures are generated by `scripts/revision/rev_paper_assets.py` into
`manuscript/generated/` and `manuscript/figures/` (validation rehearsal
copies are in the `_rehearsal_val` folders and are not paper results).

**Still open**
- Authors: AI-assistance disclosure (the only remaining placeholder), float
  placement, IJCNN page and format check, coauthor review.
- Halperin (QLBS) volume and pages unconfirmed; DOI is in the entry.
- User decisions: venue (IJCNN 31 Jan, or TMLR sooner), 2025 data for an
  untouched confirmation period, cloud compute (not needed so far).
- Nothing has been committed to git.

---

## 2026-11-15 afternoon: all main-study arms trained and verified; arm F added; E4 running

- Arms A-E finished (340 cells, no failed cell). `rev_verify.py`: every fit
  passes (`phase3/M1_verification_ABCDE.json`).
- **Arm F added** after the user's question about running the method as
  published (protocol amendment written before its runs, test unopened): all
  of the original paper's settings together (trade output, CVaR@50% of the
  offset, original state, width 20, K = 100), ten tickers, seeds 0-2. Trained
  16:44, 120 fits verified (`phase3/M1_verification_F.json`). Validation: 25
  of 60 Deep Hedging and 38 of 60 ProtoHedge fits have more dispersion than
  no hedge; both have 3 to 5 times the delta hedge's dispersion; on the
  paper's own utility ProtoHedge is behind Deep Hedging in 9 of 10 tickers
  for each liability (European -0.0429 against -0.0336, Asian -0.0266 against
  -0.0230). `final_spec.json` now has 16 arms (`dh_orig`, `ph_orig`
  secondary) and the same 3 primary contrasts.
- Validation, arm D (trade output, variance objective): Deep Hedging 0.36 /
  0.41 of the delta hedge's std, ProtoHedge 0.56 / 0.68. Arm E (K = 100):
  ProtoHedge 0.75 / 0.77.
- Stage E4 started 16:44 (pooled, then subsets).
- Bibliography: `buehler2021drift` and `statarb2024deep` confirmed on arXiv.
- Venue check for the user (2026-11-15): ESANN 18 Nov, PAKDD and AAAI-27
  workshops 20 Nov (all too soon), IJCAI 12 Jan, ICML about 22 Jan, IJCNN 31
  Jan; TMLR (rolling) suggested as the sooner option. No decision yet.
- Next: when E4 finishes, full rehearsal, then the single test evaluation.

---

## 2026-11-15 midday: arms A-C trained and verified, pipeline rehearsed, mechanism stage queued

**Main study.** Arms A, B, C finished (220 cells, 760 fits, no failed cell);
arm D running, arm E after it. Much faster than estimated (arm A took 12
hours).

**Integrity check (`scripts/revision/rev_verify.py`, report
`phase3/M1_verification_ABC.json`): 760 of 760 fits pass.** For every fit:
files present; recorded budget and size match the spec; all numbers finite;
the stored policy is the best-validation checkpoint; reloading reproduces the
saved outcomes exactly; an independent simulator (the Phase 2 accounting) fed
the same positions agrees to 1e-7; position and trade limits hold; every
prototype is a training-episode state visited by the cell's width-20 Deep
Hedging agent; no shared seeds; training ends before validation starts; test
episodes locked and unopened. Epoch cap reached: 0 Deep Hedging fits, 2 of
100 ProtoHedge, 13 of 100 plain-distance ProtoHedge (arm A).

**Validation results (not paper results; `phase3/M1_rows_train_val.csv`).**
Std of the hedging error over the liability delta hedge's, equal-ticker mean
of seed means, European / Asian:
- Arm A (variance objective): Deep Hedging 0.30 / 0.40; ProtoHedge 0.60 /
  0.62; plain-distance ProtoHedge 0.53 / 0.60. All three beat the delta hedge
  in all ten tickers for both liabilities. Against the delta-vega hedge: Deep
  Hedging ahead in 10 / 10 tickers, ProtoHedge in 0 / 7, plain-distance in
  0 / 10. Seed spread of the std: Deep Hedging 3%, ProtoHedge 18-20%,
  plain-distance 8-11%.
- Arm B (CVaR@10% of hedging error): both families have more dispersion than
  the delta hedge (1.3 to 1.8 times) with a positive mean hedging error
  (+0.006): the tail objective is met partly by directional exposure in a
  rising validation period.
- Arm C (the submitted objective): 3 to 5 times the delta hedge's dispersion;
  32 of 60 Deep Hedging and 15 of 60 ProtoHedge fits end with more dispersion
  than no hedge.

**Final-evaluation pipeline rehearsed on validation data.**
`phase3_final.py` has a `--rehearsal` mode (validation episodes, separate
folder `final_rehearsal_val`, never touches the test episodes). Evaluate,
report (bootstrap, Holm) and `rev_paper_assets.py` (LaTeX tables, per-ticker
and convergence figures, written to `manuscript/generated_rehearsal_val` and
`figures_rehearsal_val`) all run end to end; panel numbers agree with the
independent summary above.

**Stage E4 (training-data mechanism) declared and queued** behind the main
study (`PHASE3_PROTOCOL.md`, "Stage E4"; specs `phase3/E4_subset_spec.json`,
`E4_pooled_spec.json`; log `.deephedging_real_runs/revision_v1/
phase3_E4_queue.log`). Both runners smoke-tested. `final_spec.json` now has
14 arms (three pooled arms added as secondary) and the same 3 primary
contrasts.

**Manuscript.** Method sections updated to protocol v2 (draft v1).

**Next, in order:** when arms D, E and E4 finish: run `rev_verify.py` on all
specs; full rehearsal; then the single test evaluation
(`phase3_final.py evaluate` and `report` with `phase3/final_spec.json`);
`rev_explain.py` on the final models; fill the manuscript's results.

---

## 2026-11-14 evening: capacities frozen, main study launched

**E3 + E3b (capacity pilot, five levels per family): 180 fits, no failure.**
Table and decision in `PHASE3_PROTOCOL.md` ("E3b results and capacity
decision"); rows in `phase3/E3/rows_with_E3b.csv`.
- Frozen: Deep Hedging width 128 (tie with 256 on mean rank; the nearer to the
  original size was taken, recorded as a gap in the rule), ProtoHedge K = 10,
  plain-distance ProtoHedge K = 10.
- Validation std of the hedging error over the liability delta hedge's at
  these sizes: Deep Hedging 0.35, ProtoHedge 0.58, plain-distance ProtoHedge
  0.52. Over the delta-vega hedge's: 0.74, 1.28, 1.16.
- Fewer prototypes generalise better for both ProtoHedge forms (K = 100:
  0.75 and 0.76). At K = 10 the plain-distance layer is ahead of the encoder
  layer.
- The 800-epoch cap was binding for the prototype families (9 of 12
  plain-distance and 4 of 12 encoder fits at K = 10), never for Deep Hedging.
  The main study uses a 2000-epoch cap for every family.

**Explanations at the chosen size (validation, 12 pilot cells per form;
`explanation_val.json` in each fit folder).**

| | ProtoHedge K = 10 | Plain-distance K = 10 | ProtoHedge K = 100 | Plain-distance K = 100 |
|---|---|---|---|---|
| Weight on the top 3 prototypes | 0.93 | 0.87 | 0.74 | 0.64 |
| Mean abs error of a 3-prototype reconstruction (output std) | 0.012 (0.30) | 0.025 (0.42) | 0.013 (0.17) | 0.028 (0.24) |
| Effective prototypes per decision | 2.9 | 3.6 | 8.6 | 9.7 |
| Heaviest prototype is the nearest in input space | 32% | 71% | 13% | 78% |
| Median distance rank of the heaviest prototype | 3.7 of 10 | 1 of 10 | 20.5 of 100 | 1 of 100 |

With ten prototypes a three-prototype explanation is nearly exact for both
forms; only the plain-distance form's heaviest prototype is usually the most
similar past state (P23).

**Main study (protocol v2) launched 17:54**, background queue, ten workers,
arms A to E in order, about 1,000 fits, two to four days. Protocol:
`PHASE3_PROTOCOL.md` ("Main study, protocol v2"); specs
`phase3/M1_A_spec.json` to `M1_E_spec.json`; final evaluation spec
`phase3/final_spec.json` (written before any fit; 11 arms, 3 primary
contrasts x 2 liabilities x 2 metrics). Outputs:
`.deephedging_real_runs/revision_v1/phase3/M1_main/`; queue log
`.deephedging_real_runs/revision_v1/phase3_M1_queue.log`.
- Restart after an interruption: rerun the same loop over
  `phase3_fit.py queue --spec paper/revision_2026_11/phase3/M1_<arm>_spec.json
  --workers 10` with `PYTHONPATH=/Users/yaniszaim`; finished fits are skipped.
- Do not edit `rev_core.py` or `phase3_fit.py` until the queue has finished.
- Next: mechanism stage (data fraction, pooled training) declared and run
  after the main arms; then the single test evaluation with
  `phase3_final.py evaluate` and `report`; then the manuscript's results.

---

## 2026-11-14: stage E2 done, decisions applied, stage E3 running

**E2 (three-ticker pilot): 288 fits, no crash, 22.7 hours.** Results and the
four pre-declared decisions are in `PHASE3_PROTOCOL.md` ("E2 results"); rows
in `phase3/E2/rows.csv`. Outcome:
- D1: richer state adopted. D2: three objectives kept. D3: no
  parameterisation removed. D4: Deep Hedging adequacy met.
- Under the variance objective with the richer state, validation std of the
  hedging error relative to the liability delta hedge: Deep Hedging 0.39 to
  0.45, ProtoHedge 0.75 to 0.93. Relative to the delta-vega hedge: Deep
  Hedging 0.81 to 0.98, ProtoHedge 1.7 to 2.0.
- The original objective does not produce hedges in either family.
- The residual parameterisation (correction to the liability delta hedge)
  ranks first on validation for both families under all three objectives.

**Plan change (recorded in the protocol before further runs).** Each family's
configuration is chosen on the three development tickers and frozen for all
ten, instead of carrying every option into the ten-ticker study.

**E3 (capacity pilot) launched 10:58**, 12 cells in parallel: Deep Hedging
width 8/20/64/128, ProtoHedge K = 10/25/50/100, plain-distance ProtoHedge;
outputs in `.deephedging_real_runs/revision_v1/phase3/E3_capacity/`. Rule in
`PHASE3_PROTOCOL.md`. Do not edit `phase3_fit.py` while a queue is starting
cells; `rev_core.py` additions are safe once all cells have started.

**Interpretability (P23 registered).** On five pilot models the heaviest
prototype is the nearest one in input space in 5-21% of decisions; the top
five prototypes carry 85-98% of the weight. `rev_explain.py` now handles every
ProtoHedge variant.

**New code.** `phase3_final.py` (the only script that opens the test
episodes; writes `TEST_OPENED.json`), `rev_core.load_policy`, family variants
in `phase3_fit.py` (`dh:w=64`, `ph:k=25`, `phd`). Tests: 9 pass.

**Next.** When E3 ends: choose capacities by the rule, write "Protocol v2:
main study" into `PHASE3_PROTOCOL.md`, copy the reusable pilot fits into the
main study folder, launch arms A-D (about 280 cells, roughly two to three
days at the current load), then run `phase3_final.py` once.

---

## 2026-11-13 (afternoon): stage E1 done, stage E2 running

**E1 (SPY exploration, 36 fits, none failed).** Results and reading are in
`PHASE3_PROTOCOL.md` ("E1 results"); rows in `phase3/E1/rows.csv`. In short:
- With standardised inputs, small initial outputs and a hedging-error
  objective, Deep Hedging learns a hedge in every parameterisation and, under
  the variance objective, beats the liability delta hedge on validation
  (std 0.003 against 0.007 European, 0.003 against 0.0055 Asian) by using the
  listed call. Problems P3 and P4 look fixable; E2 tests this on three tickers.
- The original objective (CVaR@50% of the premium-excluded offset) still
  produces directional positions in both families once the parameterisation
  is repaired, so P1a is an objective problem.
- ProtoHedge (K = 100) trails Deep Hedging under the variance objective
  (validation std 1.4 to 2.5 times larger) but matches or beats the liability
  delta hedge in four of six cells.
- New interpretability finding (to be registered as P23 once E2 confirms it):
  the encoder makes the heaviest prototype an unrelated market state in the
  trade and target parameterisations (median distance rank 78 and 58 of 100),
  although the top three prototypes reproduce the output almost exactly.
  A plain-distance ProtoHedge (`rev_plain.py`, the authors' earlier layer) is
  being tried on two E1 cells as an interpretable-by-construction arm.

**Plain-distance ProtoHedge (two E1 cells, exploratory).** Validation std of
hedging error 0.0060 (correction form) and 0.0084 (trade form), against 0.0048
and 0.0074 for the encoder layer, 0.003 for Deep Hedging and 0.0071 for the
delta hedge. A sharper initial similarity did not help (0.0061, 0.0106). The
plain layer stays as a fixed interpretable-by-construction arm
(`rev_plain.py`, initial weight 1), not a tuned one.

**Manuscript draft v0** (`manuscript/main.tex`, compiles, 4 pages): reframed
introduction with explicit attribution of ProtoHedge, background (encoder and
plain layers), task and episode construction in five steps, what the policy
observes, methods, metric-purpose table, inference, and the reproduction
result. Everything that needs the main study is inside `\pending{}`.
`manuscript/references.bib` has the added references; entries marked VERIFY
need a final check.

**E2 (three-ticker pilot, 144 cells) launched 12:12**, 7 workers; design and
decision rules D1-D4 in `PHASE3_PROTOCOL.md`; outputs in
`.deephedging_real_runs/revision_v1/phase3/E2_pilot/`. Do not edit
`rev_core.py` or `phase3_fit.py` while it runs (each cell imports them fresh).
To finish: `python scripts/revision/phase3_fit.py table --study E2_pilot`, apply
D1-D4, write protocol v2 for the main study.

---

## 2026-11-13 (later): Phase 1B gate passed; Phase 3 started

**Authorisation.** The user asked to continue and work through the problems.
Local compute only; no paid compute, commit or submission has been authorised.

**Phase 1B (done): all three parity gates pass** (`PHASE1_REPORT.md`, Part B).
PyTorch Deep Hedging is within 0.05% (BS) and 0.15% (SV) of TensorFlow;
ProtoHedge gaps are -0.08%, -0.19%, -0.02% (BS) and -3.52% (one stall),
-0.30%, -0.22% (SV). Problem P15 is resolved.

**Phase 3 code (new files only, in `scripts/revision/`).**
- `rev_core.py`: episodes with extra observable features, training-data
  standardisation, three output parameterisations (trade, target position,
  correction to the liability delta hedge), objectives with other tail levels
  and the premium, validation-objective checkpointing, prototype extraction
  that stores each prototype's source episode and date, a delta-vega
  rule-based hedge. Test episodes are locked in code.
- `test_rev_core.py`: 9 tests pass (the new hedging loop reproduces the
  liability delta hedge and buy-and-hold exactly; exact CVaR with ties; no
  look-ahead in features; prototypes are real visited states).
- `phase3_fit.py`: cell runner, queue, summary table.
- `rev_inference.py`: calendar-aligned block bootstrap and Holm adjustment.
- `rev_explain.py`: worked prototype decomposition and fidelity diagnostics.
- Protocol: `PHASE3_PROTOCOL.md` (stage E1 written before its runs).

**Inference finding (P14, measured on the Phase 2 outputs).** Resampling whole
calendar dates across tickers widens intervals by a median factor of 1.5 (up
to 2.0). The submitted European tail-selected ProtoHedge advantage on
CVaR@50% goes from [+0.0004, +0.0080] to [-0.0029, +0.0121]; the Asian one
stays positive, [+0.0011, +0.0073]. `phase2/bootstrap_scheme_comparison.csv`.

**Stage E1 (running since 11:26).** SPY only, training and validation data,
36 fits; outputs in `.deephedging_real_runs/revision_v1/phase3/E1_spy_exploration/`.
Interim (4 Deep Hedging fits, European): with the target-position
parameterisation and CVaR@10% of the hedging error, validation hedging-error
std is 0.0064 (liability delta hedge 0.0071, delta-vega hedge 0.0022, unhedged
0.0245) and the hedge delta correlates 0.96 with the liability delta; with the
original objective the same network returns to a directional long position
(correlation -0.71, std 0.0316). To finish: `python scripts/revision/phase3_fit.py
table --study E1_spy_exploration`, then write stage E2 into `PHASE3_PROTOCOL.md`.

**Related work.** `phase4/related_work_notes.md`: Reviewer 2's two references
verified; closest empirical prior work listed; no prior prototype-based
hedging study on historical option data found.

**Machine.** Load average 250-380 from other sessions; a Deep Hedging fit
takes about 7 min and a ProtoHedge fit about 30 min of wall time at present.

---

## 2026-11-13: Phases 0, 1 and 2

**Authorisation.** The user asked for Phases 0-2 to be run and to be told when
Phase 3 can start. No commit was made.

**Phase 0 (done).**
- Created `paper/revision_2026_11/`, `scripts/revision/`,
  `.deephedging_real_runs/revision_v1/`.
- Baseline: 51 unit tests run, 50 pass, 1 CUDA skip
  (`phase0_baseline/unittest_2026-11-13.txt`); handoff audit saved
  (`phase0_baseline/audit_snapshot_2026-11-13.txt`). Audit mismatches: the
  rebuilt manuscript PDF (noted in the handoff) and four source files that
  differ from the archive-time snapshot (`agents_torch.py`, `run_train_torch.py`,
  `softclip_torch.py`, `tests/test_original_synthetic_reproduction.py`).

**Phase 1A (done, one arm pending elsewhere).** The original paper's
near-parity result reproduces with the authors' code in two independent sets
of runs; ProtoHedge stalls in 2 of 5 Black-Scholes restarts; seeds do not fix
initial weights. See `PHASE1_REPORT.md`. The sibling project's plain-distance
arm was still training at the time of writing.

**Phase 1B (running).** `scripts/revision/phase1_pytorch_parity.py`, six
chains launched 00:21 at low priority; outputs in
`.deephedging_real_runs/revision_v1/phase1_pytorch_parity/paper800/`. Data
identity gate passes. Pilot runtimes: Deep Hedging 3.3 s/epoch, ProtoHedge
5.5 s/epoch (K=100) and 15.5 s/epoch (K=500); slower in practice because the
machine is oversubscribed (load average above 200).
To finish: wait for all six `complete.json` pairs, then run
`python scripts/revision/phase1_pytorch_parity.py summarize` and append the
gate results to `PHASE1_REPORT.md`.

**Phase 2 (done).** `scripts/revision/phase2_diagnostics.py` re-evaluated
about 270 saved fits (the v4 Deep Hedging, selected and fixed ProtoHedge fits,
plus the 36 retrained ones) and eight rule-based strategies on the test
episodes without training. Headline findings (`PHASE2_REPORT.md`):
1. The re-evaluation reproduces the submitted table exactly.
2. A delta hedge of the liability beats every learned model on every metric in
   all ten tickers.
3. Deep Hedging as submitted is mostly a static long position; the cause is
   the output parameterisation with hard clipping, and outcomes follow the
   seed rather than the data.
4. A single fixed original-style ProtoHedge ties Deep Hedging; the submitted
   advantage is a selection effect.
5. The paper's Spot-Delta baseline hedges the wrong option for the Asian call.
6. CVaR@50% with the premium excluded cannot identify a hedge on the training
   data (European); for the Asian call the networks fail in sample.
7. ProtoHedge decisions rest on about four to six prototypes, so a faithful
   worked decomposition is feasible.

**New problems registered:** P17 to P22, P1a, P1b (`PROBLEM_REGISTER.md`).

**Ready for Phase 3?** Yes for design and implementation. Training should
wait for the Phase 1B gate and for the user's decisions on compute, scope and
2025 data (`EXPERIMENT_PLAN.md`, last section).

---

## 2026-11-11: Takeover and plan

Read the handoff, reviews, submitted manuscript, original paper and code;
verified the main handoff claims; produced plan version 1. No repository
change.

## 2026-11-24: IDA draft revised (title, framing, verification pass)

Title chosen by the user: "ProtoHedge Beyond Simulation: Real-Market Performance and Explanations of
Interpretable Deep Hedging". Draft rewritten around Q1 (performance retained) and Q2 (when a cited
prototype can be trusted); 12 pages LNCS, no overfull boxes; previous version kept as
`manuscript_ida/main_v1_2026-11-22.tex`.

Verified from code/data while revising (no new training, test not reopened):
- Eligibility requires the chosen contract to have valid quotes on all 21 observations and expire after
  the episode (`panel_data_pipeline.py`): a look-ahead in eligibility, not in decisions. Strike/expiry/
  optionid changes break the path, so no episode spans a split (checked: no one-day spot move > 40%;
  CRSP spot is unadjusted, e.g. NVDA 2024-06-10).
- Trades at end-of-day mid; no terminal liquidation cost; listed-call cost uses OptionMetrics vega per unit
  vol, prices/vega in units of S0. Asian V0 by Levy moment matching, zero rates, listed call's IV.
- Prototype decomposition is exact for the residual correction; PH positions at a limit in 0.12% of test
  decisions on average (max 4.9% in one fit).
- Rel. in Table 1 = gap / |DH CVaR50(O)| (original paper's convention); column added.
- Plain-distance European PH is detectably behind DH on CVaR50(O) (secondary contrast) — now in Table 1.
- X1a(i) and X1b(i) pass marks were set with values known (PHASE3_PROTOCOL.md) — marked ‡ in Table 3.
- X1d split into local removal (all pass) and deletion correlation (plain European fails, 0.46).
- March 2020: encoder models are heterogeneous (3 rise to 11-26%, 1 has none, 1 falls 68%->21%);
  previous "encoder flat near 15%" wording corrected. Each layer clusters its own bank from the same DH.
- Bootstrap holds trained models fixed; seed variability is not in the intervals.

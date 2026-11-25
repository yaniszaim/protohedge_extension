# Writing brief: ProtoHedge on real option data (IDA 2027: abstract 28 Nov, paper 5 Dec 2026)

Everything needed to write the paper without the working conversation: venue rules, thesis and claims,
section plan with a page budget, every number with its source, reviewer comments, and what must be
disclosed. Numbers are test period 2022-2024, ten tickers weighted equally, seeds averaged, unless
stated otherwise. Written 2026-11-19 for ESANN; converted to IDA 2027 on 2026-11-22. ESANN was dropped
because its call says applications are not a central focus.

---

## 1. Venue: IDA 2027 (Intelligent Data Analysis)

| Item | Rule (ida2027.eu call for papers and FAQ, read 2026-11-22) |
|---|---|
| Abstract registration | **28 Nov 2026, 23:59 AoE**, in CMT (https://cmt3.research.microsoft.com/IDA2027), Regular Paper Track. Whether the abstract can be edited afterwards is not stated, so treat it as final |
| Full paper | **5 Dec 2026, 23:59 AoE** |
| Decision | 22 Jan 2027. A rejection leaves the IJCNN 2027 deadline (31 Jan) open |
| Camera-ready | 1 Feb 2027 |
| Conference | Pisa, 21-23 April 2027. In-person attendance strongly encouraged; online only in exceptional cases (e.g. visas). Early registration by 7 Mar 2027 |
| Length | **10-12 pages including figures, title pages, references and appendices** |
| Template | Springer LNCS (llncs class; Overleaf has the official template) |
| Review | **Single-blind** (author names stay on the paper), at least three reviewers. Criteria: **novelty, technical quality, potential impact, clarity** |
| Mission | "Ideas over performance: a solid motivation can be as convincing as exhaustive empirical evaluation" |
| Topics to select | Explainable, Interpretable, and Trustworthy AI (primary); Temporal, Spatial, and Multimodal Data Analysis; Machine Learning, Data Mining, and Pattern Recognition |
| AI policy | "Use of LLMs for editing, polishing, and coding is permitted, but must be disclosed in the methods section." LLMs cannot be co-authors |
| Dual submission | Not simultaneously submitted elsewhere (EasyChair call), so IJCNN only after the 22 Jan decision |
| Preprints | arXiv allowed |
| Publication | Springer LNCS |

**AI-use note.** Our code, experiments and analysis were built with an AI coding assistant, and this
brief was drafted by one. IDA permits LLM use for editing, polishing and coding if it is disclosed in
the methods section. Two consequences:
1. **Disclose in the methods section** what the assistant did (code, running analyses, editing), and
   that the authors checked every result.
2. **Write the paper text yourself.** The wordings in this brief (claims, the March 2020 paragraph) are
   notes, not text to paste in.

Space: an LNCS page holds about 500 words. Twelve pages, less two tables, two figures, the title block
and about 25 references, leave **about 4,000 words of text**.

---

## 2. The paper in one paragraph (thesis)

> ProtoHedge's main claim carries over from simulation to real option markets. Across ten US
> underlyings and two options, it is level with Deep Hedging on the original paper's own metric, once
> both models are set up to hedge. Its explanations can be checked against market history: they are
> stable, they name the past days that drive each decision, and in stressed markets they lean on past
> crises. During the March 2020 crash, plain-distance models trained on 2005-2018 explained their hedges
> mainly by 2008-09 crisis days, most often 29 September 2008. The price is a wider hedging error than
> Deep Hedging, wider than in simulation and shrinking with more training data. With the released
> encoder layer the named days are often not the most similar ones in visible market features; the
> authors' simpler plain-distance layer fixes this.

Title options:
1. ProtoHedge in Real Option Markets: Near-Parity with Deep Hedging and Explanations that Link New Crises to Past Ones
2. Interpretable Hedging on Real Option Data: ProtoHedge versus Deep Hedging
3. Does Interpretable Deep Hedging Survive Real Markets?

---

## 3. The two claims, worded for the paper

**Claim 1: performance.**
> On the original paper's own metric, ProtoHedge stays level with Deep Hedging on real option data, as
> it did in simulation. Its hedging error is wider than Deep Hedging's, by more than in simulation, and
> the gap narrows with more training data.

Backing: Table 1 (section 5.3); training-data results (section 5.5).

**Claim 2: explanations.**
> ProtoHedge's explanations can be verified against market history. They are stable and name the past
> days that drive each decision. In stressed markets they lean on past crises: during the March 2020
> crash, every plain-distance SPY model explained its hedges mainly by 2008-09 crisis days, most often
> 29 September 2008, the behaviour the original paper anticipated. With the released encoder layer the
> explanations are less faithful: the days it names are usually not the most similar ones.

Backing: Table 2 (section 5.6) and the March 2020 case (section 5.7).

**Do not write** (each can be refuted from our own results):

| Avoid | Why | Say instead |
|---|---|---|
| "works just like in simulation" | True on the original metric only; the hedging-error gap is larger than in simulation | "level with DH on the original metric, as in simulation" |
| "PH matches/outperforms DH" | PH is behind DH on hedging error and tail loss (p = 0.006) | "never detectably behind DH on the original metric" (in the main setup PH is slightly ahead on it, detectably for the Asian call, interval computed afterwards) |
| "faithful" (unqualified) | Two of the faithfulness tests fail for both layers; the similarity test fails for the released layer | "stable and necessary; similarity faithful for the plain-distance layer" |
| "useful in real-world use" | The practical fallback test failed; no user study | "verifiable", "meaningful", "auditable" |
| "untouched test set" | 2022-2024 was examined in the earlier submission | "evaluated once, with all choices fixed beforehand" |
| "every fit beats the delta hedge" | 296 of 300 do | "all ten tickers; 296 of 300 fits" |
| "equal", "parity" | Intervals are wide on real data | "no detectable difference" for a single comparison; "never detectably behind" across the three setups |

---

## 4. Section plan for 12 LNCS pages

| Section | Pages | Words | Content |
|---|---|---|---|
| Abstract | - | 150-200 | Skeleton below; registered in CMT by 28 Nov |
| 1 Introduction | 1.0 | 500 | Why explainable hedging matters; PH (Faloughi, Guo and Luk) and its simulated claim; the two questions (does it carry over to real markets? are its explanations real?); contributions; one-paragraph summary of findings |
| 2 Background and related work | 1.0 | 450 | DH; PH (a prototype is a real past state with an attached action; a decision is the similarity-weighted average; the two layers); learned hedging on real data (Ruf and Wang, Mikkila and Kanniainen); RL hedging (Halperin; Vittori et al.; Kolm and Ritter; Cao et al.); prototype networks (Li et al.; Chen et al.; Rudin) |
| 3 Data and task | 1.0 | 450 + small table | Data, episodes, splits; the liability; costs and limits; outcome and hedging error; what a policy sees (section 5.1) |
| 4 Protocol | 1.0 | 450 | The three setups; equal tuning; training and checkpoint rule; seeds; the three measures in plain words; bootstrap; the single pre-declared test; the simulation reproduction as a correctness check |
| 5 Does PH match DH on real data? | 2.0 | 650 + Table 1 + Figure A | Original metric (Table 1); share of risk removed and worst-5% loss; costs; per-ticker pattern; why the original settings fail (5.4); training data (5.5) |
| 6 Are the explanations real? | 2.5 | 800 + Table 2 + Figure B | Audit design (pre-declared tests); results (Table 2); the layer finding; crisis prototypes and crisis recall; the March 2020 case (Figure B); warning signal and failed fallback |
| 7 Limitations | 0.5 | 250 | The items of this brief's honesty checklist (its section 7) that limit the claims |
| 8 Conclusion | 0.3 | 150 | The thesis restated; recommendation to use the plain-distance layer for explanations |
| Methods note on AI use, data and code availability | 0.2 | 80 | Required AI disclosure (section 1); licensed data; code availability |
| References | 1.5 | - | About 25 |

**A complete 12-page LNCS draft exists** (2026-11-22): `manuscript_ida/main.tex`, compiled to
`manuscript_ida/build/main.pdf`. Its tables and figures are regenerated from the result files by
`scripts/revision/rev_ida_assets.py`. The section plan below is the one the draft follows.

Figures (two, chosen for space; both produced at LNCS width):
- **Figure A**, spread by ticker against the delta hedge: `manuscript_ida/figures/fig_tickers.pdf`
  (the November draft's figure, final test results).
- **Figure B**, the March 2020 case: `manuscript_ida/figures/fig_covid.pdf`. It shows the daily weight
  on 2008-09 crisis prototypes for the SPY models through 2019-2021, crash shaded, implied volatility
  above. Data: `phase3/extensions/x3_covid_daily_SPY.csv`.

How the paper meets IDA's criteria:
- **Novelty:** the first audit of whether prototype-hedge explanations are real on market data. It
  yields the layer finding and the crisis correspondence.
- **Technical quality:** equal tuning, verified training, one pre-declared test, calendar-aligned
  intervals, pre-declared explanation tests.
- **Potential impact:** a practical recommendation (use the plain-distance layer for explanations), an
  explanation of why the original settings fail on real data and how to fix it, and a protocol others
  can reuse.
- **Clarity:** plain definitions; the three-setup table in the PH paper's format; one worked real-world
  case.
- **"Ideas over performance":** lead with the questions and the insight, not with PH's ranking.

Abstract skeleton, one sentence each:
1. PH expresses each hedge as a weighted average of actions attached to real past market states, and was
   reported to stay level with DH in simulation.
2. We test it on 20 years of US option data (ten underlyings, European and Asian calls) against DH, with
   the same inputs and tuning budget for both and a single pre-declared test on 2022-2024.
3. On the original metric PH stays level with DH once both are set up to hedge (no detectable
   difference); with the original settings neither model hedges on real data.
4. The cost is a wider hedging error: PH removes 89% / 85% of the risk against 93% / 89% for DH.
5. Pre-declared tests show the explanations are stable, name the decisive past days, and in stressed
   markets lean on past crises (e.g. March 2020 explained by 29 September 2008).
6. With the released encoder layer the named days are often not the most similar ones; a plain-distance
   layer gives explanations that are.

Contributions paragraph (introduction):
1. A fair test of ProtoHedge on real option data: equal tuning, properly trained DH, one checkpoint rule,
   one pre-declared test. Near-parity on the original metric carries over; the hedging-error cost is
   measured.
2. A pre-declared audit of the explanations on real data: stability, necessity, similarity, and
   correspondence to real market crises, with a worked case (March 2020 against 2008). It shows the
   explanation quality depends on the layer.

---

## 5. Results reference (all numbers)

### 5.1 Setup (for section 3)

- **Data.** OptionMetrics IvyDB US and CRSP; AAPL, IWM, NVDA, QQQ, SPY, TLT, XLE, XLF, XLK, XLV;
  2005-2024. Training ends 2018, validation 2019-2021, test 2022-2024, with a 20-day embargo at each
  boundary. 20,537 training, 6,881 validation and 6,977 test episodes (34,395 in all). Start dates are
  consecutive trading days, so a ticker's ~2,000 training episodes are only about 100 non-overlapping
  months.
- **Episode.** 20 daily decisions. The hedger is short one at-the-money synthetic call, European or
  arithmetic-average Asian, and trades the underlying and one fixed listed call. The listed call has
  20-40 days to expiry and a strike within 10% of spot, and its option identifier stays fixed for the
  episode. Prices are relative to the start price.
- **Costs** (the original study's): 0.0002 S per share; 0.02 vega + 0.0002 delta + 0.0005 price per
  listed call. Positions and daily trades limited to [-1, 1].
- **Outcome and hedging error.** O = payoff owed + trading gains - costs (the original paper's outcome).
  Hedging error E = O + V0, where V0 is the liability's model value at the start (Black-Scholes;
  moment matching for the Asian call), so a perfect hedge has E near 0.
- **What the policy sees.** Underlying and listed-call prices, time left, the listed call's implied
  volatility, log-moneyness and time to expiry, its own two positions, and for the Asian call the
  running average.
- **Main setup ("trained to hedge").** Output is a correction to the liability's delta hedge. Objective
  is the variance of E. DH: width 128, depth 3. PH: K = 10 prototypes, taken as in the original recipe
  (k-means on the states a width-20 DH visits in training, each prototype the nearest real state). Adam,
  learning rate 1e-3, batch 32, at most 2,000 epochs, stop after 150 without improvement, checkpoint at
  the best validation value. Five seeds; other setups three.
- **Equal tuning.** Both families got the same search (output form, objective, inputs, five sizes each)
  on the validation data of SPY, TLT and NVDA only; the configurations were then frozen for all ten
  tickers. Released encoder = the layer in the authors' public code; plain-distance layer = the authors'
  earlier layer (distance in the input features with learned feature weights).
- **Test.** Opened once, on 15 Nov 2026, with the configurations and contrasts fixed beforehand.
  Intervals: calendar-aligned circular block bootstrap (2,000 draws, 20-day blocks), so a market-wide
  shock stays in all tickers at once. Holm correction over the 12 pre-declared primary contrasts.
- **Checks.** All 1,120 main-study fits passed an integrity check: budget, checkpoint, reload,
  independent simulator, position limits, prototypes being real training states. No fit failed in the
  main setup. The epoch limit was reached by no DH fit, 2 of 100 encoder-PH and 13 of 100 plain-PH fits.

Measures, in plain words:
- **Original metric:** CVaR at 50% of O, the average of the worst half of outcomes (higher is better).
  Reported, as in the PH paper, as the gain over not hedging. We use the plain worst-half average; it
  agrees with the PH paper's number to within 0.00001 in simulation.
- **Share of risk removed:** 1 - std(E) / std(E unhedged).
- **Worst-5% loss:** CVaR at 5% of E, the average of the worst 5% of hedging errors.

### 5.2 Simulation reproduction (authors' unmodified code, their test set)

| World | Source | DH vs unhedged | PH vs unhedged | PH - DH | Relative |
|---|---|---|---|---|---|
| Black-Scholes | PH paper | +0.04570 | +0.04547 | -0.00023 | -0.40% |
| Black-Scholes | our rerun | +0.04566 | +0.04549 | -0.00020 | -0.34% |
| Stochastic vol. | PH paper | +0.03440 | +0.03410 | -0.00030 | -0.33% |
| Stochastic vol. | our rerun | +0.03438 | +0.03417 | -0.00021 | -0.23% |

Caveats for one sentence: PH stalled in 2 of 5 Black-Scholes restarts (gaps of -23% and -30%); the
paper reports one run. In simulation PH's hedging-error spread is within 0.4-2.7% of DH's (17.5% in
the one stalled stochastic-volatility run).

### 5.3 Table 1: PH against DH on the original metric (test 2022-2024)

Gain over not hedging on the original metric (fractions of the start price), PH minus DH with 95%
interval, relative gap (PH - DH)/|DH utility|, and share of risk removed.

| Option | Setting | DH gain | PH gain | PH - DH [95% CI] | Relative | Risk removed, DH / PH |
|---|---|---|---|---|---|---|
| European | PH paper's settings | +0.02114 | +0.02122 | +0.00008 [-0.00665, +0.00737] | +0.2% | 29% / 20% |
| European | Equal tuning, PH paper's objective | +0.03056 | +0.02919 | -0.00137 [-0.00366, +0.00140] | -4.4% | 28% / 41% |
| European | Trained to hedge (main) | +0.02570 | +0.02625 | +0.00056 [+0.00000, +0.00106]* | +1.5% | 93% / 89% |
| Asian | PH paper's settings | +0.00944 | +0.00989 | +0.00045 [-0.00385, +0.00523] | +1.7% | -14% / -14% |
| Asian | Equal tuning, PH paper's objective | +0.01591 | +0.01478 | -0.00112 [-0.00394, +0.00212] | -5.5% | -15% / 13% |
| Asian | Trained to hedge (main) | +0.01278 | +0.01343 | +0.00065 [+0.00041, +0.00086]* | +2.7% | 89% / 85% |

\* Interval computed after the main results were seen; label it. References: the delta hedge gains
+0.02766 and removes 83% of the risk (European); +0.01537 and 81% (Asian). Plain-distance PH, main
setup: gain +0.02437, PH - DH -0.00133 [-0.00187, -0.00080], 91% removed (European); +0.01298,
+0.00020 [-0.00005, +0.00042], 87% removed (Asian).

Reading for the text:
- On the original metric no setup shows a detectable PH-DH difference. In the main setup PH is ahead in
  8 of 10 tickers (European) and 9 of 10 (Asian), because it trades less.
- The original metric does not reveal whether a model hedges: in the PH paper's settings both models
  remove under 30% of the risk, and on the Asian call they add risk.
- The delta hedge scores well on the original metric, because that metric rewards average outcome and
  the delta hedge has near-zero costs. Say this in one sentence.

Pre-declared primary measures, main setup (% of the start price; Holm-adjusted p):

| Quantity | European | Asian |
|---|---|---|
| Std of E: unhedged / delta / delta-vega / DH / PH / plain PH | 3.99 / 0.66 / 0.29 / 0.27 / 0.46 / 0.37 | 2.38 / 0.44 / 0.30 / 0.27 / 0.36 / 0.31 |
| Worst-5% loss: delta / DH / PH | -1.48 / -0.84 / -1.27 | -1.13 / -1.00 / -1.17 |
| PH - DH, std of E | +0.191 [0.164, 0.213], p = 0.006 | +0.089 [0.066, 0.109], p = 0.006 |
| PH - DH, worst-5% loss | -0.429 [-0.514, -0.311], p = 0.006 | -0.164 [-0.246, -0.068], p = 0.006 |
| PH - delta, std of E | -0.203 [-0.244, -0.159], p = 0.006 | -0.084 [-0.116, -0.051], p = 0.006 |
| DH - delta, std of E | -0.394 [-0.447, -0.332], p = 0.006 | -0.174 [-0.210, -0.132], p = 0.006 |
| Trading cost: delta / DH / PH | 0.04 / 0.31 / 0.25 | 0.03 / 0.38 / 0.32 |

By ticker:
- PH and DH beat the delta hedge on spread in 10 of 10 tickers for both options (296 of 300 fits; the
  exceptions are PH on XLK European, seeds 1 and 4; NVDA Asian, seed 2; XLE Asian, seed 0).
- PH is behind DH on spread in 10 of 10.
- The plain layer has a narrower spread than the encoder in 9 and 8 of 10.
- Against the delta-vega hedge, DH is ahead in 9 of 10 and PH in 0 and 2.
- Seed-to-seed variation is small (std across seeds of the panel std(E) at most 0.03% of the price).

### 5.4 Why the PH paper's settings fail on real data (one paragraph)

- 54 of 120 fits had more hedging-error spread than not hedging: DH 1/30 European and 23/30 Asian; PH
  10/30 and 20/30. Both models' gains on the original metric are below the delta hedge (intervals
  exclude zero).
- **The cause is the objective** (corrected 2026-11-22). CVaR at 50% of an outcome that excludes the
  premium is only weakly risk-averse, so on trending real prices a directional position scores as well
  as a hedge. Evidence:
  - With this objective, even the tuned correction-to-delta policies (setup ii) remove at most 41% of
    the risk.
  - Trade outputs trained on the variance of E remove 92% / 88% (DH) and 87% / 82% (PH), close to the
    main setup.
- The other diagnosis, raw inputs pushing positions to their limits at initialisation, belongs to the
  **earlier, rejected runs**. Our rerun of the original settings used standardised inputs; there the spot
  position was at its limit in only 11-12% of decisions. Mention it only as "in the earlier version".
- The fix (variance objective; correction-to-delta outputs; standardised inputs; straight-through limits)
  is applied to both models identically.

### 5.5 Training data (validation, SPY/TLT/NVDA; spread relative to the delta hedge, lower is better)

| Training data | DH | PH (encoder) | PH (plain) |
|---|---|---|---|
| 25% of a ticker | 0.42 / 0.65 | 0.93 / 1.03 | 0.88 / 0.82 |
| 50% | 0.39 / 0.46 | 0.77 / 0.92 | 0.50 / 0.63 |
| 100% | 0.27 / 0.44 | 0.51 / 0.64 | 0.45 / 0.56 |
| All ten tickers pooled | 0.27 / 0.39 | 0.38 / 0.50 | 0.49 / 0.59 |

Values are European / Asian. On the test period, pooled training removes 92% / 89% of the risk for PH
against 94% / 91% for pooled DH. One sentence: PH needs more data than DH; with a quarter of the data it
is no better than a delta hedge.

### 5.6 Table 2: the explanation audit (test period; 100 fits per layer; European / Asian)

Pass marks were written down before running (`PHASE3_PROTOCOL.md`, Extensions X1-X3). The 28%-style
similarity numbers and top-k weights were known beforehand; the protocol says so.

| Test | Question | Released encoder | Plain distance |
|---|---|---|---|
| Stability | Same heaviest prototype after a change smaller than a typical day's move? | 94% / 91%, pass | 92% / 94%, pass |
| Necessity | Removing the 3 named prototypes changes the decision much more than removing the 3 lightest? | 0.63 vs 0.001 / 0.76 vs 0.004, pass | 0.35 vs 0.005 / 0.75 vs 0.010, pass (outcome-level part narrowly fails for European) |
| Similarity | Heaviest prototype = most similar past state in visible features? (chance 10%) | 19% / 37%, fail (untrained encoder: 65% / 75%) | 73% / 78%, pass |
| Crisis prototypes | Share of prototypes from past stress periods (vs 9.4% of training days) | 20% / 24%, pass | 22% / 23%, pass |
| Crisis recall | Weight on those prototypes, calm to stressed test days | 17% to 20% / 20% to 26%, pass | 13% to 26% / 15% to 28%, pass |
| Sufficiency | Hedging with the top 3 only gives the same hedge? (limit 5% change) | 7% / 15% change, fail | 63% / 35% change, fail |
| Self-consistency | At a prototype's own state, does the model do what its attached action says? (limit 10%) | off by 14% / 23%, fail | off by 18% / 28%, fail |
| Large trades | Largest trades explained by crisis prototypes? | fail* | fail* |

Necessity figures are output changes in units of the output's standard deviation. \* The test was badly
aimed: 94-100% of the largest trades are the opening trade of an episode, in every market regime.

Further facts:
- Three prototypes carry 95% / 90% of the weight (encoder) and 82% / 87% (plain); a decision uses about
  2.6-4.2 prototypes effectively.
- The plain layer's learned feature weights put the largest share on the listed call's log-moneyness
  (41% / 32%).
- **Warning signal.** Distance from today's market to the nearest prototype rises with the size of the
  hedging error (rank correlation 0.14 / 0.25 encoder, 0.21 / 0.22 plain; intervals above 0). It mostly
  tracks implied volatility: monthly correlation 0.93, and volatility alone predicts about as well. It
  also flags DH's bad episodes. The least familiar months are March 2020 and September 2008 to March
  2009; 2022 is elevated.
- **Fallback test (failed).** Switching to the delta hedge on the 9-10% least familiar decisions makes
  hedging worse (spread +15% / +12% encoder, +23% / +11% plain), and does no better than switching at
  random.
- In the paper, the similarity result is the novel finding: training the released encoder moves its
  notion of similarity away from what a person can see. An untrained encoder with the same prototypes
  would agree with visible similarity 65-75% of the time.

### 5.7 The March 2020 case (illustration; rule fixed before looking: SPY, European, all 5 seeds, both layers)

**The training case.** 29 September 2008: the House rejected the bank bailout (TARP) and the S&P 500
fell 8.8%. It is a prototype in 17 of SPY's 20 models. In the data it is day 13 of a hedge started on
10 September 2008: SPY 10% below the start, implied volatility 48%, listed call 11% out of the money.
The plain layer's attached action: keep roughly the delta hedge in shares (-0.05) and add 0.36 listed
calls.

**The new crisis.** 24 Feb to 31 Mar 2020 lies in the validation period: no model trained on it, but
stopping points were chosen on 2019-2021, so the performance figures are slightly optimistic.

| Plain-distance model (seed) | Heaviest reference day in the crash | Weight | Weight on 2008-09 days, crash | Same, calm 2019 |
|---|---|---|---|---|
| 0 | 29 Sep 2008 | 54% | 54% | 0% |
| 1 | 6 Mar 2009 | 53% | 68% | 0% |
| 2 | 29 Sep 2008 | 50% | 61% | 0.2% |
| 3 | 4 Mar 2009 | 60% | 71% | 0% |
| 4 | 29 Sep 2008 | 49% | 49% | 0% |

- 6 and 4 March 2009 are the last days before the March 2009 market bottom.
- **What it did** on 9, 12 and 16 March 2020 (falls of 7.6%, 9.5% and 12%): the liability's delta had
  fallen to 0.09, so a delta hedge would hold 0.09 shares and no options. PH held a small share position
  and about 0.44 listed calls, close to the 2008 day's attached action. DH held about 0.73 calls.
- **How it did:** average hedging error of the 20 hedges running on 16 March 2020, % of the start price:

| Unhedged | Delta hedge | PH plain | PH encoder | DH |
|---|---|---|---|---|
| 3.6 | 2.7 | 1.5 | 1.1 | 0.5 |

- **Released encoder:** 0-25% of its weight on 2008-09 days during the crash. Its heaviest references
  were mostly calm days (6 Feb 2015, 9 Sep 2016), and one model leaned on crisis days more in 2019 than
  in the crash.
- **Other famous days among prototypes:** 6 May 2010 flash crash (3 QQQ and 1 XLK models), 5 Feb 2018
  volatility shock (7 XLK models). No prototype falls on election days or the Brexit vote: prototypes
  are chosen by clustering market conditions, so they land on days with unusual conditions.
- **Second example, if space allows** (5 Aug 2024, yen carry unwind, test period, chosen after the
  test): QQQ's encoder model put 67% of its weight on 16 Feb 2018, just after the February 2018
  volatility shock. The other three models checked pointed to calm days or to pre-crisis 2008 days.

Notes in paragraph form (facts and order only; write your own sentences, see section 1):
> **A new crisis explained by an old one.** One of SPY's reference days is 29 September 2008, when the
> House rejected the bank bailout and the S&P 500 fell 8.8%. In that state SPY stood 10% below the start
> of the hedge and implied volatility was 48%; the action attached to it is to keep roughly the delta
> hedge in shares and add about a third of a listed call. No model saw 2020 in training. During the
> COVID crash (24 February to 31 March 2020), every plain-distance SPY model put most of its weight on
> 2008-09 crisis days (49-71%, against under 0.2% in 2019), and the three models that keep 29 September
> 2008 chose it as their heaviest reference. On the worst days they held about 0.4 listed calls on top of
> a small share position, close to the 2008 day's action. Across the hedges open on 16 March, their
> average hedging error was 1.5% of the starting price, against 2.7% for the delta hedge and 0.5% for
> Deep Hedging. The released encoder made no such link; its heaviest references were mostly calm days.

---

## 6. Reviewer comments

### Fixed with new evidence (say so in the text)

| Comment (reviewer) | What we did | Where in the paper |
|---|---|---|
| PH chosen from 16 configurations, DH from 1 (R3) | Same search for both on three development tickers, then frozen | Section 4 |
| 25 of 40 selected PH models used add-ons (R3) | Add-ons removed; only the authors' two layers | Section 2 |
| DH possibly under-trained; untrained network selected; width fixed (R3) | Inputs standardised, correction outputs, width tuned (128), 2,000-epoch limit that no DH fit reached; all DH fits beat the delta hedge | Section 4, one sentence each |
| Wants a tuned DH or one fixed PH for all assets (R3) | Both: tuned main setup and the original sizes (width 20, K = 100) | Section 5, one sentence |
| Checkpoint selection (R3) | One rule for every fit (best validation value) | Section 4 |
| Show a prototype decomposition of a real trade (R1, R3) | The March 2020 / 29 Sep 2008 case | Section 6 |
| Unconvinced that the method is interpretable (R1) | Pre-declared explanation tests (Table 2) | Section 6 |
| Limited novelty; what new insight? (R2, R3) | The explanation audit, the layer finding, why the original settings fail and how to fix it, the training-data effect | Introduction, contributions |

### Still to fix in the writing

| Comment (reviewer) | What to do |
|---|---|
| Input data unclear, "explain II.B" (R1) | Section 3: four sentences on the episode and one on what the policy sees (5.1) |
| Prototypes / case-based reasoning unexplained (R1) | Section 2: a prototype is a real past market state with an attached action; a decision is the weighted average of those actions, weights from similarity; then the 29 Sep 2008 example |
| Sections hard to read (R1) | One question per section; plain words; define every measure at first use |
| PH read as our own method (R1) | Attribute PH to Faloughi, Guo and Luk in the abstract, introduction and section 2 |
| Three CVaR notions unclear (R3) | Use only the three measures in 5.1, defined once; drop the others |
| Figure 1: overlapping labels, different scales (R3) | Any figure: shared axes, readable labels; the main results are tables |
| Missing QLBS and Vittori et al. (R2) | One related-work sentence citing both (`halperin2020qlbs`; its volume and pages are unconfirmed, the DOI is in the entry; `vittori2020option`) |

---

## 7. Honesty checklist (each needs a sentence somewhere)

1. 2022-2024 was examined in the earlier version of this study. All present choices were made on
   training and validation data, and the test was evaluated once with a fixed specification.
2. The explanation tests were designed after the main results, with pass marks set before running. Two
   thresholds were set with their numbers known.
3. After-the-fact items are labelled: the main-setup PH-DH interval on the original metric, the March
   2020 case (rule fixed, all models reported, validation period) and the August 2024 example.
4. Which layer each result uses. The released encoder is the main PH model, as declared; the
   explanation case uses the plain layer.
5. With the PH paper's own settings, neither model hedges on real data (54 of 120 fits worse than not
   hedging).
6. In simulation, PH stalled in 2 of 5 Black-Scholes restarts.
7. PH has a wider hedging error and worse tail loss than DH (p = 0.006).
8. Two tests failed for both layers (sufficiency, self-consistency) and the fallback rule failed.
9. Limits of the data: synthetic liability, stylised costs, about 100 independent training months per
   ticker, ten US underlyings.
10. Data are licensed (OptionMetrics, CRSP via WRDS) and cannot be shared; code can be.
11. Disclose AI assistance in the **methods section** (IDA's rule): code, running analyses, editing; the
    authors checked all results.

---

## 8. Writing rules

- Plain words. Define each measure at first use and keep the same term throughout: **ProtoHedge (PH)**,
  **Deep Hedging (DH)**, **delta hedge**, **released encoder layer**, **plain-distance layer**,
  **original metric** (CVaR at 50% of the outcome), **hedging error**, **share of risk removed**,
  **worst-5% loss**.
- Lead with the original metric (as the PH paper reports it), then the share of risk removed.
- Every number carries its period and averaging (test 2022-2024, ten tickers, equal weight).
- Differences come with intervals; write "no detectable difference", never "equal".
- One idea per paragraph; no undefined abbreviations.
- Two tables and two figures (section 4). LNCS figures are single-column: keep labels at least 8 pt and
  use shared scales across panels (Reviewer 3's figure comment).

---

## 9. Plan to 5 December

| Day | Task |
|---|---|
| Sun 22 Nov | Read this brief; decide the open items in section 11; agree the thesis with the coauthors; set up the LNCS template; create a CMT account |
| Mon 23 | Sections 3 and 4 (data, task, protocol) |
| Tue 24 | Section 5 and Table 1; choose Figure A |
| Wed 25 | Section 6 and Table 2; Figure B |
| Thu 26 | Introduction, background and related work |
| Fri 27 | Title and abstract final; coauthors read the abstract |
| **Sat 28** | **Register the paper in CMT: title, authors, abstract, topics (by 23:59 AoE)** |
| Sun 29 - Tue 1 Dec | Limitations, conclusion, AI-use and data notes, references; coauthors read the full draft |
| Wed 2 - Thu 3 | Revise; check every number against the files in section 10; fit 10-12 pages |
| Fri 4 | Final read; LNCS format and PDF check |
| **Sat 5 Dec** | **Upload the paper in CMT (by 23:59 AoE)** |

---

## 10. Where every number lives (all under `paper/revision_2026_11/` unless noted)

| What | File |
|---|---|
| Protocols with pre-declared tests | `PHASE3_PROTOCOL.md` (main study, final evaluation, Extensions X1-X3) |
| Everything in date order | `PROGRESS_LOG.md` |
| Reviewer matrix | `REVIEWER_RESPONSE_MATRIX.md` |
| Main test results | `phase3/final_results/scoreboard_panel.csv`, `scoreboard_by_ticker.csv`, `contrasts.csv`, `seed_dispersion.csv` |
| Explanation tests | `phase3/extensions/x1_summary_test.csv`, `x2a_correlations_test.csv`, `x2b_contrasts_test.csv`, `x_verdict_test.json` |
| Crisis tests | `phase3/extensions/x3a_representation.csv`, `x3bc_invocation.csv`, `x3_verdict.json`, `x3_prototype_dates.csv` |
| March 2020 case | `phase3/extensions/x3_case_covid2020_SPY.csv`, `x3_case_covid2020_SPY_prototype.json` |
| August 2024 example | `phase3/extensions/x3_illustration_2024-08-05.json` |
| Simulation reproduction | `PHASE1_REPORT.md`, `phase1_reproduction/five_replicate_tf/reproduction_rows.csv` |
| Training-data results | `manuscript/generated/e4_dev_tickers.csv` |
| Earlier full draft (IEEE, 7 pages) | `manuscript/main.tex`, `manuscript/references.bib`, `manuscript/figures/` |
| Code | `scripts/revision/` (`rev_core.py`, `phase3_fit.py`, `phase3_final.py`, `rev_faithful.py`, `rev_events.py`) |

---

## 11. Open items for you

1. **Layer emphasis.** Recommended: report both layers; the released encoder stays the main model for
   Table 1, as declared; the explanation case uses the plain layer.
2. **Author list and affiliations** (single-blind, so they appear).
3. **AI-use disclosure in the methods section** (IDA's rule; authors' wording).
4. **Figures:** done (both in `manuscript_ida/figures/`).
5. **Attendance:** who presents in Pisa (21-23 April 2027); IDA strongly encourages attending in person.
6. **If IDA rejects on 22 Jan:** the IJCNN version (31 Jan; 6 IEEE pages plus up to 4 at $100 each;
   double-blind; AI text disclosed in the acknowledgements and cited where used) can be built from this
   brief, the reviews and `manuscript/main.tex`.

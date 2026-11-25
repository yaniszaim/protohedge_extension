# Related-work notes for the rewrite (problem P12, P13)

Checked by web search on 2026-11-13. "Verified" = bibliographic details
confirmed from a publisher, repository or arXiv listing. "To verify" = recalled
or seen in a snippet only; confirm before it goes into the .bib file.

## Requested by Reviewer 2

| Work | Details | Status | Use in the paper |
|---|---|---|---|
| Halperin, "QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds" | arXiv 1712.04609 (Dec 2017); The Journal of Derivatives, 2020, DOI 10.3905/jod.2020.1.108 (DOI from a secondary listing) | verified title, author, arXiv; confirm journal volume and pages | RL formulation of pricing and hedging; value-based alternative to policy search |
| Vittori, Trapletti, Restelli, "Option Hedging with Risk Averse Reinforcement Learning" | ICAIF 2020; arXiv 2010.12245 | verified | risk-averse RL hedging with costs; frontier over risk aversion. Already cited by the original ProtoHedge paper |

## Empirical learned hedging on historical option data (closest prior work)

| Work | Details | Status | Why it matters here |
|---|---|---|---|
| Mikkila, Kanniainen, "Empirical deep hedging" | Quantitative Finance 23(1), 2023, 111-122, DOI 10.1080/14697688.2022.2136037 | verified | trains a deep RL hedger on intraday S&P 500 option prices; the standard reference for learning from historical options |
| Ruf, Wang, "Hedging with Linear Regressions and Neural Networks" | Journal of Business & Economic Statistics 40(4), 2022, 1442-1454, DOI 10.1080/07350015.2021.1931241; arXiv 2004.08891 | verified | on real S&P 500 and Euro Stoxx 50 options a network beats the Black-Scholes hedge, but a linear regression with the leverage effect does about as well. Supports comparing against simple hedges, which the submitted paper did not do |
| Zernikov, "What Does Deep Hedging Actually Learn? Delta Corrections, Regime Fragility, and Symbolic Distillation" | arXiv 2605.21696 (May 2026) | verified listing; read the full text | RL agents against a daily Black-Scholes delta hedge on S&P 500 option episodes; the learned policy is a delta correction; gains are fragile across regimes; distilled to formulas for interpretability. Closest to our "correction to the delta hedge" parameterisation and to our interpretability aim, by a different route (symbolic distillation, not prototypes) |
| Lucius et al., RL hedging of SPX/SPY option exposures with interpretable inputs | arXiv 2512.12420 (Dec 2025) | to verify (title, authors) | reports a modest, not statistically robust edge over rules |
| Bracha, Sakowski, Michankow, "Application of Deep Reinforcement Learning to At-the-Money S&P 500 Options Hedging" | arXiv 2510.09247 (Oct 2025) | to verify | another empirical deep hedging study |

## Why a weakly risk-averse objective produces directional positions (our findings F3, F7)

| Work | Details | Status | Why it matters here |
|---|---|---|---|
| "Is the difference between deep hedging and delta hedging a statistical arbitrage?" | arXiv 2407.14736 | to verify (authors, venue) | argues that deep hedging's departure from delta hedging under some risk measures is speculative positioning |
| Buehler, Murray, Pakkanen, Wood, "Deep Hedging: Learning to Remove the Drift under Trading Frictions with Minimal Equivalent Near-Martingale Measures" | arXiv 2111.07844 | to verify | drift in the training data makes a deep hedger take positions unrelated to the liability; proposes removing it |

## Architecture and method

| Work | Details | Status | Use |
|---|---|---|---|
| Buehler, Gonon, Teichmann, Wood, "Deep hedging" | Quantitative Finance 19(8), 2019 | to verify pages | the baseline method |
| Faloughi, Guo, Luk, "ProtoHedge: Interpretable Hedging with Market Prototypes" | ICAIF 2025, DOI 10.1145/3768292.3770347 | from the paper itself | the method we evaluate; must be attributed to its authors in the abstract and introduction (Reviewer 1 read it as ours) |
| Imaki, Imajo, Ito, Minami, Nakagawa, "No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging" | arXiv 2103.01775 (2021) | verified | precedent for changing the output parameterisation of a deep hedger to make training reliable |
| Chen, Li, Tao, Barnett, Su, Rudin, "This Looks Like That: Deep Learning for Interpretable Image Recognition" | NeurIPS 2019 | to verify | prototype networks; background Reviewer 1 asked for |
| Hull, White, "Optimal delta hedging for options" | Journal of Banking & Finance, 2017 | to verify | minimum-variance delta; a stronger rule-based hedge than Black-Scholes delta |

## Novelty check (2026-11-13)

A search for prototype-based or case-based hedging on historical option data
found no work other than the original ProtoHedge paper (simulated data only).
The nearest neighbour is Zernikov (2026), which gets interpretability by
symbolic distillation of a black-box agent.

## 2026-11-11 — X7 outcome: gate X7-0 not met on the three-seed mean, because ProtoHedge fails to train on one seed of three in each world; the other two seeds reproduce Table 2. X7-1 met

| World | DH − unhedged (paper) | PH − unhedged (paper) | PH vs DH, mean of 3 seeds (paper) | PH vs DH per seed |
|---|---|---|---|---|
| Black–Scholes | +0.04514 (+0.04570) | +0.04081 (+0.04547) | −7.50% (−0.40%) | s0 −0.22%, s1 −21.99%, s2 −0.30% |
| Stochastic vol. | +0.03306 (+0.03440) | +0.02841 (+0.03410) | −5.16% (−0.33%) | s0 +0.03%, s1 −15.37%, s2 −0.15% |

Test utility on 10 fresh worlds (seeds 21–30, 50,000 paths); in-sample numbers agree (−6.83%, −3.88%). Full table: `results/x7/SUMMARY.md`.

- **Gate X7-0 not met** in either world as pre-registered (the threshold was −1% on the three-seed mean).
- **X7-1 met.** Deep Hedging's gain over the unhedged utility is within 0.0006 (BS) and 0.0013 (SV) of the paper's.
- **Why the mean fails.** ProtoHedge seed 1 stalls on a worse plateau in both worlds; its Deep Hedging, which supplies its prototypes, is as good as the other seeds' (SV test utility −0.09021 vs −0.09036, −0.09023). In SV its training loss sits near 0.230 from epoch 5 to 800, against a best of 0.213 for seeds 0 and 2. The trainer keeps the epoch with the lowest training loss, so this is the best seed 1 reached. Seeds 0 and 2 reproduce the paper's gap in both worlds (−0.22% and −0.30% in BS; +0.03% and −0.15% in SV, against −0.40% and −0.33%). The authors did not seed initialisation and report one run per world, so a one-in-three failure rate is compatible with their table.
- **The released code with the authors' prototype extraction therefore reproduces the paper on two of three seeds.** Our X6 port, with prototypes from a Black–Scholes delta hedge holding the option at zero, was 21% behind on every seed. This supports the prototype source as the explanation of X6. It was not tested separately in our port: that rerun was conditional on X7-0 passing, and it is not needed now that X8 builds on the original code.
- **Reported without prediction.**
  - On its successful seeds, ProtoHedge matches Deep Hedging on CVaR50, the objective both are trained on. It has fatter tails under stochastic volatility: CVaR5 of the gains −0.1777 and −0.1734 vs −0.1632 and −0.1623 (8.9% and 6.8% worse), and CVaR1 18.5% and 15.2% worse. In Black–Scholes the tails are close (CVaR5 2.4% and 3.0% worse).
  - ProtoHedge with 16 prototypes is erratic across seeds: within 0.1% of Deep Hedging on SV seed 0, 6% behind on SV seed 2, and up to 26% behind in BS.
- **Consequence for X8.** ProtoHedge's training failures dominate seed-to-seed variation, so X8's confirmation must report, pre-registered:
  - the mean over all seeds, failures included;
  - the failure rate of each method;
- **Correction to X7** (the X7 pre-registration, line "the authors did not seed initialisation", and the X7 outcome). The authors do set seeds: their gym calls `tf.random.set_seed(423423423)` at construction, which overrides the seed our runner set just before. Keras 2.13's unseeded initialisers draw their seeds from Python's `random` module, which neither their notebooks nor our runner seed. Two processes with identical numpy and TensorFlow seeds therefore start from different weights; with `random.seed(0)` added they start from the same weights. So our "model seed" labels in X7 and X8 are labels for independent random restarts, not reproducible seeds, and the authors' published runs are not reproducible from their seeds either. ProtoHedge seed 1 failing in both X7 worlds was a coincidence of two independent restarts. The X7 and X8 statistics are unaffected: each seed is still an independent training run, which is what the seeds-and-paths interval assumes.

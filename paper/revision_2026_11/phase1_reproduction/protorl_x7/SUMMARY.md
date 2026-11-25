# X7: ProtoHedge's original code against the paper's Table 2

| World | Agent | Seeds | Utility (test) | Utility (in-sample) | Unhedged (test) | CVaR50 | CVaR5 | CVaR1 |
|---|---|---|---|---|---|---|---|---|
| BS | dh | 3 | -0.05808 | -0.05813 | -0.10322 | -0.05808 | -0.07438 | -0.08611 |
| BS | ph100 | 3 | -0.06244 | -0.06211 | -0.10325 | -0.06242 | -0.09261 | -0.11639 |
| BS | ph16 | 3 | -0.06760 | -0.06759 | -0.10333 | -0.06757 | -0.12014 | -0.16339 |
| SV | dh | 3 | -0.09027 | -0.09057 | -0.12332 | -0.09027 | -0.16298 | -0.20503 |
| SV | ph16 | 3 | -0.09395 | -0.09403 | -0.12336 | -0.09392 | -0.17739 | -0.22968 |
| SV | ph500 | 3 | -0.09492 | -0.09408 | -0.12333 | -0.09488 | -0.18694 | -0.25734 |

| World | Comparison | Ours (test) | Ours (in-sample) | Paper |
|---|---|---|---|---|
| BS | DH vs unhedged | +0.04514 | +0.04514 | +0.04570 |
| BS | PH vs unhedged | +0.04081 | +0.04119 | +0.04547 |
| BS | PH vs DH | -0.00436 | -0.00397 | -0.00023 |
| BS | PH vs DH, relative | -7.50% | -6.83% | -0.40% |
| BS | gate X7-0 (PH within 1% of DH, test) | NOT met | | |
| BS | PH vs DH per seed | s0 -0.22%, s1 -21.99%, s2 -0.30% | | |
| SV | DH vs unhedged | +0.03306 | +0.03304 | +0.03440 |
| SV | PH vs unhedged | +0.02841 | +0.02953 | +0.03410 |
| SV | PH vs DH | -0.00466 | -0.00352 | -0.00030 |
| SV | PH vs DH, relative | -5.16% | -3.88% | -0.33% |
| SV | gate X7-0 (PH within 1% of DH, test) | NOT met | | |
| SV | PH vs DH per seed | s0 +0.03%, s1 -15.37%, s2 -0.15% | | |

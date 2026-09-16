# Locked Results Ledger

Generated from checksummed inputs by `generate_assets.py`. Higher utility is better.

## Dataset

- 34,395 contract-consistent episodes across ten underlyings.
- 6,977 test episodes in 2022--2024.
- 20 trading decisions and 21 observations per episode.

## Original-source synthetic confirmation

- Black--Scholes ProtoHedge minus Deep Hedging: -0.000910146.
- Stochastic-volatility ProtoHedge minus Deep Hedging: -0.000148437.
- Both reproduce the paper's direction: Deep Hedging is slightly better.
- Black--Scholes parameter ratio, DH/PH: 9.25x.

## Targeted checkpoint-sensitivity historical OCE CVaR@50%

- European Validation-Mean: +0.001009 [-0.002777, +0.005364].
- European Validation-Tail: +0.000514 [-0.003418, +0.004734].
- Asian Validation-Mean: +0.001859 [-0.000309, +0.003863].
- Asian Validation-Tail: +0.004015 [+0.001925, +0.006038].
- Only the Asian Validation-Tail interval excludes zero.

## Robustness facts

- The executed penalized selector selected initialization in 12 of 60 Deep Hedging fits.
- Of 40 historical headline mean/tail selections, 15 are released-paper-like and 25 use at least one extension.
- Exact-SoftClip ProtoHedge minus DH on one European SPY seed: +0.001745 [-0.009190, +0.014586].

## Claim boundary

The evidence supports synthetic near-parity, European historical near-parity, and a positive Asian tail-selected result for the validation-selected expanded ProtoHedge family. It does not support universal ProtoHedge superiority, an exact full-panel numerical port, or superiority over an equally tuned Deep Hedging family.

# Historical checkpoint and fairness audit

Analysis version: `historical-evidence-v1`.

## Meaning of `selected_epoch = -1`

The trainer evaluates the initialized model before epoch 0 and stores it as the
initial best checkpoint.  A value of `-1` therefore means that none of the 800
trained epochs achieved a lower penalized validation-selection score.  The
artifact is valid and loadable, but its policy weights are untrained random
initialization weights; `-1` is not a missing epoch and not an 801st epoch.

## Findings

- Deep Hedging selected the initialized checkpoint in
  12/60 fits
  (20.0%).
- The concentration is 11/
  30 European fits versus
  1/30 Asian fits.
- Mean DH test position-bound occupancy is
  39.0% for European and
  39.3% for Asian liabilities.  Mean fractions
  of paths touching a bound are 66.5% and
  71.7%, respectively.
- Affected fits: asian_call/TLT/seed-2345, european_call/AAPL/seed-2345, european_call/IWM/seed-2345, european_call/QQQ/seed-2345, european_call/SPY/seed-2345, european_call/TLT/seed-1234, european_call/TLT/seed-2345, european_call/XLE/seed-2345, european_call/XLF/seed-1234, european_call/XLF/seed-2345, european_call/XLK/seed-2345, european_call/XLV/seed-2345.

The implied unpenalized validation loss in `dh_checkpoint_audit.csv` is
reconstructed as selected score minus the four configured diagnostic penalties.
The full epoch histories were not serialized into model artifacts, so the old
run cannot be retrospectively reselected under a new rule.

## Benchmark-tuning fairness

Deep Hedging used one fixed paper architecture per seed.  ProtoHedge evaluated
16
validation-only configurations per seed and selected separate mean- and
tail-oriented configurations.  This is test-leakage safe, but it is an
asymmetric model-development budget that must be disclosed and stress-tested.

## Decision

Do not discard the completed run.  It remains valid evidence for the protocol
that was actually executed.  Before a definitive claim that ProtoHedge beats
Deep Hedging, run a newly named, locked sensitivity in which checkpoint
selection uses the unpenalized validation OCE loss and saturation is reported as
a diagnostic rather than folded into the checkpoint score.  Preserve the
original architecture, optimizer, seeds, splits, bounds, and test set.  First
run the affected European DH fits plus matched selected ProtoHedge fits; expand
to the complete comparison if the conclusion changes materially.

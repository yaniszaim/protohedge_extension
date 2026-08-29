# Pre-writing reproducibility decision

## Current status

The completed 20-task historical run is preserved as evidence for the protocol
that was actually executed. Its raw artifacts, panel inputs, source snapshot,
locked configuration, derived original-utility tables, and paired Spot Delta
tables have SHA-256 manifests in this directory.

The main historical result is encouraging, but it is not yet the final
superiority result. ProtoHedge exceeds Deep Hedging on the panel-level original
OCE CVaR@50% utility under the executed protocol. A checkpoint-selection
sensitivity is required before that difference is described as robust because
12 of 60 Deep Hedging fits selected their initialized, untrained policy.

## What `selected_epoch = -1` means

Both the original TensorFlow trainer and the PyTorch trainer evaluate the model
before epoch 0. They save that initialized state as the first checkpoint
candidate. Therefore, `selected_epoch = -1` is a real, loadable checkpoint, not
a missing file and not epoch 801. It means that no trained epoch beat the
initialized model under the configured checkpoint score.

This is not automatically an error. The original ProtoHedge implementation
also allowed initialization to win, using minimum training OCE. It is a concern
in the historical run because its checkpoint score was not pure OCE. It was:

```text
validation OCE
+ 0.005 * mean absolute action
+ 0.010 * mean absolute position
+ 0.100 * fraction of positions at a bound
```

In a constrained hedge, occupying a position bound can be economically
appropriate. Penalizing saturation during checkpoint selection can therefore
prefer a weak near-zero initialized policy over a trained policy that takes
larger, useful positions. Saturation should be reported as a diagnostic, not
silently combined with the paper's utility objective.

## What was preserved and what changed

The real-data Deep Hedging benchmark preserves the original paper's core model:
width 20, depth 3, Softplus activations, Adam at 0.001, 800 epochs, batch size
32, and OCE CVaR@50% (`lambda = 1`).

Real-data evaluation necessarily adds chronological train/validation/test
splits, listed-option paths, cumulative inventory accounting, explicit position
bounds, and a separate validation set. These changes make the benchmark a
real-data adaptation of the paper model rather than a byte-for-byte rerun of
the synthetic experiment. The questionable additional change was putting
action, position, and saturation diagnostics into checkpoint selection.

For the synthetic reproduction, the frozen paper-era TensorFlow source is used
directly and retains its original rule: minimum training OCE, including the
initialized checkpoint. For the historical sensitivity, pure validation OCE is
the appropriate locked rule because a chronological validation set exists and
the test period must remain untouched.

## Audit findings

- Deep Hedging selected initialization in 12/60 fits (20 percent).
- The concentration is 11/30 European fits and 1/30 Asian fits.
- Mean test position-bound occupancy is about 39 percent in both liabilities.
- ProtoHedge selected initialization in 0 selected fits.
- ProtoHedge searched 16 validation-only configurations per seed, while Deep
  Hedging used one fixed paper architecture.
- The tuning asymmetry is test-leakage safe, but it must be disclosed. The
  fixed Deep Hedging architecture is appropriate for testing whether the
  original paper comparison generalizes, not for claiming superiority over an
  optimally tuned neural benchmark.
- Full epoch histories were not serialized in the completed run, so the old
  weights cannot be retrospectively reselected under a new score.

## Locked sensitivity and gate

`scripts/run_checkpoint_selection_sensitivity.py` reruns the 12 affected Deep
Hedging fits and the two originally validation-selected ProtoHedge
configurations for each affected ticker/seed. It changes only checkpoint
selection to unpenalized validation OCE. Initialization remains eligible, and
saturation remains measured. The run consists of 36 restart-safe model fits.

This is a targeted sensitivity of the originally selected configurations. If
the panel conclusion remains materially unchanged, the completed historical
run plus this conservative audit is sufficient. If the conclusion changes, the
entire affected ProtoHedge configuration grid and Deep Hedging comparison must
be rerun under the same locked checkpoint rule before writing the final claim.

## Original synthetic reproduction

`scripts/reproduce_original_synthetic.py` exports source from Git commit
`aedb450606220462d99c7ca333ab51b600c423b1` and reproduces Deep Hedging and
ProtoHedge in Black-Scholes and stochastic-volatility worlds. It records full
training histories, weights, test paths, environment versions, protocol, and
checksums. The paper-facing comparison uses expected OCE CVaR@50% evaluated
with each trained utility threshold `y`, matching Figure 4 and Table 2.

The paper requires Black-Scholes drift 0 and 100 prototypes. Its notebook names
`prototypes_100_nodrift.pkl`, but that asset is absent from every Git revision;
the committed `prototypes_100.pkl` was extracted from the default drift 0.1
world. The reproduction therefore regenerates the missing no-drift prototypes
from the reproduced Deep Hedging checkpoint using the notebook's documented
StandardScaler, KMeans, and medoid procedure. This deviation is explicit and
checksummed. The committed 500-prototype stochastic-volatility asset is used
directly.

## Compute estimate

Local full-data probes on 2026-08-28 measured about 20 to 30 minutes per
historical sensitivity fit and about 5 seconds per original TensorFlow epoch
for Black-Scholes Deep Hedging. The current serial estimate is 12 to 18 hours
for the 36-fit historical sensitivity and 3 to 5 hours for all four synthetic
models. Both runners are resumable; these estimates should not be combined by
running the CPU-heavy workflows concurrently.

## Claim policy before writing

The paper may already state that ProtoHedge was evaluated under a fully fixed,
chronological historical protocol and that the executed results are favorable.
It should not yet state an unconditional robust advantage over Deep Hedging.
That stronger sentence is gated on the checkpoint sensitivity. Claims against
Spot Delta must use the paired block-bootstrap tables and distinguish clear
mean/CVaR improvements from metrics whose confidence intervals include zero.

# Final Confirmation Runbook

## What We Are Running

We are **not** rerunning the 10-ticker experiment. Its raw outputs remain the
main empirical evidence.

The final confirmation has two small, separate parts:

1. **European SPY SoftClip check:** reuse the completed audited Deep Hedging
   result and train two matched ProtoHedge models. One uses the exact original
   TensorFlow Probability SoftClip formula; one uses our historical PyTorch
   approximation. Everything else is held fixed.
2. **Black-Scholes notebook check:** train one original TensorFlow Deep Hedging
   model and one original TensorFlow ProtoHedge model under the configuration
   that produced the authors' saved near-parity result.

The SPY check is expected to take about **54 minutes** locally because prior
matched SPY ProtoHedge fits took about 27 minutes each. The Black-Scholes pair
previously took about **112 minutes** locally. These are estimates, not promises.

## Run European SPY

From `/Users/yaniszaim/deephedging`:

```bash
nohup caffeinate -i /opt/anaconda3/envs/dh/bin/python -u \
  scripts/run_spy_softclip_confirmation.py \
  --device auto \
  > paper/spy_softclip_confirmation.log 2>&1 &
echo $!
```

This command does the following:

- Chooses `K=25` using validation data only, inside the restricted
  vanilla-derived and unweighted ProtoHedge family.
- Reuses the already completed pure-validation Deep Hedging baseline for seed
  `2345`; it does not retrain DH.
- Trains exact-SoftClip ProtoHedge first, then the historical approximation.
- Evaluates both once on the untouched SPY test paths.
- Produces checksums, model metrics, and a paired temporal-block sensitivity
  table.

Watch it without stopping it:

```bash
tail -f paper/spy_softclip_confirmation.log
```

Press `Ctrl-C` to stop watching the log. That does not stop the run.
`caffeinate -i` keeps the Mac awake while the process is active. Closing the
terminal is safe after `nohup` starts; shutting down the computer stops the
current fit.

Check completed-fit status without stopping it:

```bash
/opt/anaconda3/envs/dh/bin/python \
  scripts/run_spy_softclip_confirmation.py \
  --device auto \
  --status-only
```

If the machine restarts, rerun the original command. A completed fit is reused.
Only a fit interrupted before it wrote `complete.json` will restart, so at most
one roughly 27-minute fit is lost.

Main outputs:

- `.deephedging_real_runs/spy_european_softclip_confirmation_v1/SPY_CONFIRMATION_INTERPRETATION.md`
- `.deephedging_real_runs/spy_european_softclip_confirmation_v1/spy_model_metrics.csv`
- `.deephedging_real_runs/spy_european_softclip_confirmation_v1/spy_paired_block_bootstrap.csv`
- `.deephedging_real_runs/spy_european_softclip_confirmation_v1/HISTORICAL_FAMILY_SCOPE.md`

## Run Black-Scholes

Create a clean TensorFlow environment once. This is necessary because the old
`dh` environment currently has a NumPy/TensorFlow binary mismatch:

```bash
bash scripts/setup_original_synthetic_env.sh
```

Then run the two-model confirmation:

```bash
nohup caffeinate -i .venv-original-synthetic/bin/python -u \
  scripts/reproduce_original_synthetic.py \
  --profile notebook-black-scholes \
  > paper/black_scholes_notebook_confirmation.log 2>&1 &
echo $!
```

Watch it:

```bash
tail -f paper/black_scholes_notebook_confirmation.log
```

The command is model-level restartable. If one model finishes, rerunning the
command skips it and starts only the missing model.

Main outputs:

- `paper/black_scholes_notebook_confirmation/synthetic_summary.csv`
- `paper/black_scholes_notebook_confirmation/reproduction_assessment.json`
- `paper/black_scholes_notebook_confirmation/protocol.json`
- `paper/black_scholes_notebook_confirmation/artifact_checksums.sha256`

## One-Command Option

After the TensorFlow environment exists, run both studies sequentially:

```bash
nohup caffeinate -i bash scripts/run_final_confirmation.sh \
  --device auto \
  --with-black-scholes \
  > paper/final_confirmation.log 2>&1 &
echo $!
```

## What Any Outcome Means

- If exact and historical SoftClip are close, report implementation-fidelity
  robustness.
- If they differ, report the sensitivity and call the historical headline
  method the **validation-selected expanded PyTorch ProtoHedge family**, not an
  exact numerical port.
- If the Black-Scholes ordering and near-parity reproduce, that directly answers
  the most likely reviewer concern.
- If Black-Scholes differs, report the authors' archived notebook reproduction,
  our deterministic result, and the stochastic-training/configuration caveat.

No outcome from either small confirmation invalidates the already completed
real-data measurements. It changes the precision and scope of the claims we
write, not whether the empirical study can be reported.

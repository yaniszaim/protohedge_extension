# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta`: mean `-0.019956`, CVaR5 `-0.075825`, shortfall `0.804825`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.020070`, CVaR5 `-0.075484`, shortfall `0.815789`, selected band `0.100000`, bound occupancy `0.000000`
- `unhedged`: mean `-0.020650`, CVaR5 `-0.080852`, shortfall `0.559211`, bound occupancy `0.000000`

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
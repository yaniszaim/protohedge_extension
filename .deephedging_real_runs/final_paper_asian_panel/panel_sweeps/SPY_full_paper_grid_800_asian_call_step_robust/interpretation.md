# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.011982`, CVaR5 `-0.053213`, shortfall `0.952849`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.012040`, CVaR5 `-0.054190`, shortfall `0.852652`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.012060`, CVaR5 `-0.054145`, shortfall `0.856582`, selected band `0.023333`, bound occupancy `0.000000`
- `unhedged`: mean `-0.012198`, CVaR5 `-0.055393`, shortfall `0.685658`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=100_weighted=0_learnw=0` with mean `-0.012015` and CVaR5 `-0.052223`.
- Best downside tail: `proto_source=vanilla_k=10_weighted=0_learnw=0` with CVaR5 `-0.051873` and mean `-0.012583`.
- Lowest shortfall probability: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with shortfall `0.914211`.

Best ProtoHedge mean minus vanilla mean: `-0.000033`.
Best ProtoHedge mean minus unhedged mean: `0.000182`.
Best ProtoHedge mean minus spot-delta mean: `0.000024`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000045`.
Best ProtoHedge bound occupancy: `0.002292`.
Best ProtoHedge path touch rate: `0.004584`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=100_weighted=0_learnw=0` with mean `-0.012015` and bound occupancy `0.002292`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
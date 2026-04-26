# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `0.005633`, CVaR5 `-0.201112`, shortfall `0.375000`, bound occupancy `0.675000`
- `unhedged`: mean `-0.028933`, CVaR5 `-0.073264`, shortfall `0.875000`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.030300`, CVaR5 `-0.078341`, shortfall `1.000000`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=2_weighted=1_learnw=0` with mean `-0.025935` and CVaR5 `-0.058890`.
- Best downside tail: `proto_source=spot_delta_k=2_weighted=1_learnw=0` with CVaR5 `-0.058890` and mean `-0.025935`.
- Lowest shortfall probability: `proto_source=spot_delta_k=2_weighted=1_learnw=0` with shortfall `0.875000`.

Best ProtoHedge mean minus vanilla mean: `-0.031568`.
Best ProtoHedge mean minus unhedged mean: `0.002998`.
Best ProtoHedge mean minus spot-delta mean: `0.004366`.
Best ProtoHedge bound occupancy: `0.000000`.
Best ProtoHedge path touch rate: `0.000000`.
Vanilla bound occupancy: `0.675000`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
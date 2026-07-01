# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta`: mean `-0.013530`, CVaR5 `-0.052928`, shortfall `0.733333`, bound occupancy `0.006667`
- `unhedged`: mean `-0.017248`, CVaR5 `-0.051681`, shortfall `0.533333`, bound occupancy `0.000000`
- `vanilla`: mean `-0.066030`, CVaR5 `-0.332597`, shortfall `0.533333`, bound occupancy `0.875000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.020807` and CVaR5 `-0.069659`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with CVaR5 `-0.069659` and mean `-0.020807`.
- Lowest shortfall probability: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with shortfall `0.600000`.

Best ProtoHedge mean minus vanilla mean: `0.045223`.
Best ProtoHedge mean minus unhedged mean: `-0.003559`.
Best ProtoHedge mean minus spot-delta mean: `-0.007276`.
Best ProtoHedge bound occupancy: `0.450000`.
Best ProtoHedge path touch rate: `1.000000`.
Vanilla bound occupancy: `0.875000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.020807` and bound occupancy `0.450000`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
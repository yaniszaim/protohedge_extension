# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta_band`: mean `-0.012122`, CVaR5 `-0.049570`, shortfall `0.872299`, selected band `0.020000`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.012126`, CVaR5 `-0.049732`, shortfall `0.870334`, bound occupancy `0.000000`
- `unhedged`: mean `-0.012164`, CVaR5 `-0.049544`, shortfall `0.701375`, bound occupancy `0.000000`
- `vanilla`: mean `-0.014012`, CVaR5 `-0.045884`, shortfall `0.994106`, bound occupancy `0.725000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.015437` and CVaR5 `-0.056528`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with CVaR5 `-0.056350` and mean `-0.015437`.
- Lowest shortfall probability: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with shortfall `0.937132`.

Best ProtoHedge mean minus vanilla mean: `-0.001425`.
Best ProtoHedge mean minus unhedged mean: `-0.003273`.
Best ProtoHedge mean minus spot-delta mean: `-0.003310`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `-0.003314`.
Best ProtoHedge bound occupancy: `0.325442`.
Best ProtoHedge path touch rate: `1.000000`.
Vanilla bound occupancy: `0.725000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.015437` and bound occupancy `0.325442`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
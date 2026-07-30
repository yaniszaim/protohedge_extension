# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta_band`: mean `-0.036997`, CVaR5 `-0.170122`, shortfall `0.826729`, selected band `0.010000`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.037105`, CVaR5 `-0.170573`, shortfall `0.825175`, bound occupancy `0.000000`
- `vanilla`: mean `-0.037544`, CVaR5 `-0.167153`, shortfall `0.949495`, bound occupancy `0.166667`
- `unhedged`: mean `-0.038251`, CVaR5 `-0.180119`, shortfall `0.630925`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.036376` and CVaR5 `-0.177241`.
- Best downside tail: `proto_source=vanilla_k=25_weighted=0_learnw=0` with CVaR5 `-0.163549` and mean `-0.037490`.
- Lowest shortfall probability: `proto_source=spot_delta_k=100_weighted=0_learnw=0` with shortfall `0.799534`.

Best ProtoHedge mean minus vanilla mean: `0.001168`.
Best ProtoHedge mean minus unhedged mean: `0.001875`.
Best ProtoHedge mean minus spot-delta mean: `0.000729`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000622`.
Best ProtoHedge bound occupancy: `0.167541`.
Best ProtoHedge path touch rate: `0.562549`.
Vanilla bound occupancy: `0.166667`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.036376` and bound occupancy `0.167541`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
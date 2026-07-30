# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.027113`, CVaR5 `-0.123981`, shortfall `0.945645`, bound occupancy `0.090701`
- `spot_delta_band`: mean `-0.027552`, CVaR5 `-0.129866`, shortfall `0.814669`, selected band `0.083333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.027582`, CVaR5 `-0.130486`, shortfall `0.812705`, bound occupancy `0.000000`
- `unhedged`: mean `-0.027591`, CVaR5 `-0.130607`, shortfall `0.618861`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=1_learnw=0` with mean `-0.026990` and CVaR5 `-0.124727`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with CVaR5 `-0.122507` and mean `-0.027192`.
- Lowest shortfall probability: `proto_source=vanilla_k=50_weighted=1_learnw=0` with shortfall `0.866405`.

Best ProtoHedge mean minus vanilla mean: `0.000122`.
Best ProtoHedge mean minus unhedged mean: `0.000600`.
Best ProtoHedge mean minus spot-delta mean: `0.000592`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000562`.
Best ProtoHedge bound occupancy: `0.100229`.
Best ProtoHedge path touch rate: `0.331369`.
Vanilla bound occupancy: `0.090701`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=1_learnw=0` with mean `-0.026990` and bound occupancy `0.100229`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
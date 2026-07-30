# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.015436`, CVaR5 `-0.075057`, shortfall `0.973150`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.015483`, CVaR5 `-0.075180`, shortfall `0.815324`, selected band `0.083333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.015513`, CVaR5 `-0.075485`, shortfall `0.806156`, bound occupancy `0.000000`
- `unhedged`: mean `-0.015521`, CVaR5 `-0.077279`, shortfall `0.595940`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=100_weighted=1_learnw=0` with mean `-0.015112` and CVaR5 `-0.071476`.
- Best downside tail: `proto_source=spot_delta_k=100_weighted=1_learnw=0` with CVaR5 `-0.071476` and mean `-0.015112`.
- Lowest shortfall probability: `proto_source=vanilla_k=50_weighted=0_learnw=0` with shortfall `0.859201`.

Best ProtoHedge mean minus vanilla mean: `0.000324`.
Best ProtoHedge mean minus unhedged mean: `0.000409`.
Best ProtoHedge mean minus spot-delta mean: `0.000401`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000371`.
Best ProtoHedge bound occupancy: `0.000000`.
Best ProtoHedge path touch rate: `0.000000`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=100_weighted=1_learnw=0` with mean `-0.015112` and bound occupancy `0.000000`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
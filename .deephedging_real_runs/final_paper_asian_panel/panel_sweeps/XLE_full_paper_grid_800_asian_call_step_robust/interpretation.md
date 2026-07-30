# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `unhedged`: mean `-0.015267`, CVaR5 `-0.092833`, shortfall `0.493416`, bound occupancy `0.000000`
- `vanilla`: mean `-0.015382`, CVaR5 `-0.088240`, shortfall `0.968815`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.015430`, CVaR5 `-0.091414`, shortfall `0.765073`, selected band `0.083333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.015522`, CVaR5 `-0.091291`, shortfall `0.756757`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.015263` and CVaR5 `-0.090634`.
- Best downside tail: `proto_source=vanilla_k=25_weighted=1_learnw=0` with CVaR5 `-0.085528` and mean `-0.015683`.
- Lowest shortfall probability: `proto_source=spot_delta_k=100_weighted=1_learnw=0` with shortfall `0.829522`.

Best ProtoHedge mean minus vanilla mean: `0.000119`.
Best ProtoHedge mean minus unhedged mean: `0.000004`.
Best ProtoHedge mean minus spot-delta mean: `0.000259`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000167`.
Best ProtoHedge bound occupancy: `0.002252`.
Best ProtoHedge path touch rate: `0.020790`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.015263` and bound occupancy `0.002252`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
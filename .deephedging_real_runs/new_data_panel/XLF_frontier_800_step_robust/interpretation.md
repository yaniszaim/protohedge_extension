# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta_band`: mean `-0.023646`, CVaR5 `-0.104283`, shortfall `0.812865`, selected band `0.033333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.023677`, CVaR5 `-0.104822`, shortfall `0.815789`, bound occupancy `0.000000`
- `vanilla`: mean `-0.023765`, CVaR5 `-0.103492`, shortfall `1.000000`, bound occupancy `0.000000`
- `unhedged`: mean `-0.023930`, CVaR5 `-0.107268`, shortfall `0.630117`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.023684` and CVaR5 `-0.101902`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.101902` and mean `-0.023684`.
- Lowest shortfall probability: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with shortfall `0.873538`.

Best ProtoHedge mean minus vanilla mean: `0.000081`.
Best ProtoHedge mean minus unhedged mean: `0.000247`.
Best ProtoHedge mean minus spot-delta mean: `-0.000007`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `-0.000038`.
Best ProtoHedge bound occupancy: `0.008772`.
Best ProtoHedge path touch rate: `0.108918`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.023684` and bound occupancy `0.008772`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
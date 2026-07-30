# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.015366`, CVaR5 `-0.062710`, shortfall `0.927963`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.015683`, CVaR5 `-0.064436`, shortfall `0.850033`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.015684`, CVaR5 `-0.064563`, shortfall `0.852652`, selected band `0.066667`, bound occupancy `0.000000`
- `unhedged`: mean `-0.015992`, CVaR5 `-0.067649`, shortfall `0.676490`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with mean `-0.015575` and CVaR5 `-0.058335`.
- Best downside tail: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with CVaR5 `-0.058335` and mean `-0.015575`.
- Lowest shortfall probability: `proto_source=spot_delta_k=100_weighted=1_learnw=0` with shortfall `0.893910`.

Best ProtoHedge mean minus vanilla mean: `-0.000208`.
Best ProtoHedge mean minus unhedged mean: `0.000417`.
Best ProtoHedge mean minus spot-delta mean: `0.000108`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000109`.
Best ProtoHedge bound occupancy: `0.000000`.
Best ProtoHedge path touch rate: `0.000000`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with mean `-0.015575` and bound occupancy `0.000000`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
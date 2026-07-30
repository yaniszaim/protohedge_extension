# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.021682`, CVaR5 `-0.085948`, shortfall `0.929928`, bound occupancy `0.192960`
- `spot_delta`: mean `-0.022007`, CVaR5 `-0.090207`, shortfall `0.857891`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.022028`, CVaR5 `-0.090119`, shortfall `0.857891`, selected band `0.023333`, bound occupancy `0.000000`
- `unhedged`: mean `-0.022165`, CVaR5 `-0.090722`, shortfall `0.707924`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.021665` and CVaR5 `-0.084377`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with CVaR5 `-0.084377` and mean `-0.021665`.
- Lowest shortfall probability: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with shortfall `0.899804`.

Best ProtoHedge mean minus vanilla mean: `0.000016`.
Best ProtoHedge mean minus unhedged mean: `0.000499`.
Best ProtoHedge mean minus spot-delta mean: `0.000341`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000362`.
Best ProtoHedge bound occupancy: `0.037787`.
Best ProtoHedge path touch rate: `0.286182`.
Vanilla bound occupancy: `0.192960`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.021665` and bound occupancy `0.037787`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
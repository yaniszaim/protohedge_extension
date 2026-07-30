# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.014945`, CVaR5 `-0.061751`, shortfall `0.958723`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.015162`, CVaR5 `-0.061191`, shortfall `0.848131`, selected band `0.033333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.015172`, CVaR5 `-0.060948`, shortfall `0.845794`, bound occupancy `0.000000`
- `unhedged`: mean `-0.015334`, CVaR5 `-0.065845`, shortfall `0.644860`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with mean `-0.014761` and CVaR5 `-0.059093`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with CVaR5 `-0.059093` and mean `-0.014761`.
- Lowest shortfall probability: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with shortfall `0.890187`.

Best ProtoHedge mean minus vanilla mean: `0.000185`.
Best ProtoHedge mean minus unhedged mean: `0.000574`.
Best ProtoHedge mean minus spot-delta mean: `0.000411`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000402`.
Best ProtoHedge bound occupancy: `0.000740`.
Best ProtoHedge path touch rate: `0.002336`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with mean `-0.014761` and bound occupancy `0.000740`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
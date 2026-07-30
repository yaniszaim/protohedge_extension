# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `unhedged`: mean `-0.007845`, CVaR5 `-0.045692`, shortfall `0.491433`, bound occupancy `0.000000`
- `vanilla`: mean `-0.007970`, CVaR5 `-0.044874`, shortfall `0.971184`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.008070`, CVaR5 `-0.043702`, shortfall `0.793614`, selected band `0.150000`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.008172`, CVaR5 `-0.043486`, shortfall `0.778816`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.007346` and CVaR5 `-0.043936`.
- Best downside tail: `proto_source=spot_delta_k=100_weighted=0_learnw=0` with CVaR5 `-0.042743` and mean `-0.007386`.
- Lowest shortfall probability: `proto_source=spot_delta_k=50_weighted=1_learnw=0` with shortfall `0.809190`.

Best ProtoHedge mean minus vanilla mean: `0.000624`.
Best ProtoHedge mean minus unhedged mean: `0.000499`.
Best ProtoHedge mean minus spot-delta mean: `0.000826`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000724`.
Best ProtoHedge bound occupancy: `0.002414`.
Best ProtoHedge path touch rate: `0.017913`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.007346` and bound occupancy `0.002414`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
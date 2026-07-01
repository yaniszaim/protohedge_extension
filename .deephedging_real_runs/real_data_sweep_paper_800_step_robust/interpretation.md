# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `1.659881`, CVaR5 `-0.044903`, shortfall `0.027201`, bound occupancy `0.602967`
- `spot_delta_band`: mean `-0.020898`, CVaR5 `-0.090389`, shortfall `0.818497`, selected band `0.026667`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.020930`, CVaR5 `-0.090437`, shortfall `0.818497`, bound occupancy `0.011771`
- `unhedged`: mean `-0.021229`, CVaR5 `-0.095208`, shortfall `0.651335`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `1.646594` and CVaR5 `-0.056544`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.029352` and mean `1.638294`.
- Lowest shortfall probability: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with shortfall `0.022255`.

Best ProtoHedge mean minus vanilla mean: `-0.013287`.
Best ProtoHedge mean minus unhedged mean: `1.667822`.
Best ProtoHedge mean minus spot-delta mean: `1.667523`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `1.667491`.
Best ProtoHedge bound occupancy: `0.317099`.
Best ProtoHedge path touch rate: `1.000000`.
Vanilla bound occupancy: `0.602967`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `1.646594` and bound occupancy `0.317099`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
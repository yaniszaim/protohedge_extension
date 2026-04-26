# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `5.333175`, CVaR5 `0.486808`, shortfall `0.010386`
- `spot_delta`: mean `-0.014690`, CVaR5 `-0.128939`, shortfall `0.962908`
- `unhedged`: mean `-0.020197`, CVaR5 `-0.092059`, shortfall `0.646884`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=1_learnw=0` with mean `5.236279` and CVaR5 `0.503272`.
- Best downside tail: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with CVaR5 `0.662364` and mean `5.219158`.
- Lowest shortfall probability: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with shortfall `0.002967`.

Best ProtoHedge mean minus vanilla mean: `-0.096897`.
Best ProtoHedge mean minus unhedged mean: `5.256476`.
Best ProtoHedge mean minus spot-delta mean: `5.250969`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
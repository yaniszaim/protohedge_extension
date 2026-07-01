# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `4.849409`, CVaR5 `-0.625266`, shortfall `0.065282`, bound occupancy `0.805935`
- `spot_delta`: mean `-0.019698`, CVaR5 `-0.084592`, shortfall `0.808605`, bound occupancy `0.000000`
- `unhedged`: mean `-0.020197`, CVaR5 `-0.092059`, shortfall `0.646884`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with mean `4.680333` and CVaR5 `-1.004846`.
- Best downside tail: `proto_source=vanilla_k=100_weighted=0_learnw=0` with CVaR5 `-0.344575` and mean `4.148735`.
- Lowest shortfall probability: `proto_source=vanilla_k=100_weighted=1_learnw=0` with shortfall `0.047478`.

Best ProtoHedge mean minus vanilla mean: `-0.169075`.
Best ProtoHedge mean minus unhedged mean: `4.700530`.
Best ProtoHedge mean minus spot-delta mean: `4.700031`.
Best ProtoHedge bound occupancy: `0.462092`.
Best ProtoHedge path touch rate: `1.000000`.
Vanilla bound occupancy: `0.805935`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta`: mean `-0.011689`, CVaR5 `-0.042338`, shortfall `1.000000`
- `unhedged`: mean `-0.014064`, CVaR5 `-0.058960`, shortfall `0.500000`
- `vanilla`: mean `-0.274691`, CVaR5 `-1.214135`, shortfall `0.625000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=zero_k=2_weighted=1_learnw=0` with mean `-0.033171` and CVaR5 `-0.125186`.
- Best downside tail: `proto_source=zero_k=2_weighted=1_learnw=0` with CVaR5 `-0.125186` and mean `-0.033171`.
- Lowest shortfall probability: `proto_source=zero_k=2_weighted=1_learnw=0` with shortfall `0.500000`.

Best ProtoHedge mean minus vanilla mean: `0.241520`.
Best ProtoHedge mean minus unhedged mean: `-0.019107`.
Best ProtoHedge mean minus spot-delta mean: `-0.021482`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
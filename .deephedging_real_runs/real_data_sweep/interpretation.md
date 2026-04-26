# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `0.011958`, CVaR5 `-0.183436`, shortfall `0.410256`
- `spot_delta`: mean `-0.011322`, CVaR5 `-0.078742`, shortfall `0.987179`
- `unhedged`: mean `-0.022705`, CVaR5 `-0.103893`, shortfall `0.705128`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with mean `0.071614` and CVaR5 `-0.127764`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.048359` and mean `0.018853`.
- Lowest shortfall probability: `proto_source=vanilla_k=10_weighted=1_learnw=0` with shortfall `0.153846`.

Best ProtoHedge mean minus vanilla mean: `0.059656`.
Best ProtoHedge mean minus unhedged mean: `0.094319`.
Best ProtoHedge mean minus spot-delta mean: `0.082936`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
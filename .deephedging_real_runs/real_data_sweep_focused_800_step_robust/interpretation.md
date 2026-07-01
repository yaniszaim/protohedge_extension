# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `1.638817`, CVaR5 `-0.123911`, shortfall `0.038576`, bound occupancy `0.691840`
- `spot_delta`: mean `-0.019698`, CVaR5 `-0.084592`, shortfall `0.808605`, bound occupancy `0.011573`
- `unhedged`: mean `-0.020197`, CVaR5 `-0.092059`, shortfall `0.646884`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=100_weighted=1_learnw=0` with mean `1.650981` and CVaR5 `-0.119677`.
- Best downside tail: `proto_source=vanilla_k=50_weighted=1_learnw=0` with CVaR5 `-0.058914` and mean `1.468957`.
- Lowest shortfall probability: `proto_source=vanilla_k=25_weighted=0_learnw=0` with shortfall `0.029674`.

Best ProtoHedge mean minus vanilla mean: `0.012164`.
Best ProtoHedge mean minus unhedged mean: `1.671178`.
Best ProtoHedge mean minus spot-delta mean: `1.670679`.
Best ProtoHedge bound occupancy: `0.304674`.
Best ProtoHedge path touch rate: `1.000000`.
Vanilla bound occupancy: `0.691840`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=100_weighted=1_learnw=0` with mean `1.650981` and bound occupancy `0.304674`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
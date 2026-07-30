# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `unhedged`: mean `-0.028477`, CVaR5 `-0.176785`, shortfall `0.506584`, bound occupancy `0.000000`
- `vanilla`: mean `-0.028560`, CVaR5 `-0.170521`, shortfall `1.000000`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.028640`, CVaR5 `-0.175639`, shortfall `0.765766`, selected band `0.083333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.028732`, CVaR5 `-0.175576`, shortfall `0.763687`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=vanilla_k=100_weighted=0_learnw=0` with mean `-0.028505` and CVaR5 `-0.169580`.
- Best downside tail: `proto_source=vanilla_k=10_weighted=0_learnw=0` with CVaR5 `-0.168814` and mean `-0.028727`.
- Lowest shortfall probability: `proto_source=spot_delta_k=50_weighted=1_learnw=0` with shortfall `0.733888`.

Best ProtoHedge mean minus vanilla mean: `0.000055`.
Best ProtoHedge mean minus unhedged mean: `-0.000028`.
Best ProtoHedge mean minus spot-delta mean: `0.000227`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000135`.
Best ProtoHedge bound occupancy: `0.000329`.
Best ProtoHedge path touch rate: `0.011088`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=vanilla_k=100_weighted=0_learnw=0` with mean `-0.028505` and bound occupancy `0.000329`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
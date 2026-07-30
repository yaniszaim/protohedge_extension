# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta_band`: mean `-0.013059`, CVaR5 `-0.061926`, shortfall `0.804094`, selected band `0.033333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.013090`, CVaR5 `-0.062577`, shortfall `0.799708`, bound occupancy `0.000000`
- `vanilla`: mean `-0.013256`, CVaR5 `-0.064256`, shortfall `0.924708`, bound occupancy `0.000000`
- `unhedged`: mean `-0.013344`, CVaR5 `-0.066261`, shortfall `0.600877`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.013181` and CVaR5 `-0.062390`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.061647` and mean `-0.013348`.
- Lowest shortfall probability: `proto_source=vanilla_k=50_weighted=1_learnw=0` with shortfall `0.875000`.

Best ProtoHedge mean minus vanilla mean: `0.000075`.
Best ProtoHedge mean minus unhedged mean: `0.000163`.
Best ProtoHedge mean minus spot-delta mean: `-0.000090`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `-0.000122`.
Best ProtoHedge bound occupancy: `0.000037`.
Best ProtoHedge path touch rate: `0.001462`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.013181` and bound occupancy `0.000037`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
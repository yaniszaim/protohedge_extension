# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.028807`, CVaR5 `-0.110404`, shortfall `0.941061`, bound occupancy `0.165963`
- `spot_delta`: mean `-0.029252`, CVaR5 `-0.112778`, shortfall `0.861821`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.029253`, CVaR5 `-0.112823`, shortfall `0.860511`, selected band `0.066667`, bound occupancy `0.000000`
- `unhedged`: mean `-0.029561`, CVaR5 `-0.115993`, shortfall `0.712508`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.028631` and CVaR5 `-0.106865`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.106865` and mean `-0.028631`.
- Lowest shortfall probability: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with shortfall `0.890635`.

Best ProtoHedge mean minus vanilla mean: `0.000176`.
Best ProtoHedge mean minus unhedged mean: `0.000930`.
Best ProtoHedge mean minus spot-delta mean: `0.000621`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000622`.
Best ProtoHedge bound occupancy: `0.030435`.
Best ProtoHedge path touch rate: `0.321546`.
Vanilla bound occupancy: `0.165963`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.028631` and bound occupancy `0.030435`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
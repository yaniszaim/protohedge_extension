# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.028783`, CVaR5 `-0.110270`, shortfall `0.915521`, bound occupancy `0.186182`
- `spot_delta`: mean `-0.029252`, CVaR5 `-0.112778`, shortfall `0.861821`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.029253`, CVaR5 `-0.112823`, shortfall `0.860511`, selected band `0.066667`, bound occupancy `0.000000`
- `unhedged`: mean `-0.029561`, CVaR5 `-0.115993`, shortfall `0.712508`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.028884` and CVaR5 `-0.108852`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.108852` and mean `-0.028884`.
- Lowest shortfall probability: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with shortfall `0.905697`.

Best ProtoHedge mean minus vanilla mean: `-0.000101`.
Best ProtoHedge mean minus unhedged mean: `0.000677`.
Best ProtoHedge mean minus spot-delta mean: `0.000369`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000369`.
Best ProtoHedge bound occupancy: `0.003847`.
Best ProtoHedge path touch rate: `0.077931`.
Vanilla bound occupancy: `0.186182`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.028884` and bound occupancy `0.003847`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
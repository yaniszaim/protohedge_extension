# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta_band`: mean `-0.071915`, CVaR5 `-0.332653`, shortfall `0.819736`, selected band `0.010000`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.072023`, CVaR5 `-0.332719`, shortfall `0.821290`, bound occupancy `0.000000`
- `vanilla`: mean `-0.072375`, CVaR5 `-0.326534`, shortfall `0.990676`, bound occupancy `0.174398`
- `unhedged`: mean `-0.073169`, CVaR5 `-0.342731`, shortfall `0.632479`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with mean `-0.069880` and CVaR5 `-0.324399`.
- Best downside tail: `proto_source=vanilla_k=25_weighted=1_learnw=0` with CVaR5 `-0.320364` and mean `-0.071309`.
- Lowest shortfall probability: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with shortfall `0.831391`.

Best ProtoHedge mean minus vanilla mean: `0.002495`.
Best ProtoHedge mean minus unhedged mean: `0.003289`.
Best ProtoHedge mean minus spot-delta mean: `0.002143`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.002035`.
Best ProtoHedge bound occupancy: `0.275447`.
Best ProtoHedge path touch rate: `0.930070`.
Vanilla bound occupancy: `0.174398`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with mean `-0.069880` and bound occupancy `0.275447`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
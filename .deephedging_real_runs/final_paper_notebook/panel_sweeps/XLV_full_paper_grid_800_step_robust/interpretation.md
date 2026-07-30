# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.018417`, CVaR5 `-0.086612`, shortfall `1.000000`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.018440`, CVaR5 `-0.085206`, shortfall `0.792645`, selected band `0.020000`, bound occupancy `0.000000`
- `unhedged`: mean `-0.018445`, CVaR5 `-0.087943`, shortfall `0.571205`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.018449`, CVaR5 `-0.085164`, shortfall `0.796557`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with mean `-0.018224` and CVaR5 `-0.083131`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with CVaR5 `-0.082658` and mean `-0.018228`.
- Lowest shortfall probability: `proto_source=vanilla_k=100_weighted=1_learnw=0` with shortfall `0.833333`.

Best ProtoHedge mean minus vanilla mean: `0.000194`.
Best ProtoHedge mean minus unhedged mean: `0.000222`.
Best ProtoHedge mean minus spot-delta mean: `0.000225`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000217`.
Best ProtoHedge bound occupancy: `0.005869`.
Best ProtoHedge path touch rate: `0.129108`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with mean `-0.018224` and bound occupancy `0.005869`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
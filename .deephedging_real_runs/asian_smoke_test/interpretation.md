# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `unhedged`: mean `-0.011168`, CVaR5 `-0.040930`, shortfall `0.692308`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.011460`, CVaR5 `-0.035914`, shortfall `0.871795`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.011460`, CVaR5 `-0.035914`, shortfall `0.871795`, selected band `0.000000`, bound occupancy `0.000000`
- `vanilla`: mean `-0.013408`, CVaR5 `-0.037515`, shortfall `0.948718`, bound occupancy `0.775000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.013866` and CVaR5 `-0.046613`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with CVaR5 `-0.046613` and mean `-0.013866`.
- Lowest shortfall probability: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with shortfall `0.846154`.

Best ProtoHedge mean minus vanilla mean: `-0.000458`.
Best ProtoHedge mean minus unhedged mean: `-0.002698`.
Best ProtoHedge mean minus spot-delta mean: `-0.002406`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `-0.002406`.
Best ProtoHedge bound occupancy: `0.449359`.
Best ProtoHedge path touch rate: `1.000000`.
Vanilla bound occupancy: `0.775000`.
No ProtoHedge configuration passed the current robustness screen.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
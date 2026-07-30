# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta_band`: mean `-0.071915`, CVaR5 `-0.332653`, shortfall `0.819736`, selected band `0.010000`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.072023`, CVaR5 `-0.332719`, shortfall `0.821290`, bound occupancy `0.000000`
- `vanilla`: mean `-0.073166`, CVaR5 `-0.327771`, shortfall `0.982906`, bound occupancy `0.333566`
- `unhedged`: mean `-0.073169`, CVaR5 `-0.342731`, shortfall `0.632479`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.069547` and CVaR5 `-0.339599`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with CVaR5 `-0.339599` and mean `-0.069547`.
- Lowest shortfall probability: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with shortfall `0.790210`.

Best ProtoHedge mean minus vanilla mean: `0.003619`.
Best ProtoHedge mean minus unhedged mean: `0.003622`.
Best ProtoHedge mean minus spot-delta mean: `0.002476`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.002368`.
Best ProtoHedge bound occupancy: `0.324009`.
Best ProtoHedge path touch rate: `0.984460`.
Vanilla bound occupancy: `0.333566`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.069547` and bound occupancy `0.324009`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
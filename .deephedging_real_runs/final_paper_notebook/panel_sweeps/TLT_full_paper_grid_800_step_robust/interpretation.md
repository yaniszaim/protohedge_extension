# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `unhedged`: mean `-0.014193`, CVaR5 `-0.083402`, shortfall `0.502336`, bound occupancy `0.000000`
- `vanilla`: mean `-0.014295`, CVaR5 `-0.081935`, shortfall `0.997664`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.014418`, CVaR5 `-0.082353`, shortfall `0.778037`, selected band `0.150000`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.014521`, CVaR5 `-0.082107`, shortfall `0.774922`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.013392` and CVaR5 `-0.082719`.
- Best downside tail: `proto_source=vanilla_k=25_weighted=1_learnw=0` with CVaR5 `-0.081149` and mean `-0.014396`.
- Lowest shortfall probability: `proto_source=vanilla_k=100_weighted=1_learnw=0` with shortfall `0.811526`.

Best ProtoHedge mean minus vanilla mean: `0.000903`.
Best ProtoHedge mean minus unhedged mean: `0.000802`.
Best ProtoHedge mean minus spot-delta mean: `0.001129`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.001027`.
Best ProtoHedge bound occupancy: `0.001577`.
Best ProtoHedge path touch rate: `0.024143`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with mean `-0.013392` and bound occupancy `0.001577`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.021681`, CVaR5 `-0.085902`, shortfall `0.927963`, bound occupancy `0.192665`
- `spot_delta`: mean `-0.022007`, CVaR5 `-0.090207`, shortfall `0.857891`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.022028`, CVaR5 `-0.090119`, shortfall `0.857891`, selected band `0.023333`, bound occupancy `0.000000`
- `unhedged`: mean `-0.022165`, CVaR5 `-0.090722`, shortfall `0.707924`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.021620` and CVaR5 `-0.083971`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.083971` and mean `-0.021620`.
- Lowest shortfall probability: `proto_source=spot_delta_k=10_weighted=1_learnw=0` with shortfall `0.888671`.

Best ProtoHedge mean minus vanilla mean: `0.000061`.
Best ProtoHedge mean minus unhedged mean: `0.000545`.
Best ProtoHedge mean minus spot-delta mean: `0.000387`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000408`.
Best ProtoHedge bound occupancy: `0.000147`.
Best ProtoHedge path touch rate: `0.002620`.
Vanilla bound occupancy: `0.192665`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.021620` and bound occupancy `0.000147`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
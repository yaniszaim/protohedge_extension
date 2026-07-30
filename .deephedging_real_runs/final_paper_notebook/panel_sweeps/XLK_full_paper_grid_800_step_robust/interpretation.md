# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.028089`, CVaR5 `-0.109331`, shortfall `0.989097`, bound occupancy `0.026382`
- `spot_delta_band`: mean `-0.028292`, CVaR5 `-0.109714`, shortfall `0.837227`, selected band `0.033333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.028301`, CVaR5 `-0.109241`, shortfall `0.841121`, bound occupancy `0.000000`
- `unhedged`: mean `-0.028463`, CVaR5 `-0.113481`, shortfall `0.660436`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.027727` and CVaR5 `-0.107290`.
- Best downside tail: `proto_source=spot_delta_k=100_weighted=0_learnw=0` with CVaR5 `-0.106429` and mean `-0.027883`.
- Lowest shortfall probability: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with shortfall `0.860592`.

Best ProtoHedge mean minus vanilla mean: `0.000362`.
Best ProtoHedge mean minus unhedged mean: `0.000736`.
Best ProtoHedge mean minus spot-delta mean: `0.000574`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000564`.
Best ProtoHedge bound occupancy: `0.006192`.
Best ProtoHedge path touch rate: `0.081776`.
Vanilla bound occupancy: `0.026382`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.027727` and bound occupancy `0.006192`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
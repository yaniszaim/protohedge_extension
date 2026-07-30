# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.027966`, CVaR5 `-0.108312`, shortfall `0.993769`, bound occupancy `0.017348`
- `spot_delta_band`: mean `-0.028292`, CVaR5 `-0.109714`, shortfall `0.837227`, selected band `0.033333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.028301`, CVaR5 `-0.109241`, shortfall `0.841121`, bound occupancy `0.000000`
- `unhedged`: mean `-0.028463`, CVaR5 `-0.113481`, shortfall `0.660436`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.028106` and CVaR5 `-0.108084`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.108084` and mean `-0.028106`.
- Lowest shortfall probability: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with shortfall `0.868380`.

Best ProtoHedge mean minus vanilla mean: `-0.000140`.
Best ProtoHedge mean minus unhedged mean: `0.000358`.
Best ProtoHedge mean minus spot-delta mean: `0.000195`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000186`.
Best ProtoHedge bound occupancy: `0.000039`.
Best ProtoHedge path touch rate: `0.000779`.
Vanilla bound occupancy: `0.017348`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.028106` and bound occupancy `0.000039`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
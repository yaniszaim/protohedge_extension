# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `unhedged`: mean `-0.028477`, CVaR5 `-0.176785`, shortfall `0.506584`, bound occupancy `0.000000`
- `vanilla`: mean `-0.028568`, CVaR5 `-0.170016`, shortfall `1.000000`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.028640`, CVaR5 `-0.175639`, shortfall `0.765766`, selected band `0.083333`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.028732`, CVaR5 `-0.175576`, shortfall `0.763687`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.028324` and CVaR5 `-0.172100`.
- Best downside tail: `proto_source=spot_delta_k=10_weighted=0_learnw=0` with CVaR5 `-0.170289` and mean `-0.028737`.
- Lowest shortfall probability: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with shortfall `0.817741`.

Best ProtoHedge mean minus vanilla mean: `0.000244`.
Best ProtoHedge mean minus unhedged mean: `0.000153`.
Best ProtoHedge mean minus spot-delta mean: `0.000408`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000316`.
Best ProtoHedge bound occupancy: `0.010031`.
Best ProtoHedge path touch rate: `0.081774`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.028324` and bound occupancy `0.010031`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
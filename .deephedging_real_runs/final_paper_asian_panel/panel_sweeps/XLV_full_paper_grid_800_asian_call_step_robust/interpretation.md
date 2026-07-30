# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `vanilla`: mean `-0.010329`, CVaR5 `-0.052205`, shortfall `0.944444`, bound occupancy `0.000000`
- `spot_delta_band`: mean `-0.010386`, CVaR5 `-0.051206`, shortfall `0.798122`, selected band `0.020000`, bound occupancy `0.000000`
- `unhedged`: mean `-0.010391`, CVaR5 `-0.054886`, shortfall `0.568858`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.010395`, CVaR5 `-0.051183`, shortfall `0.797340`, bound occupancy `0.000000`

## Best ProtoHedge Configurations

- Best mean: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.010140` and CVaR5 `-0.049266`.
- Best downside tail: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with CVaR5 `-0.049266` and mean `-0.010140`.
- Lowest shortfall probability: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with shortfall `0.845070`.

Best ProtoHedge mean minus vanilla mean: `0.000189`.
Best ProtoHedge mean minus unhedged mean: `0.000252`.
Best ProtoHedge mean minus spot-delta mean: `0.000255`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000247`.
Best ProtoHedge bound occupancy: `0.001174`.
Best ProtoHedge path touch rate: `0.009390`.
Vanilla bound occupancy: `0.000000`.
Best screen-passing ProtoHedge: `proto_source=spot_delta_k=25_weighted=0_learnw=0` with mean `-0.010140` and bound occupancy `0.001174`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
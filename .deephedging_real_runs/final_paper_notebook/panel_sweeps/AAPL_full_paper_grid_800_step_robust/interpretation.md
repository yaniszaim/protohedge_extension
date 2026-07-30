# Real-Data ProtoHedge Sweep Interpretation

This report is generated from `sweep_metrics.csv`.

## Baselines

- `spot_delta_band`: mean `-0.038764`, CVaR5 `-0.149815`, shortfall `0.818713`, selected band `0.100000`, bound occupancy `0.000000`
- `spot_delta`: mean `-0.038768`, CVaR5 `-0.149254`, shortfall `0.812135`, bound occupancy `0.000000`
- `unhedged`: mean `-0.039023`, CVaR5 `-0.153983`, shortfall `0.616959`, bound occupancy `0.000000`
- `vanilla`: mean `-0.039628`, CVaR5 `-0.147927`, shortfall `1.000000`, bound occupancy `0.125037`

## Best ProtoHedge Configurations

- Best mean: `proto_source=vanilla_k=50_weighted=0_learnw=0` with mean `-0.037874` and CVaR5 `-0.150609`.
- Best downside tail: `proto_source=vanilla_k=10_weighted=0_learnw=0` with CVaR5 `-0.146371` and mean `-0.039017`.
- Lowest shortfall probability: `proto_source=spot_delta_k=50_weighted=0_learnw=0` with shortfall `0.839181`.

Best ProtoHedge mean minus vanilla mean: `0.001753`.
Best ProtoHedge mean minus unhedged mean: `0.001149`.
Best ProtoHedge mean minus spot-delta mean: `0.000894`.
Best ProtoHedge mean minus tuned spot-delta-band mean: `0.000890`.
Best ProtoHedge bound occupancy: `0.064876`.
Best ProtoHedge path touch rate: `0.279240`.
Vanilla bound occupancy: `0.125037`.
Best screen-passing ProtoHedge: `proto_source=vanilla_k=50_weighted=0_learnw=0` with mean `-0.037874` and bound occupancy `0.064876`.

## Larger Picture

This sweep is the empirical bridge between the PyTorch port and the paper extension. A positive result means more than training code running: it means a compact prototype policy can compete with neural Deep Hedging and simple classical hedges on held-out historical paths while remaining inspectable.

Treat small pilot sweeps as hypothesis generators. A paper-level claim requires full-data runs, multiple seeds, and robustness checks across prototype counts and market regimes.
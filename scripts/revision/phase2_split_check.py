#!/usr/bin/env python3
"""Phase 2 follow-up: is the gap to the liability delta hedge an optimisation
failure or a train/test shift?

Evaluates, on the TRAIN and VALIDATION episodes of every task, the training
objective (CVaR@50% of the premium-excluded offset) and the outcome dispersion
for: the liability delta hedge, the best constant position, the hybrid Deep
Hedging fits and the hybrid tail-selected ProtoHedge fits. No training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_diagnostics as p2  # noqa: E402


def main():
    from deephedging.real_data_analysis_torch import load_model_artifact
    from deephedging.real_data_sweep_torch import evaluate_gym, world_cfg
    from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch

    rows = []
    for liability in p2.LIABILITIES:
        for ticker in p2.TICKERS:
            tdir = p2.task_dir(liability, ticker)
            selected = pd.read_csv(tdir / "validation_selected_models.csv").set_index("selection")
            names = {"dh_hybrid": "vanilla",
                     "ph_tail_hybrid": selected.loc["best_screened_proto_cvar05", "model"]}
            art_root = tdir / "model_artifacts"
            meta0 = json.load(open(art_root / f"seed_{p2.SEEDS[0]}" / "risk_cvar" / "vanilla" / "artifact_metadata.json"))
            data_path, world_kwargs, splits = meta0["data_path"], dict(meta0["world_kwargs"]), meta0["split_indices"]
            train_world = RealWorld_Spot_ATM_Torch(world_cfg(data_path=data_path, indices=splits["train"], **world_kwargs))
            static = p2.select_static_positions(train_world.data.market)
            for split in ("train", "val", "test"):
                world = RealWorld_Spot_ATM_Torch(world_cfg(data_path=data_path, indices=splits[split], **world_kwargs))
                n = world.nSamples
                premium, liab_delta = p2.liability_premium_and_delta(world)
                spot_all = np.asarray(world.details.spot_all, dtype=np.float64)
                base = {"liability": liability, "ticker": ticker, "split": split, "n": n,
                        "mean_spot_return": float((spot_all[:, -1] / spot_all[:, 0] - 1).mean())}
                tg = np.zeros((n, p2.N_STEPS, 2))
                tg[:, :, 0] = liab_delta
                rule_targets = {"unhedged": p2.constant_targets(n, 0, 0), "liability_delta": tg,
                                "hold_share": p2.constant_targets(n, 1, 0),
                                "static_oce": p2.constant_targets(n, *static["static_oce"])}
                for name, targets in rule_targets.items():
                    sim = p2.simulate_targets(world.data.market, targets)
                    rows.append({**base, "model": name, "seed": -1, **p2.outcome_metrics(sim["offset"], premium)})
                for key, model_name in names.items():
                    for seed in p2.SEEDS:
                        adir = art_root / f"seed_{seed}" / "risk_cvar" / p2.slug(model_name)
                        alt = (p2.SENS / liability / ticker / f"seed_{seed}" / p2.slug(model_name) / "artifact"
                               / f"seed_{seed}" / "risk_cvar" / p2.slug(model_name))
                        if (alt / "gym_state.pt").exists():
                            adir = alt
                        bundle = load_model_artifact(adir)
                        _, result, _ = evaluate_gym(bundle["gym"], data_path, splits[split], key, world_kwargs)
                        offset = result["liability_offset"].numpy().astype(np.float64).reshape(-1)
                        rows.append({**base, "model": key, "seed": seed, **p2.outcome_metrics(offset, premium)})
            print(liability, ticker, "done", flush=True)
    df = pd.DataFrame(rows)
    num = ["offset_mean", "offset_std", "offset_cvar50", "offset_cvar05", "he_mean", "he_std", "mean_spot_return"]
    by_ticker = df.groupby(["liability", "split", "ticker", "model"], sort=False)[num].mean().reset_index()
    by_ticker.to_csv(p2.OUT / "split_check_by_ticker.csv", index=False)
    panel = by_ticker.groupby(["liability", "split", "model"], sort=False)[num].mean().reset_index()
    panel.to_csv(p2.OUT / "split_check_panel.csv", index=False)
    wins = by_ticker.pivot_table(index=["liability", "split", "ticker"], columns="model", values="offset_cvar50")
    for m in ("dh_hybrid", "ph_tail_hybrid", "static_oce"):
        w = (wins[m] > wins["liability_delta"]).groupby(level=[0, 1]).sum()
        print(f"tickers where {m} beats liability_delta on CVaR50:\n{w}")
    pd.set_option("display.width", 200)
    print(panel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

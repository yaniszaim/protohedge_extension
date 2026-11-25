#!/usr/bin/env python3
"""Prototype explanations for a trained ProtoHedge cell (problem P10).

1. A worked decomposition of one historical decision: the state, the top
   prototypes (each a real training-day state with its date), their weights and
   attached outputs, the reconstructed output, and the position actually held.
2. Fidelity over all decisions of a split: how much of the output the top-k
   prototypes reproduce, how many prototypes a decision uses, how often the
   position limits change the output.

The episode for the worked example is chosen by a fixed rule, not by eye: the
episode with the median hedging error under the ProtoHedge policy, at the
middle decision step (step 10 of 20).

    python scripts/revision/rev_explain.py --study S --ticker SPY --liability european_call \
        --mode target --objective cvar10_he --state s0 --seed 0 --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import phase3_fit  # noqa: E402

OUTPUT_MEANING = {"trade": "trade", "target": "position to hold", "residual": "correction to the delta hedge"}


def load_ph(cell: Path, train_ep, state, family="ph"):
    scaler = rc.Scaler(train_ep, state)
    bank = np.load(cell / family / "prototypes.npz", allow_pickle=False)
    policy, _ = rc.load_policy(cell, family, scaler)
    return policy, scaler, bank


def topk_reconstruction(weights, proto_out, k):
    """Output rebuilt from the k heaviest prototypes, weights renormalised."""
    order = np.argsort(-weights, axis=-1)[..., :k]
    w = np.take_along_axis(weights, order, axis=-1)
    w = w / w.sum(axis=-1, keepdims=True)
    return np.einsum("...k,...kd->...d", w, proto_out[order])


def explain(args):
    a = {k: getattr(args, k) for k in ("ticker", "liability", "mode", "objective", "state", "seed")}
    cell = phase3_fit.cell_dir(args.study, a)
    train_ep = rc.Episodes(a["ticker"], a["liability"], "train")
    ep = train_ep if args.split == "train" else rc.Episodes(a["ticker"], a["liability"], args.split,
                                                            allow_test=args.allow_test)
    policy, scaler, bank = load_ph(cell, train_ep, a["state"], args.family)
    ev = rc.evaluate_policy(policy, ep, a["state"], a["mode"])
    weights, outputs, positions = ev["weights"], ev["outputs"], ev["positions"]
    proto_out = rc.prototype_outputs(policy)                                        # [K, 2]

    # ---- fidelity over every decision of the split
    full = np.einsum("ntk,kd->ntd", weights, proto_out)
    assert np.allclose(full, outputs, atol=1e-5)
    scale = float(outputs.std())
    fidelity = {"decisions": int(weights.shape[0] * weights.shape[1]), "output_std": scale}
    for k in (1, 3, 5):
        err = np.abs(topk_reconstruction(weights, proto_out, k) - outputs)
        fidelity[f"top{k}_mean_abs_error"] = float(err.mean())
        fidelity[f"top{k}_p95_abs_error"] = float(np.quantile(err, 0.95))
        fidelity[f"top{k}_weight"] = float(np.sort(weights, axis=-1)[..., -k:].sum(axis=-1).mean())
    entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(axis=-1)
    fidelity["effective_prototypes_mean"] = float(np.exp(entropy).mean())
    fidelity["prototypes_ever_top1"] = int(len(np.unique(weights.argmax(axis=-1))))
    fidelity["prototypes_total"] = int(weights.shape[-1])
    prev = np.concatenate([np.zeros_like(positions[:, :1]), positions[:, :-1]], axis=1)
    if a["mode"] == "trade":
        intended = prev + outputs
    elif a["mode"] == "target":
        intended = outputs
    else:
        base = np.stack([ep.np["liab_delta"], np.zeros_like(ep.np["liab_delta"])], axis=2)
        intended = base + outputs
    fidelity["share_changed_by_limits"] = float(np.mean(np.abs(intended - positions) > 1e-4))

    # ---- do the heaviest prototypes resemble the state? (distance in standardised input space)
    states = np.concatenate([ep.exogenous(a["state"]), prev], axis=2).reshape(-1, len(scaler.mean))
    scaled = (states - scaler.mean) / scaler.std
    protos = bank["prototypes_scaled"].astype(np.float64)
    dist = np.sqrt(((scaled[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2))        # [decisions, K]
    rank = dist.argsort(axis=1).argsort(axis=1)                                         # 0 = nearest
    top1 = weights.reshape(-1, weights.shape[-1]).argmax(axis=1)
    rows_idx = np.arange(len(top1))
    fidelity["top1_is_nearest_share"] = float(np.mean(rank[rows_idx, top1] == 0))
    fidelity["top1_among_5_nearest_share"] = float(np.mean(rank[rows_idx, top1] < 5))
    fidelity["top1_median_distance_rank"] = float(np.median(rank[rows_idx, top1]) + 1)
    fidelity["top1_distance_over_nearest_distance"] = float(
        np.median(dist[rows_idx, top1] / np.maximum(dist.min(axis=1), 1e-9)))

    # ---- worked example, chosen by the fixed rule
    he = ev["offset"] + ep.np["premium"]
    i = int(np.argsort(he)[len(he) // 2])
    t = args.step
    names = scaler.names
    state_raw = np.concatenate([ep.exogenous(a["state"])[i, t], prev[i, t]])
    order = np.argsort(-weights[i, t])[:args.top]
    rows = []
    for rank, kx in enumerate(order, 1):
        rows.append({
            "rank": rank, "prototype": int(kx), "weight": float(weights[i, t, kx]),
            "source_start_date": str(bank["source_start_date"][kx]), "source_step": int(bank["source_step"][kx]),
            "state": {n: float(v) for n, v in zip(names, bank["prototypes_raw"][kx])},
            "output_spot": float(proto_out[kx, 0]), "output_call": float(proto_out[kx, 1]),
        })
    example = {
        "rule": "median hedging-error episode, decision step %d" % t,
        "episode_start_date": str(ep.start_date[i]), "episode_end_date": str(ep.end_date[i]),
        "hedging_error": float(he[i]), "state": {n: float(v) for n, v in zip(names, state_raw)},
        "output_meaning": OUTPUT_MEANING[a["mode"]],
        "output": outputs[i, t].tolist(),
        "top_reconstruction": topk_reconstruction(weights[i, t], proto_out, args.top).tolist(),
        "position_held": positions[i, t].tolist(), "previous_position": prev[i, t].tolist(),
        "liability_delta": float(ep.np["liab_delta"][i, t]), "top": rows,
    }
    result = {"cell": a, "family": args.family, "split": args.split, "fidelity": fidelity, "example": example}
    rc.write_json(cell / args.family / f"explanation_{args.split}.json", result)

    print(f"Fidelity over {fidelity['decisions']} decisions ({args.split}):")
    for k in (1, 3, 5):
        print(f"  top-{k}: weight {fidelity[f'top{k}_weight']:.2f}, mean abs error "
              f"{fidelity[f'top{k}_mean_abs_error']:.4f} (output std {scale:.3f})")
    print(f"  effective prototypes {fidelity['effective_prototypes_mean']:.1f}; "
          f"{fidelity['prototypes_ever_top1']} of {fidelity['prototypes_total']} are ever the top prototype; "
          f"limits change {100 * fidelity['share_changed_by_limits']:.1f}% of outputs")
    print(f"  state similarity: the heaviest prototype is the nearest one in {100 * fidelity['top1_is_nearest_share']:.0f}% "
          f"of decisions and among the 5 nearest in {100 * fidelity['top1_among_5_nearest_share']:.0f}%; "
          f"median distance rank {fidelity['top1_median_distance_rank']:.0f} of {fidelity['prototypes_total']}")
    print(f"\nWorked example: episode {example['episode_start_date']} to {example['episode_end_date']}, step {t}")
    print("  state:", {k: round(v, 4) for k, v in example["state"].items()})
    print(f"  output ({example['output_meaning']}): spot {outputs[i, t, 0]:+.3f}, call {outputs[i, t, 1]:+.3f}; "
          f"held {positions[i, t, 0]:+.3f}, {positions[i, t, 1]:+.3f}; liability delta {example['liability_delta']:.3f}")
    for r in rows:
        print(f"  #{r['rank']} weight {r['weight']:.2f} from {r['source_start_date']} step {r['source_step']}: "
              f"spot {r['state']['spot']:.3f}, time_left {r['state']['time_left']:.3f} -> "
              f"spot {r['output_spot']:+.3f}, call {r['output_call']:+.3f}")
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("study", "ticker", "liability", "mode", "objective"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--state", default="s0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--family", default="ph", help="ph, ph_k25, phd, ...")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--allow-test", dest="allow_test", action="store_true")
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--top", type=int, default=3)
    explain(p.parse_args())


if __name__ == "__main__":
    main()

"""
Prototype extraction helpers for the PyTorch ProtoHedge pipeline.

The original notebooks build prototypes by:
1. constructing the same feature matrix the agent sees,
2. standardizing it with ``StandardScaler``,
3. clustering with KMeans, and
4. storing the closest real standardized point to each cluster center.

This module makes that workflow reusable for synthetic and real-data worlds.
"""

from pathlib import Path
import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_3d_per_step(x):
    x = _np(x)
    if x.ndim == 2:
        return x[:, :, None]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected a 2D or 3D per-step feature, got shape {x.shape}")


def _actions_to_delta(result, n_paths, n_steps, n_inst, dtype=np.float32):
    if result is None or "actions" not in result:
        return np.zeros((n_paths, n_steps, n_inst), dtype=dtype)

    actions = _as_3d_per_step(result["actions"]).astype(dtype, copy=False)
    if actions.shape[:2] != (n_paths, n_steps):
        raise ValueError(
            f"Action path shape {actions.shape[:2]} does not match world shape {(n_paths, n_steps)}"
        )
    if actions.shape[2] != n_inst:
        raise ValueError(f"Action instrument count {actions.shape[2]} does not match world nInst {n_inst}")

    # Delta at step t is the position before trading at t, matching the TF notebooks.
    return np.cumsum(actions, axis=1) - actions


def extract_agent_feature_matrix(world, result=None, feature_names=None):
    """
    Return the flattened raw feature matrix used for prototype extraction.

    ``feature_names`` are sorted to match ``run_train_torch.py`` and the
    TensorFlow ProtoAgent behavior.
    """
    feature_names = sorted(feature_names or ["price", "delta", "time_left"])
    per_step = world.data.features.per_step
    hedges = _as_3d_per_step(world.data.market.hedges)
    n_paths, n_steps, n_inst = hedges.shape
    dtype = hedges.dtype
    delta = _actions_to_delta(result, n_paths, n_steps, n_inst, dtype=dtype)

    columns = []
    for name in feature_names:
        if name in per_step:
            value = _as_3d_per_step(per_step[name]).astype(dtype, copy=False)
            if value.shape[:2] != (n_paths, n_steps):
                raise ValueError(
                    f"Feature '{name}' shape {value.shape[:2]} does not match world shape {(n_paths, n_steps)}"
                )
            columns.append(value)
        elif name in ["delta", "action"]:
            columns.append(delta)
        elif name in ["pnl", "cost"]:
            if result is None or name not in result:
                value = np.zeros((n_paths, n_steps, 1), dtype=dtype)
            else:
                value = _np(result[name]).reshape(n_paths, 1, 1).astype(dtype, copy=False)
                value = np.repeat(value, n_steps, axis=1)
            columns.append(value)
        else:
            raise KeyError(f"Unknown feature '{name}' for prototype extraction")

    x = np.concatenate(columns, axis=2)
    return x.reshape(n_paths * n_steps, x.shape[2]), feature_names


def build_payload_from_feature_matrix(x_raw, n_prototypes, random_state=0, max_points=None):
    x_raw = np.asarray(x_raw, dtype=np.float32)
    if x_raw.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {x_raw.shape}")
    if not np.isfinite(x_raw).all():
        raise ValueError("Prototype feature matrix contains NaN or infinite values")

    if max_points is not None and int(max_points) > 0 and x_raw.shape[0] > int(max_points):
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(x_raw.shape[0], size=int(max_points), replace=False)
        x_raw = x_raw[idx]

    n_prototypes = int(n_prototypes)
    if n_prototypes <= 0:
        raise ValueError("n_prototypes must be positive")
    if n_prototypes > x_raw.shape[0]:
        raise ValueError(
            f"Cannot extract {n_prototypes} prototypes from only {x_raw.shape[0]} feature rows"
        )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)

    with threadpool_limits(limits=1):
        kmeans = KMeans(n_clusters=n_prototypes, random_state=int(random_state), n_init="auto")
        labels = kmeans.fit_predict(x_scaled)

        real_prototypes = []
        for cluster_idx in range(n_prototypes):
            cluster_points = x_scaled[labels == cluster_idx]
            if len(cluster_points) == 0:
                continue
            center = kmeans.cluster_centers_[cluster_idx].reshape(1, -1)
            closest_idx, _ = pairwise_distances_argmin_min(center, cluster_points)
            real_prototypes.append(cluster_points[closest_idx[0]])

    if not real_prototypes:
        raise ValueError("KMeans produced no non-empty prototype clusters")

    return {
        "prototypes": np.stack(real_prototypes, axis=0).astype(np.float32),
        "scaler": scaler,
    }


def build_prototype_payload(
    world,
    n_prototypes,
    result=None,
    feature_names=None,
    random_state=0,
    max_points=None,
):
    x_raw, sorted_feature_names = extract_agent_feature_matrix(
        world=world,
        result=result,
        feature_names=feature_names,
    )
    payload = build_payload_from_feature_matrix(
        x_raw=x_raw,
        n_prototypes=n_prototypes,
        random_state=random_state,
        max_points=max_points,
    )
    payload["feature_names"] = sorted_feature_names
    payload["input_dim"] = int(x_raw.shape[1])
    return payload


def save_prototype_payload(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def build_and_save_prototype_payload(path, **kwargs):
    payload = build_prototype_payload(**kwargs)
    save_prototype_payload(payload, path)
    return payload

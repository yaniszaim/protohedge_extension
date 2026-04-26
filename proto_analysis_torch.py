import pickle
from pathlib import Path

import numpy as np
import torch

_BS_CONTEXT_CACHE = {}
_STOCH_CONTEXT_CACHE = {}


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_pickle_path(path):
    path = Path(path)
    repo_root = Path(__file__).resolve().parent
    candidates = [path]

    if not path.is_absolute():
        candidates.append(repo_root / path)
        candidates.append(repo_root / "notebooks" / path)
        candidates.append(repo_root / "notebooks" / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find prototype file '{path}'")


def load_prototype_payload(path):
    with open(resolve_pickle_path(path), "rb") as f:
        return pickle.load(f)


def binned_terminal_pnl(world, result, n_bins=20, use_raw_pnl=False):
    spot_t = _np(world.details.spot_all[:, -1])
    payoff = _np(result["payoff"]).reshape(-1)
    pnl = _np(result["pnl"]).reshape(-1)
    hedge_curve = pnl if use_raw_pnl else -pnl
    terminal_pnl = payoff + pnl

    bins = np.linspace(spot_t.min(), spot_t.max(), int(n_bins) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    payoff_binned = np.full((n_bins,), np.nan)
    gains_binned = np.full((n_bins,), np.nan)
    total_binned = np.full((n_bins,), np.nan)

    for i in range(n_bins):
        mask = (spot_t >= bins[i]) & (spot_t < bins[i + 1])
        if np.any(mask):
            payoff_binned[i] = np.mean(payoff[mask])
            gains_binned[i] = np.mean(hedge_curve[mask])
            total_binned[i] = np.mean(terminal_pnl[mask])

    order = np.argsort(spot_t)
    return {
        "spot_t": spot_t,
        "payoff": payoff,
        "hedge_curve": hedge_curve,
        "terminal_pnl": terminal_pnl,
        "bins": bins,
        "bin_centers": centers,
        "payoff_binned": payoff_binned,
        "gains_binned": gains_binned,
        "total_binned": total_binned,
        "sorted_spot_t": spot_t[order],
        "sorted_payoff": payoff[order],
        "sorted_gains": hedge_curve[order],
        "sorted_total": terminal_pnl[order],
    }


def extract_bs_path_features(world, result, path_num):
    price = _np(world.data.features.per_step["price"][path_num]).reshape(-1)
    time_left = _np(world.data.features.per_step["time_left"][path_num]).reshape(-1)
    actions = _np(result["actions"][path_num]).reshape(-1)
    delta = np.concatenate([[0.0], np.cumsum(actions[:-1])], axis=0)
    x = np.stack([delta, price, time_left], axis=1)
    return {
        "price": price,
        "time_left": time_left,
        "actions": actions,
        "delta": delta,
        "x": x,
    }


def extract_stochastic_path_features(world, result, path_num):
    price = _np(world.data.features.per_step["price"][path_num])
    time_left = _np(world.data.features.per_step["time_left"][path_num]).reshape(-1)
    actions = _np(result["actions"][path_num])
    delta = np.cumsum(actions, axis=0) - actions

    delta_1 = delta[:, 0]
    delta_2 = delta[:, 1]
    price_1 = price[:, 0]
    price_2 = price[:, 1]
    x = np.stack([delta_1, delta_2, price_1, price_2, time_left], axis=1)

    return {
        "price": price,
        "time_left": time_left,
        "actions": actions,
        "delta": delta,
        "x": x,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "price_1": price_1,
        "price_2": price_2,
    }


def _compute_similarity(agent, x_raw):
    x_tensor = torch.tensor(np.asarray(x_raw), dtype=torch.float32)
    similarities, distances = agent.compute_similarity(x_tensor, return_distances=True)
    return _np(similarities), _np(distances)


def _cpu_result_dict(result):
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}


def ensure_bs_test_context(results_proto_bs_weighted, results_bs=None, seed=1, samples=5000):
    """
    Build or retrieve the BS inference context used by the notebook analysis cells.
    This makes later cells robust even when run out of order.
    """
    cache_key = (id(results_proto_bs_weighted), int(seed), int(samples))
    cached = _BS_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return (
            cached["bs_test_world"],
            cached["bs_test_result"],
            cached["bs_test_result_vanilla"],
            cached["results_bs"],
        )

    import random
    from deephedging.base_torch import torchCast
    from deephedging.run_train_torch import run_experiment

    if results_proto_bs_weighted is None:
        raise ValueError("results_proto_bs_weighted is required to build the BS test context")

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if results_bs is None:
        print("Vanilla BS model not found in memory; training it now so the analysis cell can run.")
        config = results_proto_bs_weighted.get("config", {})
        world_cfg = dict(config.get("world", {}))
        objective_cfg = dict(config.get("objective", {}))
        training_cfg = dict(config.get("training", {}))
        model_cfg = {
            "agent_type": "feed_forward",
            "feature_list": ["price", "delta", "time_left"],
            "network_depth": 3,
            "network_width": 20,
            "activation": "softplus",
        }
        results_bs = run_experiment(
            override_world=world_cfg,
            override_objective=objective_cfg,
            override_model=model_cfg,
            override_training=training_cfg,
        )

    bs_test_world = results_proto_bs_weighted["world"].clone(seed=seed, samples=samples)
    bs_test_data = torchCast(bs_test_world.torch_data)

    with torch.no_grad():
        bs_test_result = results_proto_bs_weighted["gym"].forward(bs_test_data, training=False, return_paths=True)
        bs_test_result_vanilla = results_bs["gym"].forward(bs_test_data, training=False, return_paths=True)

    payload = {
        "bs_test_world": bs_test_world,
        "bs_test_result": _cpu_result_dict(bs_test_result),
        "bs_test_result_vanilla": _cpu_result_dict(bs_test_result_vanilla),
        "results_bs": results_bs,
    }
    _BS_CONTEXT_CACHE[cache_key] = payload

    return (
        payload["bs_test_world"],
        payload["bs_test_result"],
        payload["bs_test_result_vanilla"],
        payload["results_bs"],
    )


def ensure_stoch_test_context(results_proto_stoch, results_stoch_vanilla=None, seed=42, samples=5000):
    """
    Build or retrieve the stochastic inference context used by the notebook
    analysis cells so they remain runnable out of order.
    """
    cache_key = (id(results_proto_stoch), int(seed), int(samples))
    cached = _STOCH_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return (
            cached["stoch_test_world"],
            cached["stoch_test_data"],
            cached["stoch_test_result"],
            cached["stoch_test_result_vanilla"],
            cached["results_stoch_vanilla"],
        )

    import random
    from deephedging.base_torch import torchCast
    from deephedging.run_train_torch import run_experiment

    if results_proto_stoch is None:
        raise ValueError("results_proto_stoch is required to build the stochastic test context")

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if results_stoch_vanilla is None:
        print("Vanilla stochastic model not found in memory; training it now so the analysis cell can run.")
        config = results_proto_stoch.get("config", {})
        world_cfg = dict(config.get("world", {}))
        objective_cfg = dict(config.get("objective", {}))
        training_cfg = dict(config.get("training", {}))
        proto_model_cfg = dict(config.get("model", {}))
        model_cfg = {
            "agent_type": "feed_forward",
            "feature_list": proto_model_cfg.get("feature_list", ["price", "delta", "time_left"]),
            "network_depth": proto_model_cfg.get("network_depth", 3),
            "network_width": proto_model_cfg.get("network_width", 20),
            "activation": proto_model_cfg.get("activation", "softplus"),
            "prototype_path": None,
            "prototype_init": None,
            "weighted_similarity": False,
            "distance_feature_weights": None,
            "learn_distance_feature_weights": False,
        }
        results_stoch_vanilla = run_experiment(
            override_world=world_cfg,
            override_objective=objective_cfg,
            override_model=model_cfg,
            override_training=training_cfg,
        )

    stoch_test_world = results_proto_stoch["world"].clone(seed=seed, samples=samples)
    stoch_test_data = torchCast(stoch_test_world.torch_data)

    with torch.no_grad():
        stoch_test_result = results_proto_stoch["gym"].forward(
            stoch_test_data, training=False, return_paths=True
        )
        stoch_test_result_vanilla = results_stoch_vanilla["gym"].forward(
            stoch_test_data, training=False, return_paths=True
        )

    payload = {
        "stoch_test_world": stoch_test_world,
        "stoch_test_data": stoch_test_data,
        "stoch_test_result": _cpu_result_dict(stoch_test_result),
        "stoch_test_result_vanilla": _cpu_result_dict(stoch_test_result_vanilla),
        "results_stoch_vanilla": results_stoch_vanilla,
    }
    _STOCH_CONTEXT_CACHE[cache_key] = payload

    return (
        payload["stoch_test_world"],
        payload["stoch_test_data"],
        payload["stoch_test_result"],
        payload["stoch_test_result_vanilla"],
        payload["results_stoch_vanilla"],
    )


def _bs_price_actions(world, result):
    price = _np(world.data.features.per_step["price"])
    actions = _np(result["actions"])

    if price.ndim == 3:
        if price.shape[-1] != 1:
            raise ValueError(
                "bs_prototype_usage expects a Black-Scholes single-instrument world, "
                f"but got price shape {price.shape}. This usually means a stochastic "
                "test world/result was passed by mistake."
            )
        price = price[..., 0]
    elif price.ndim != 2:
        raise ValueError(f"Unexpected BS price shape {price.shape}")

    if actions.ndim == 3:
        if actions.shape[-1] != 1:
            raise ValueError(
                "bs_prototype_usage expects single-instrument actions with shape [B, T, 1], "
                f"but got {actions.shape}. Use stochastic_prototype_usage(...) for the "
                "stochastic notebook section."
            )
        actions = actions[..., 0]
    elif actions.ndim != 2:
        raise ValueError(f"Unexpected BS action shape {actions.shape}")

    return price, actions


def bs_similarity_details(agent, world, result, prototype_path, path_num, top_k=3):
    payload = load_prototype_payload(prototype_path)
    scaler = payload.get("scaler")
    prototypes = _np(payload["prototypes"])

    path = extract_bs_path_features(world, result, path_num)
    x_scaled = scaler.transform(path["x"]) if scaler is not None else path["x"]
    similarities, distances = _compute_similarity(agent, path["x"])
    prototype_actions = _np(agent.prototype_actions)
    prototypes_raw = scaler.inverse_transform(prototypes) if scaler is not None else prototypes

    rows = []
    for t in range(path["x"].shape[0]):
        top_idx = np.argsort(-similarities[t])[:top_k]
        rows.append(
            {
                "step": int(t),
                "input": {
                    "delta": float(path["delta"][t]),
                    "price": float(path["price"][t]),
                    "time_left": float(path["time_left"][t]),
                },
                "action": float(path["actions"][t]),
                "top_indices": top_idx,
                "top_prototypes": prototypes_raw[top_idx],
                "top_actions": prototype_actions[top_idx],
                "top_similarities": similarities[t, top_idx],
                "top_distances": distances[t, top_idx],
            }
        )

    return {
        "path_features": path,
        "x_scaled": x_scaled,
        "similarities": similarities,
        "distances": distances,
        "prototype_actions": prototype_actions,
        "prototypes_raw": prototypes_raw,
        "rows": rows,
        "feature_weights": _np(agent.feature_weights) if hasattr(agent, "feature_weights") else None,
    }


def stochastic_similarity_details(agent, world, result, prototype_path, path_num, top_k=3):
    payload = load_prototype_payload(prototype_path)
    scaler = payload.get("scaler")
    prototypes = _np(payload["prototypes"])

    path = extract_stochastic_path_features(world, result, path_num)
    x_scaled = scaler.transform(path["x"]) if scaler is not None else path["x"]
    similarities, distances = _compute_similarity(agent, path["x"])
    prototype_actions = _np(agent.prototype_actions)
    prototypes_raw = scaler.inverse_transform(prototypes) if scaler is not None else prototypes

    rows = []
    for t in range(path["x"].shape[0]):
        top_idx = np.argsort(-similarities[t])[:top_k]
        rows.append(
            {
                "step": int(t),
                "input": {
                    "delta": tuple(float(v) for v in path["delta"][t]),
                    "price": tuple(float(v) for v in path["price"][t]),
                    "time_left": float(path["time_left"][t]),
                },
                "action": tuple(float(v) for v in path["actions"][t]),
                "top_indices": top_idx,
                "top_prototypes": prototypes_raw[top_idx],
                "top_actions": prototype_actions[top_idx],
                "top_similarities": similarities[t, top_idx],
                "top_distances": distances[t, top_idx],
            }
        )

    return {
        "path_features": path,
        "x_scaled": x_scaled,
        "similarities": similarities,
        "distances": distances,
        "prototype_actions": prototype_actions,
        "prototypes_raw": prototypes_raw,
        "rows": rows,
        "feature_weights": _np(agent.feature_weights) if hasattr(agent, "feature_weights") else None,
    }


def bs_prototype_usage(agent, world, result, prototype_path, top_n=25):
    payload = load_prototype_payload(prototype_path)
    prototypes = _np(payload["prototypes"])

    price, actions = _bs_price_actions(world, result)
    time_left = _np(world.data.features.per_step["time_left"])

    n_paths, n_steps = price.shape
    delta = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(actions[:, :-1], axis=1)], axis=1)
    x = np.stack([delta, price, time_left], axis=2).reshape(-1, 3)

    similarities = agent.compute_similarity(torch.tensor(x, dtype=torch.float32))
    closest = _np(torch.argmax(similarities, dim=-1)).reshape(n_paths, n_steps)

    heatmap = np.zeros((prototypes.shape[0], n_steps), dtype=np.float32)
    for t in range(n_steps):
        unique, counts = np.unique(closest[:, t], return_counts=True)
        heatmap[unique, t] = counts

    totals = heatmap.sum(axis=1)
    top_indices = np.argsort(totals)[-top_n:][::-1]

    return {
        "heatmap": heatmap,
        "top_indices": top_indices,
        "top_heatmap": heatmap[top_indices],
        "top_totals": totals[top_indices],
        "totals": totals,
    }


def stochastic_prototype_usage(agent, world, result, prototype_path, top_n=25):
    payload = load_prototype_payload(prototype_path)
    prototypes = _np(payload["prototypes"])

    price = _np(world.data.features.per_step["price"])
    time_left = _np(world.data.features.per_step["time_left"])
    actions = _np(result["actions"])

    n_paths, n_steps, _ = price.shape
    delta = np.cumsum(actions, axis=1) - actions
    x = np.stack(
        [
            delta[:, :, 0],
            delta[:, :, 1],
            price[:, :, 0],
            price[:, :, 1],
            time_left,
        ],
        axis=2,
    ).reshape(-1, 5)

    similarities = agent.compute_similarity(torch.tensor(x, dtype=torch.float32))
    closest = _np(torch.argmax(similarities, dim=-1)).reshape(n_paths, n_steps)

    heatmap = np.zeros((prototypes.shape[0], n_steps), dtype=np.float32)
    for t in range(n_steps):
        unique, counts = np.unique(closest[:, t], return_counts=True)
        heatmap[unique, t] = counts

    totals = heatmap.sum(axis=1)
    top_indices = np.argsort(totals)[-top_n:][::-1]

    return {
        "heatmap": heatmap,
        "top_indices": top_indices,
        "top_heatmap": heatmap[top_indices],
        "top_totals": totals[top_indices],
        "totals": totals,
    }

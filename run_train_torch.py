"""
Main training entry point for PyTorch ProtoHedge.
"""

import copy
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm

from deephedging.world_torch import SimpleWorld_Spot_ATM
from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch
from deephedging.base_torch import resolve_torch_device, torchCast
from deephedging.agents_torch import ProtoHedgeAgent, VanillaHedgeAgent
from deephedging.objectives_torch import HedgingObjective
from deephedging.gym_torch import DeepHedgingGymTorch
from deephedging.trainer_torch import TrainerTorch
from deephedging.prototype_extraction_torch import build_payload_from_feature_matrix
from deephedging.payoff_state import (
    ASIAN_PAYOFF_STATE_VERSION,
    is_asian_liability,
    validate_model_features_for_liability,
)


def default_config():
    return {
        "world": {
            "world_type": "synthetic",
            "steps": 10,
            "samples": 1000,
            "seed": 1234,
            "drift": 0.0,
            "dt": 1 / 50,
            "black_scholes": False,
        },
        "model": {
            "agent_type": "protopnet",
            "num_prototypes": 8,
            "prototype_init": None,
            "prototype_path": None,
            "feature_list": ["price", "delta", "time_left"],
            "network_width": 20,
            "network_depth": 3,
            "activation": "softplus",
            "weighted_similarity": False,
            "distance_feature_weights": None,
            "learn_distance_feature_weights": False,
        },
        "training": {
            "epochs": 200,
            "lr": 1e-3,
            "device": "auto",
            "batch_size": None,
            "optimizer": "adam",
            "epoch_refresh": 20,
            "seed": 0,
            "clipvalue": None,
            "global_clipnorm": None,
            "lr_decay_factor": None,
            "lr_decay_patience": None,
            "lr_min": 1e-5,
            "scheduler_monitor": "val",
            "selection_metric": "train_loss",
            "selection_alpha_action_abs": 0.0,
            "selection_alpha_delta_abs": 0.0,
            "selection_alpha_bound_occupancy": 0.0,
            "selection_alpha_path_bound_touch": 0.0,
            "action_penalty_weight": 0.0,
            "delta_penalty_weight": 0.0,
        },
        "objective": {
            "risk_measure": "cvar",
            "risk_aversion": 1.0,
        },
    }


def _load_prototype_payload(model_cfg):
    payload = model_cfg.get("prototype_init")
    path = model_cfg.get("prototype_path")

    if payload is None and path:
        path_obj = Path(path)
        repo_root = Path(__file__).resolve().parent
        candidates = [path_obj]

        if not path_obj.is_absolute():
            candidates.append(repo_root / path_obj)
            candidates.append(repo_root / "notebooks" / path_obj)
            candidates.append(repo_root / "notebooks" / path_obj.name)

        resolved = None
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break

        if resolved is None:
            raise FileNotFoundError(
                f"Could not find prototype file '{path}'. Tried: "
                + ", ".join(str(c) for c in candidates)
            )

        with open(resolved, "rb") as f:
            payload = pickle.load(f)

    if payload is None:
        return None

    if isinstance(payload, dict):
        return payload

    return {"prototypes": payload}


def _resolve_prototype_path(path):
    path_obj = Path(path)
    repo_root = Path(__file__).resolve().parent
    candidates = [path_obj]

    if not path_obj.is_absolute():
        candidates.append(repo_root / path_obj)
        candidates.append(repo_root / "notebooks" / path_obj)
        candidates.append(repo_root / "notebooks" / path_obj.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if path_obj.is_absolute():
        return path_obj

    return repo_root / path_obj


def _build_bs_prototype_payload(world, n_prototypes):
    price = np.asarray(world.data.features.per_step["price"])
    if price.ndim == 3:
        price = price[:, :, 0]
    time_left = np.asarray(world.data.features.per_step["time_left"])

    if "ivol" in world.data.features.per_step:
        vol = np.asarray(world.data.features.per_step["ivol"])
        if vol.ndim == 3:
            vol = vol[:, :, 0]
    else:
        vol = np.full_like(price, 0.2)

    safe_tau = np.maximum(time_left, 1e-12)
    safe_vol = np.maximum(vol, 1e-8)
    d1 = (
        np.log(np.maximum(price, 1e-12))
        + 0.5 * (safe_vol ** 2) * safe_tau
    ) / (safe_vol * np.sqrt(safe_tau))
    delta = norm.cdf(d1)

    name_to_array = {
        "price": price,
        "delta": delta,
        "time_left": time_left,
    }
    sorted_names = sorted(["price", "delta", "time_left"])
    x_sorted = np.stack([name_to_array[k] for k in sorted_names], axis=2).reshape(-1, 3)

    return build_payload_from_feature_matrix(x_sorted, n_prototypes=n_prototypes, random_state=0)


def _build_world(world_cfg):
    world_type = str(world_cfg.get("world_type", "synthetic")).lower()
    if world_type in ["synthetic", "simple", "simpleworld", "simulated"]:
        return SimpleWorld_Spot_ATM(world_cfg)
    if world_type in ["real", "real_data", "world_real"]:
        return RealWorld_Spot_ATM_Torch(world_cfg)
    raise ValueError(f"Unknown torch world_type '{world_type}'")


def _build_validation_world(world, world_cfg):
    val_override = world_cfg.get("val_world")
    if val_override is None and world_cfg.get("val_sample_indices") is not None:
        val_override = {
            "sample_indices": world_cfg["val_sample_indices"],
            "samples": None,
            "shuffle": False,
        }

    if val_override is not None:
        val_cfg = dict(world_cfg)
        val_cfg.pop("val_world", None)
        val_cfg.pop("val_sample_indices", None)
        val_cfg.update(val_override)
        return _build_world(val_cfg)

    return world.clone(samples=max(100, world.nSamples // 10))


def _to_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().clone().float()
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def _merged_config(
    override_world=None,
    override_training=None,
    override_model=None,
    override_objective=None,
):
    config = copy.deepcopy(default_config())

    if override_world:
        config["world"].update(override_world)
    if override_training:
        config["training"].update(override_training)
    if override_model:
        config["model"].update(override_model)
    if override_objective:
        config["objective"].update(override_objective)

    alias_vol = None
    if "sigma" in config["world"]:
        alias_vol = config["world"].pop("sigma")
    if "volatility" in config["world"]:
        alias_vol = config["world"].pop("volatility")
    if alias_vol is not None:
        config["world"].setdefault("rvol", alias_vol)
        config["world"].setdefault("ivol", alias_vol)
    return config


def build_experiment_components(config):
    config = copy.deepcopy(config)

    seed = int(config["training"].get("seed", 0))
    device = resolve_torch_device(config["training"].get("device", "auto"))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    world = _build_world(config["world"])
    val_world = _build_validation_world(world, config["world"])

    train_data = torchCast(world.torch_data, device=device)
    val_data = torchCast(val_world.torch_data, device=device)
    n_inst = int(train_data["market"]["hedges"].shape[-1])

    prototype_payload = _load_prototype_payload(config["model"])
    liability_type = config["world"].get(
        "liability_type",
        getattr(world, "liability_type", "european_call"),
    )
    feature_names = validate_model_features_for_liability(
        liability_type,
        config["model"].get("feature_list", ["price", "delta", "time_left"]),
    )
    inferred_agent_type = config["model"].get("agent_type")
    if inferred_agent_type is None:
        inferred_agent_type = "protopnet" if (
            prototype_payload is not None or config["model"].get("prototype_path") is not None
        ) else "feed_forward"
    agent_type = str(inferred_agent_type).lower()

    if isinstance(prototype_payload, dict):
        payload_feature_names = prototype_payload.get("feature_names")
        if payload_feature_names is None and is_asian_liability(liability_type):
            raise ValueError(
                "Asian ProtoHedge payloads must record the feature names used "
                "for clustering. Re-extract the prototypes with the corrected "
                "running-average state."
            )
        if payload_feature_names is not None:
            payload_feature_names = sorted(str(name) for name in payload_feature_names)
            if payload_feature_names != feature_names:
                raise ValueError(
                    "Prototype payload features do not match the policy features: "
                    f"payload={payload_feature_names}, policy={feature_names}"
                )
        if (
            is_asian_liability(liability_type)
            and prototype_payload.get("payoff_state_version")
            != ASIAN_PAYOFF_STATE_VERSION
        ):
            raise ValueError(
                "Asian ProtoHedge payload was not extracted with payoff-state "
                f"version {ASIAN_PAYOFF_STATE_VERSION!r}. Re-extract it before training."
            )

    prototype_path = config["model"].get("prototype_path")
    if (
        agent_type in ["protopnet", "proto", "protohedge"]
        and
        prototype_payload is None
        and config["world"].get("black_scholes", False)
        and feature_names == ["delta", "price", "time_left"]
    ):
        n_prototypes = int(config["model"].get("n_prototypes", config["model"].get("num_prototypes", 100)))
        print("No BS prototype payload found; generating clustered standardized prototypes.")
        prototype_payload = _build_bs_prototype_payload(world, n_prototypes)
        if prototype_path:
            write_path = _resolve_prototype_path(prototype_path)
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with open(write_path, "wb") as f:
                pickle.dump(prototype_payload, f)

    if (
        agent_type in ["protopnet", "proto", "protohedge"]
        and
        prototype_payload is not None
        and not isinstance(prototype_payload, dict)
        and config["world"].get("black_scholes", False)
        and feature_names == ["delta", "price", "time_left"]
    ):
        n_prototypes = int(np.asarray(prototype_payload).shape[0])
        prototype_payload = _build_bs_prototype_payload(world, n_prototypes)
        if prototype_path:
            write_path = _resolve_prototype_path(prototype_path)
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with open(write_path, "wb") as f:
                pickle.dump(prototype_payload, f)

    if (
        agent_type in ["protopnet", "proto", "protohedge"]
        and
        isinstance(prototype_payload, dict)
        and "scaler" not in prototype_payload
        and config["world"].get("black_scholes", False)
        and feature_names == ["delta", "price", "time_left"]
    ):
        n_prototypes = int(np.asarray(prototype_payload["prototypes"]).shape[0])
        print("Legacy BS prototype file detected; upgrading it to prototypes + scaler format.")
        prototype_payload = _build_bs_prototype_payload(world, n_prototypes)
        if prototype_path:
            write_path = _resolve_prototype_path(prototype_path)
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with open(write_path, "wb") as f:
                pickle.dump(prototype_payload, f)
    input_dim = 0
    for name in feature_names:
        if name in train_data["features"]["per_step"]:
            tensor = train_data["features"]["per_step"][name]
            input_dim += int(tensor.shape[-1] if tensor.ndim == 3 else 1)
        elif name in ["delta", "action"]:
            input_dim += n_inst
        elif name in ["pnl", "cost"]:
            input_dim += 1
        else:
            raise KeyError(f"Unknown feature '{name}' for torch ProtoHedge")

    prototype_init = None
    feature_mean = None
    feature_std = None
    if prototype_payload is not None:
        prototype_init = _to_tensor(prototype_payload.get("prototypes"))
        payload_input_dim = prototype_payload.get("input_dim")
        if payload_input_dim is not None and int(payload_input_dim) != int(input_dim):
            raise ValueError(
                "Prototype payload input dimension does not match the policy: "
                f"payload={payload_input_dim}, policy={input_dim}"
            )
        scaler = prototype_payload.get("scaler")
        if scaler is not None:
            feature_mean = _to_tensor(scaler.mean_)
            feature_std = _to_tensor(scaler.scale_)

    action_low = train_data["market"]["lbnd_a"][0, 0, :].detach().cpu()
    action_high = train_data["market"]["ubnd_a"][0, 0, :].detach().cpu()

    if agent_type in ["protopnet", "proto", "protohedge"]:
        distance_feature_weights = config["model"].get("distance_feature_weights")
        if distance_feature_weights is None and config["model"].get("weighted_similarity", False):
            distance_feature_weights = np.ones((input_dim,), dtype=np.float32)
            if input_dim >= 1:
                distance_feature_weights[-1] = 5.0

        agent = ProtoHedgeAgent(
            input_dim=input_dim,
            num_prototypes=(
                int(prototype_init.shape[0])
                if prototype_init is not None
                else int(config["model"].get("n_prototypes", config["model"].get("num_prototypes", 8)))
            ),
            output_dim=n_inst,
            prototype_init=prototype_init,
            feature_mean=feature_mean,
            feature_std=feature_std,
            action_low=action_low,
            action_high=action_high,
            distance_feature_weights_init=distance_feature_weights,
            learn_distance_feature_weights=config["model"].get("learn_distance_feature_weights", False),
            softclip_mode=config["model"].get("softclip_mode", "legacy_approx"),
        )
    elif agent_type in ["feed_forward", "feedforward", "vanilla", "dense"]:
        agent = VanillaHedgeAgent(
            input_dim=input_dim,
            output_dim=n_inst,
            hidden_width=int(config["model"].get("network_width", 20)),
            hidden_depth=int(config["model"].get("network_depth", 3)),
            activation=config["model"].get("activation", "softplus"),
        )
    else:
        raise ValueError(f"Unknown torch agent_type '{agent_type}'")

    objective = HedgingObjective(
        risk_measure=config["objective"]["risk_measure"],
        risk_aversion=config["objective"]["risk_aversion"],
    )

    gym = DeepHedgingGymTorch(
        agent=agent,
        objective=objective,
        feature_names=feature_names,
        action_penalty_weight=float(config["training"].get("action_penalty_weight", 0.0)),
        delta_penalty_weight=float(config["training"].get("delta_penalty_weight", 0.0)),
        device=device,
    )

    trainer = TrainerTorch(
        gym,
        lr=config["training"]["lr"],
        clipvalue=config["training"].get("clipvalue"),
        global_clipnorm=config["training"].get("global_clipnorm"),
        lr_decay_factor=config["training"].get("lr_decay_factor"),
        lr_decay_patience=config["training"].get("lr_decay_patience"),
        lr_min=config["training"].get("lr_min"),
        scheduler_monitor=config["training"].get("scheduler_monitor", "val"),
        device=device,
    )

    return {
        "config": config,
        "world": world,
        "val_world": val_world,
        "train_data": train_data,
        "val_data": val_data,
        "gym": gym,
        "trainer": trainer,
        "device": device,
        "feature_names": feature_names,
        "n_inst": n_inst,
    }


def run_experiment(
    override_world=None,
    override_training=None,
    override_model=None,
    override_objective=None,
):
    config = _merged_config(
        override_world=override_world,
        override_training=override_training,
        override_model=override_model,
        override_objective=override_objective,
    )
    components = build_experiment_components(config)
    world = components["world"]
    val_world = components["val_world"]
    train_data = components["train_data"]
    val_data = components["val_data"]
    gym = components["gym"]
    trainer = components["trainer"]
    device = components["device"]
    n_inst = components["n_inst"]

    print(
        f"\nPyTorch version {torch.__version__} "
        f"running on {torch.get_num_threads()} CPUs "
        f"and {torch.cuda.device_count()} GPUs | selected device={device}"
    )
    if device.type == "cuda":
        print(f"CUDA accelerator: {torch.cuda.get_device_name(device)}")

    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        train_weights=np.asarray(world.sample_weights),
        val_weights=np.asarray(val_world.sample_weights),
        n_epochs=int(config["training"]["epochs"]),
        epoch_refresh=int(config["training"].get("epoch_refresh", 20)),
        batch_size=config["training"].get("batch_size"),
        selection_metric=config["training"].get("selection_metric", "train_loss"),
        selection_alpha_action_abs=float(config["training"].get("selection_alpha_action_abs", 0.0)),
        selection_alpha_delta_abs=float(config["training"].get("selection_alpha_delta_abs", 0.0)),
        selection_alpha_bound_occupancy=float(config["training"].get("selection_alpha_bound_occupancy", 0.0)),
        selection_alpha_path_bound_touch=float(config["training"].get("selection_alpha_path_bound_touch", 0.0)),
    )

    gym.eval()
    with torch.no_grad():
        train_result = gym.forward(train_data, training=False, return_paths=True)
        val_result = gym.forward(val_data, training=False, return_paths=True)

    results = {
        "config": config,
        "device": str(device),
        "gym": gym,
        "model": gym.agent,
        "world": world,
        "val_world": val_world,
        "history": history,
        "training_result": {k: v.detach().cpu() for k, v in train_result.items()},
        "val_result": {k: v.detach().cpu() for k, v in val_result.items()},
        "sample_weights": np.asarray(world.sample_weights),
        "val_sample_weights": np.asarray(val_world.sample_weights),
        "inst_names": getattr(world, "inst_names", [f"inst_{i}" for i in range(n_inst)]),
        "loss": np.asarray(history["losses"]["training"], dtype=np.float32),
        "pnl": train_result["pnl"].detach().cpu().numpy(),
        "utility": train_result["utility"].detach().cpu().numpy(),
    }

    return results

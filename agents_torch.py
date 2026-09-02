import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deephedging.softclip_torch import legacy_proto_softclip, tfp_softclip


def _activation_module(name):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "softplus":
        return nn.Softplus()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation '{name}'")


def _inverse_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.log(torch.expm1(torch.clamp(x, min=1e-6)))


class VanillaHedgeAgent(nn.Module):
    """
    Simple feed-forward agent for the vanilla Deep Hedging notebook cells.
    """

    def __init__(
        self,
        input_dim,
        output_dim=1,
        hidden_width=20,
        hidden_depth=3,
        activation="softplus",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.internally_bounded_actions = False

        layers = []
        prev_dim = self.input_dim
        for _ in range(int(hidden_depth)):
            layers.append(nn.Linear(prev_dim, int(hidden_width)))
            layers.append(_activation_module(activation))
            prev_dim = int(hidden_width)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features, return_weights=False):
        if features.ndim == 3:
            batch, n_steps, d = features.shape
            x = features.reshape(batch * n_steps, d)
        else:
            batch = features.shape[0]
            n_steps = None
            x = features

        actions = self.network(x)

        if n_steps is not None:
            actions = actions.reshape(batch, n_steps, self.output_dim)

        if return_weights:
            return actions, None
        return actions


class ProtoHedgeAgent(nn.Module):
    """
    PyTorch port of the TensorFlow ProtoAgent + ClusteredProtoLayer setup.
    The prototypes live in feature space and each prototype carries a trainable
    action vector. Inputs and prototypes are projected into a shared latent space
    before similarity weighting.
    """

    def __init__(
        self,
        input_dim,
        num_prototypes,
        output_dim=1,
        prototype_init=None,
        feature_mean=None,
        feature_std=None,
        action_low=None,
        action_high=None,
        distance_feature_weights_init=None,
        learn_distance_feature_weights=False,
        softclip_mode="legacy_approx",
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        if prototype_init is None:
            prototype_tensor = torch.randn(num_prototypes, self.input_dim, dtype=torch.float32)
        else:
            if isinstance(prototype_init, np.ndarray):
                prototype_init = torch.tensor(prototype_init, dtype=torch.float32)
            elif not isinstance(prototype_init, torch.Tensor):
                prototype_init = torch.tensor(prototype_init, dtype=torch.float32)

            if prototype_init.ndim != 2:
                raise ValueError("prototype_init must have shape [num_prototypes, input_dim]")
            if prototype_init.shape[1] != self.input_dim:
                raise ValueError(
                    f"prototype_init feature dim {prototype_init.shape[1]} "
                    f"does not match input_dim {self.input_dim}"
                )
            prototype_tensor = prototype_init.detach().clone().float()
            num_prototypes = int(prototype_tensor.shape[0])

        self.num_prototypes = int(num_prototypes)
        self.softclip_mode = str(softclip_mode).lower()
        if self.softclip_mode not in {"legacy_approx", "tfp_exact"}:
            raise ValueError(
                "softclip_mode must be 'legacy_approx' or 'tfp_exact'"
            )

        if feature_mean is None:
            feature_mean = torch.zeros(self.input_dim, dtype=torch.float32)
        if feature_std is None:
            feature_std = torch.ones(self.input_dim, dtype=torch.float32)

        if action_low is None:
            action_low = torch.full((self.output_dim,), -5.0, dtype=torch.float32)
        if action_high is None:
            action_high = torch.full((self.output_dim,), 5.0, dtype=torch.float32)

        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.clamp(torch.as_tensor(feature_std, dtype=torch.float32), min=1e-8))
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32).reshape(self.output_dim))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32).reshape(self.output_dim))

        # Match the TensorFlow ProtoAgent: prototypes are fixed anchors in feature
        # space, while the projection and prototype actions are trainable.
        self.register_buffer("prototypes", prototype_tensor.detach().clone().float())
        self.pre_mlp = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim),
        )
        proto_action_init = torch.empty(self.num_prototypes, self.output_dim, dtype=torch.float32)
        nn.init.normal_(proto_action_init, mean=0.0, std=0.05)
        self.prototype_actions_unbounded = nn.Parameter(proto_action_init)
        self.internally_bounded_actions = True

        if distance_feature_weights_init is None:
            distance_feature_weights_init = torch.ones(self.input_dim, dtype=torch.float32)
        distance_feature_weights_init = torch.as_tensor(distance_feature_weights_init, dtype=torch.float32).reshape(self.input_dim)
        if bool(learn_distance_feature_weights):
            self.distance_feature_weights_unconstrained = nn.Parameter(
                _inverse_softplus(distance_feature_weights_init)
            )
        else:
            self.register_buffer("distance_feature_weights_fixed", distance_feature_weights_init)

    def _normalize(self, x):
        return (x - self.feature_mean) / self.feature_std

    @property
    def feature_weights(self):
        if hasattr(self, "distance_feature_weights_unconstrained"):
            return F.softplus(self.distance_feature_weights_unconstrained) + 1e-6
        return self.distance_feature_weights_fixed

    def _prepare_inputs(self, features):
        if features.ndim == 3:
            batch, n_steps, d = features.shape
            x = features.reshape(batch * n_steps, d)
        else:
            batch = features.shape[0]
            n_steps = None
            x = features
        return x, batch, n_steps

    def compute_similarity(self, features, return_distances=False):
        x, batch, n_steps = self._prepare_inputs(features)
        x = self._normalize(x)
        weights = self.feature_weights.reshape(1, -1)
        x_latent = self.pre_mlp(x * weights)
        proto_latent = self.pre_mlp(self.prototypes * weights)
        distances = torch.sum((x_latent[:, None, :] - proto_latent[None, :, :]) ** 2, dim=-1)
        similarities = torch.softmax(-distances, dim=-1)

        if n_steps is not None:
            similarities = similarities.reshape(batch, n_steps, self.num_prototypes)
            distances = distances.reshape(batch, n_steps, self.num_prototypes)

        if return_distances:
            return similarities, distances
        return similarities

    def _bounded_prototype_actions(self):
        if self.softclip_mode == "tfp_exact":
            return tfp_softclip(
                self.prototype_actions_unbounded,
                self.action_low,
                self.action_high,
            )
        return legacy_proto_softclip(
            self.prototype_actions_unbounded,
            self.action_low,
            self.action_high,
        )

    def forward(self, features, return_weights=False):
        similarities = self.compute_similarity(features, return_distances=False)
        actions = similarities @ self._bounded_prototype_actions()

        if return_weights:
            return actions, similarities
        return actions

    @property
    def prototype_actions(self):
        return self._bounded_prototype_actions().detach()

    def print_sample_prototypes(self, indices=None):
        if indices is None:
            indices = [0, min(50, self.num_prototypes - 1), min(218, self.num_prototypes - 1)]

        prototypes = self.prototypes.detach().cpu().numpy()
        actions = self.prototype_actions.detach().cpu().numpy()

        for i in indices:
            idx = int(i)
            if idx < 0 or idx >= self.num_prototypes:
                print(f"Prototype {idx}: out of range for {self.num_prototypes} stored prototypes")
                continue
            print(f"Prototype {idx}: Features = {prototypes[idx]}, Action = {actions[idx]}")

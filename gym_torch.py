"""
PyTorch version of the Deep Hedging gym.
"""

import torch
import torch.nn as nn

from deephedging.prototype_loss_torch import prototype_diversity_loss, prototype_l2_loss


class DeepHedgingGymTorch(nn.Module):
    def __init__(
        self,
        agent,
        objective,
        feature_names=None,
        proto_diversity_weight=0.0,
        proto_l2_weight=0.0,
        action_penalty_weight=0.0,
        delta_penalty_weight=0.0,
        device="cpu",
    ):
        super().__init__()
        self.agent = agent.to(device)
        self.objective = objective.to(device)
        self.feature_names = list(feature_names or ["price", "delta", "time_left"])
        self.proto_diversity_weight = float(proto_diversity_weight)
        self.proto_l2_weight = float(proto_l2_weight)
        self.action_penalty_weight = float(action_penalty_weight)
        self.delta_penalty_weight = float(delta_penalty_weight)
        self.device = device
        self.clip_actions = not bool(getattr(self.agent, "internally_bounded_actions", False))

    def _feature_tensor(self, data, t, action, delta, pnl, cost):
        live_features = {
            "action": action,
            "delta": delta,
            "pnl": pnl[:, None],
            "cost": cost[:, None],
        }

        per_step = data["features"]["per_step"]
        for name in per_step:
            value = per_step[name]
            if value.ndim == 2:
                live_features[name] = value[:, t].to(self.device)[:, None]
            else:
                live_features[name] = value[:, t, :].to(self.device)

        cols = []
        for name in self.feature_names:
            value = live_features[name]
            if value.ndim == 1:
                value = value[:, None]
            cols.append(value)
        return torch.cat(cols, dim=-1)

    def forward(self, data, training=True, return_paths=False):
        market = data["market"]
        hedges = market["hedges"].to(self.device)
        trading_cost = market["cost"].to(self.device)
        ubnd_a = market["ubnd_a"].to(self.device)
        lbnd_a = market["lbnd_a"].to(self.device)
        ubnd_delta = market["ubnd_delta"].to(self.device) if "ubnd_delta" in market else None
        lbnd_delta = market["lbnd_delta"].to(self.device) if "lbnd_delta" in market else None
        payoff = market["payoff"].to(self.device)

        if payoff.ndim > 1:
            payoff = payoff.squeeze(-1)

        n_batch, n_steps, n_inst = hedges.shape
        pnl = torch.zeros(n_batch, device=self.device)
        cost = torch.zeros(n_batch, device=self.device)
        delta = torch.zeros(n_batch, n_inst, device=self.device)
        action_prev = torch.zeros(n_batch, n_inst, device=self.device)
        actions = []
        prototype_weights = []

        for t in range(n_steps):
            x_t = self._feature_tensor(data, t, action_prev, delta, pnl, cost)

            if return_paths:
                action, weights = self.agent(x_t, return_weights=True)
                if weights is not None:
                    prototype_weights.append(weights[:, None, :])
            else:
                action = self.agent(x_t)

            if self.clip_actions:
                action = torch.maximum(torch.minimum(action, ubnd_a[:, t, :]), lbnd_a[:, t, :])

            if ubnd_delta is not None and lbnd_delta is not None:
                bounded_delta = torch.maximum(
                    torch.minimum(delta + action, ubnd_delta[:, t, :]),
                    lbnd_delta[:, t, :],
                )
                action = bounded_delta - delta

            pnl = pnl + torch.sum(action * hedges[:, t, :], dim=1)
            cost = cost + torch.sum(torch.abs(action) * trading_cost[:, t, :], dim=1)
            delta = delta + action
            action_prev = action
            actions.append(action[:, None, :])

        actions = torch.cat(actions, dim=1)
        deltas = torch.cumsum(actions, dim=1)

        objective_out = self.objective(payoff=payoff, pnl=pnl, cost=cost)

        reg = torch.tensor(0.0, device=self.device)
        if self.proto_diversity_weight:
            reg = reg + self.proto_diversity_weight * prototype_diversity_loss(self.agent.prototypes)
        if self.proto_l2_weight:
            reg = reg + self.proto_l2_weight * prototype_l2_loss(self.agent.prototypes)
        action_penalty = torch.tensor(0.0, device=self.device)
        delta_penalty = torch.tensor(0.0, device=self.device)
        if self.action_penalty_weight:
            action_penalty = self.action_penalty_weight * torch.mean(torch.abs(actions))
            reg = reg + action_penalty
        if self.delta_penalty_weight:
            delta_penalty = self.delta_penalty_weight * torch.mean(torch.abs(deltas))
            reg = reg + delta_penalty

        result = {
            "loss": objective_out["loss"] + reg,
            "loss_path": objective_out["loss_path"],
            "loss_no_reg": objective_out["loss"],
            "reg": reg,
            "action_penalty": action_penalty,
            "delta_penalty": delta_penalty,
            "utility": objective_out["utility"],
            "utility0": objective_out["utility0"],
            "gains": objective_out["gains"],
            "payoff": payoff,
            "pnl": pnl,
            "cost": cost,
            "actions": actions,
            "deltas": deltas,
        }

        if return_paths and prototype_weights:
            result["prototype_weights"] = torch.cat(prototype_weights, dim=1)

        return result

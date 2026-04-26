"""
PyTorch monetary-utility objectives matching the TensorFlow Deep Hedging setup.
"""

import torch
import torch.nn as nn


class MonetaryUtilityTorch(nn.Module):
    def __init__(self, utility="exp2", lmbda=1.0, init_y=0.0):
        super().__init__()
        self.utility = str(utility)
        self.lmbda = float(lmbda)
        self.y = nn.Parameter(torch.tensor(float(init_y), dtype=torch.float32))

    def _utility_values(self, x):
        lmbda = self.lmbda
        gains = x + self.y

        if lmbda < 1e-12:
            return gains

        if self.utility in ["mean", "expectation"]:
            return gains

        if self.utility == "cvar":
            return (1.0 + lmbda) * torch.minimum(
                gains, torch.zeros_like(gains)
            ) - self.y

        if self.utility in ["exp", "entropy"]:
            inf = torch.min(x).detach()
            return (1.0 - torch.exp(-lmbda * (gains - inf))) / lmbda - self.y + inf

        if self.utility == "exp2":
            g1 = torch.maximum(gains, torch.zeros_like(gains))
            g2 = torch.minimum(gains, torch.zeros_like(gains))
            u1 = (1.0 - torch.exp(-lmbda * g1)) / lmbda - self.y
            u2 = g2 - 0.5 * lmbda * g2 * g2 - self.y
            return torch.where(gains > 0.0, u1, u2)

        if self.utility == "quad":
            x0 = 1.0 / lmbda
            xx = torch.minimum(gains - x0, torch.zeros_like(gains))
            return -0.5 * lmbda * (xx ** 2) + 0.5 * lmbda * (x0 ** 2) - self.y

        if self.utility == "vicky":
            return (
                1.0
                + lmbda * gains
                - torch.sqrt(1.0 + (lmbda * gains) ** 2)
            ) / lmbda - self.y

        raise ValueError(f"unknown utility {self.utility}")

    def forward(self, x):
        return self._utility_values(x)


class HedgingObjective(nn.Module):
    """
    Wrapper that mirrors the original TensorFlow gym setup:
    one utility for hedged gains and one baseline utility for the unhedged payoff.
    """

    def __init__(self, risk_measure="exp2", risk_aversion=1.0):
        super().__init__()
        self.utility = MonetaryUtilityTorch(risk_measure, risk_aversion)
        self.utility0 = MonetaryUtilityTorch(risk_measure, risk_aversion)

    def forward(self, payoff, pnl, cost):
        gains = payoff + pnl - cost
        utility = self.utility(gains)
        utility0 = self.utility0(payoff)
        loss_path = -utility - utility0

        return {
            "loss": torch.mean(loss_path),
            "loss_path": loss_path,
            "utility": utility,
            "utility0": utility0,
            "gains": gains,
        }

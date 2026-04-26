import torch
import torch.nn as nn


class SoftClip(nn.Module):
    """
    smooth clipping function used in Deep Hedging
    prevents extreme hedge ratios
    """

    def __init__(self, lower=-1.0, upper=1.0):
        super().__init__()

        self.lower = lower
        self.upper = upper

    def forward(self, x):

        return (
            self.lower
            + (self.upper - self.lower)
            * torch.sigmoid(x)
        )
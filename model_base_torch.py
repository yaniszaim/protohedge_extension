import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


    def count_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
"""
PyTorch version of ProtoHedge layers
Replaces TensorFlow / Keras layers with torch.nn equivalents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from deephedging.softclip_torch import SoftClip

# ------------------------------------------------------------
# Dense network block
# ------------------------------------------------------------

class DenseBlock(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_units=[32, 32],
        activation="relu",
        output_dim=None
    ):
        super().__init__()

        layers = []

        prev_dim = input_dim

        for h in hidden_units:

            layers.append(nn.Linear(prev_dim, h))

            if activation == "relu":
                layers.append(nn.ReLU())

            elif activation == "tanh":
                layers.append(nn.Tanh())

            elif activation == "elu":
                layers.append(nn.ELU())

            prev_dim = h

        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)



# ------------------------------------------------------------
# Prototype Layer
# ------------------------------------------------------------

class PrototypeLayer(nn.Module):

    """
    Learns K prototypes in feature space
    Computes distance between input features and prototypes
    """

    def __init__(
        self,
        num_prototypes,
        feature_dim
    ):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim

        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, feature_dim)
        )

    def forward(self, x):

        """
        x shape:
            (batch, time, feature_dim)

        returns:
            distances to prototypes
        """

        # reshape for broadcasting
        x_expanded = x.unsqueeze(-2)

        proto_expanded = self.prototypes.unsqueeze(0).unsqueeze(0)

        # squared euclidean distance
        dist = torch.sum(
            (x_expanded - proto_expanded) ** 2,
            dim=-1
        )

        return dist



# ------------------------------------------------------------
# Softmax weighting over prototypes
# ------------------------------------------------------------

class PrototypeAttention(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, distances):

        weights = F.softmax(
            -distances,
            dim=-1
        )

        return weights



# ------------------------------------------------------------
# Hedge policy network
# ------------------------------------------------------------

class HedgeNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        num_prototypes,
        hidden_units=[32, 32]
    ):
        super().__init__()

        self.feature_net = DenseBlock(
            input_dim=input_dim,
            hidden_units=hidden_units
        )

        self.prototype_layer = PrototypeLayer(
            num_prototypes=num_prototypes,
            feature_dim=hidden_units[-1]
        )

        self.attention = PrototypeAttention()

        self.output_layer = nn.Sequential(
            nn.Linear(num_prototypes, 1),
            SoftClip(-1, 1)
        )

    def forward(self, x):

        """
        x shape:
            (batch, time, features)
        """

        features = self.feature_net(x)

        distances = self.prototype_layer(features)

        weights = self.attention(distances)

        hedge = self.output_layer(weights)

        return hedge



# ------------------------------------------------------------
# Full ProtoHedge model
# ------------------------------------------------------------

class ProtoHedge(nn.Module):

    def __init__(
        self,
        input_dim,
        num_prototypes=8,
        hidden_units=[32, 32]
    ):
        super().__init__()

        self.policy = HedgeNetwork(
            input_dim=input_dim,
            num_prototypes=num_prototypes,
            hidden_units=hidden_units
        )

    def forward(self, features):

        hedge = self.policy(features)

        return hedge
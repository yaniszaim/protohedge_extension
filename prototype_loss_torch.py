import torch


def prototype_diversity_loss(prototypes):

    """
    encourage prototypes to be different
    """

    n = prototypes.shape[0]

    loss = 0.0

    for i in range(n):

        for j in range(i + 1, n):

            dist = torch.sum(
                (prototypes[i] - prototypes[j])**2
            )

            loss += torch.exp(-dist)

    return loss


def prototype_l2_loss(prototypes):

    """
    prevent exploding prototype values
    """

    return torch.mean(
        prototypes**2
    )
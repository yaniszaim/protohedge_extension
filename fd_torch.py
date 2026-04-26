import torch


def finite_difference_delta(price):

    """
    simple baseline hedge:
    hedge = price change sensitivity
    """

    price_diff = price[:, 1:] - price[:, :-1]

    delta = torch.zeros_like(price)

    delta[:, :-1] = price_diff

    return delta
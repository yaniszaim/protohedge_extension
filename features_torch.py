import torch


def compute_features(

    price,

    prev_hedge=None

):

    """
    ProtoHedge feature construction
    """

    safe_price = torch.clamp(

        price,

        min=1e-6

    )

    log_price = torch.log(

        safe_price

    )


    # returns

    returns = (

        log_price[:, 1:, :]

        - log_price[:, :-1, :]

    )

    zero_pad = torch.zeros_like(

        returns[:, 0:1, :]

    )

    returns = torch.cat(

        [

            zero_pad,

            returns

        ],

        dim=1

    )


    # volatility estimate

    vol = torch.std(

        returns,

        dim=1,

        keepdim=True,

        unbiased=False

    )

    vol = vol.repeat(

        1,

        price.shape[1],

        1

    )


    # normalized time feature

    T = price.shape[1]

    time_grid = torch.linspace(

        0,

        1,

        T,

        device=price.device

    )

    time_feature = time_grid.unsqueeze(0).unsqueeze(-1)

    time_feature = time_feature.repeat(

        price.shape[0],

        1,

        1

    )


    # ensure prev hedge exists

    if prev_hedge is None:

        prev_hedge = torch.zeros_like(

            price

        )


    # concatenate all features

    features = torch.cat(

        [

            log_price,

            returns,

            vol,

            time_feature,

            prev_hedge

        ],

        dim=2

    )


    return features

def compute_proto_features(price, prev_hedge=None):
    """
    Features that match the original prototype BS experiment:
    [price, delta(previous hedge), time_left]
    """

    if prev_hedge is None:
        prev_hedge = torch.zeros_like(price)

    T = price.shape[1]

    time_left = torch.linspace(
        1.0,
        0.0,
        T,
        device=price.device
    ).unsqueeze(0).unsqueeze(-1).repeat(price.shape[0], 1, 1)

    return torch.cat(
        [
            price,
            prev_hedge,
            time_left
        ],
        dim=2
    )
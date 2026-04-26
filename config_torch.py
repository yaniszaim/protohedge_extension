def default_config():

    return {

        "world": {

            "steps": 10,
            "samples": 1000,
            "seed": 1234,

            "dt": 0.1,

            "cost_s": 0.0,
            "cost_v": 0.0,

            "ubnd_as": 5.0,
            "ubnd_av": 5.0,

            "volvol_ivol": 0.5,
            "volvol_rvol": 0.5,

            "strike": 1.0,
            "ttm": 1.0,

            "drift": 0.0,
            "volatility": 0.2,

            "payoff": "atmcall"

        },

        "model": {

            "num_prototypes": 8,
            "hidden_units": [32, 32]

        },

        "training": {

            "epochs": 200,
            "lr": 1e-3

        },

        "objective": {

            "risk_measure": "entropic",
            "risk_aversion": 1.0

        }

    }
import numpy as np
from cdxbasics.prettydict import PrettyDict as pdct

from .base import (
    Logger,
    Config,
    dh_dtype,
    tf,
    tf_dict,
    assert_iter_not_is_nan,
    DIM_DUMMY,
)

_log = Logger(__file__)


def _compute_short_call_payoff(
    spot,
    strike,
    liability_type="european_call",
    asian_average_type="arithmetic",
    asian_start_step=0,
    asian_end_step=None,
    dtype=np.float32,
):
    liability_type = str(liability_type).lower()
    if liability_type in ["european", "european_call", "atm_short_call"]:
        underlying = spot[:, -1]
    elif liability_type in ["asian", "asian_call", "asian_short_call"]:
        avg_type = str(asian_average_type).lower()
        if avg_type not in ["arithmetic", "mean"]:
            _log.throw("Unsupported asian_average_type '%s'", asian_average_type)
        n_steps = int(spot.shape[1])
        start = max(0, int(asian_start_step))
        end = n_steps if asian_end_step in [None, ""] else min(n_steps, int(asian_end_step))
        _log.verify(end > start, "Invalid Asian averaging window start=%ld end=%ld n_steps=%ld", start, end, n_steps)
        underlying = spot[:, start:end].mean(axis=1)
    else:
        _log.throw("Unknown liability_type '%s'", liability_type)
    payoff = -np.maximum(underlying - strike, 0.0).astype(dtype)
    return payoff, underlying.astype(dtype)


class RealWorld_Spot_ATM(object):
    """
    Real-data world for Deep Hedging / ProtoHedge.

    Expected input file:
        training_paths.npy

    Expected shape:
        (nSamples, nSteps, 5)

    Feature order:
        0 = spot
        1 = call_price
        2 = call_delta
        3 = call_vega
        4 = ivol
    """

    def __init__(self, config: Config, dtype=dh_dtype):
        self.tf_dtype = dtype
        self.np_dtype = dtype.as_numpy_dtype()
        self.config = config.copy()
        self.unique_id = "real_world_spot_atm"

        # -------------------------
        # Config
        # -------------------------
        data_path = config("data_path", "training_paths.npy", str, help="Path to real-data training paths")
        cost_s = config("cost_s", 0.0002, float, help="Spot trading cost coefficient")
        cost_v = config("cost_v", 0.02, float, help="Option vega cost coefficient")
        cost_p = config("cost_p", 0.0005, float, help="Option price cost coefficient")
        ubnd_as = config("ubnd_as", 5.0, float, help="Upper bound for spot trades")
        lbnd_as = config("lbnd_as", -5.0, float, help="Lower bound for spot trades")
        ubnd_av = config("ubnd_av", 5.0, float, help="Upper bound for option trades")
        lbnd_av = config("lbnd_av", -5.0, float, help="Lower bound for option trades")
        dt = config("dt", 1.0 / 252.0, float, help="Time step size")
        strike_mode = config("strike_mode", "atm_start", str, help="How to define strike: atm_start or unit")
        liability_type = config("liability_type", "european_call", str, help="Liability payoff: european_call or asian_call")
        asian_average_type = config("asian_average_type", "arithmetic", str, help="Asian averaging type (currently arithmetic only)")
        asian_start_step = config("asian_start_step", 0, int, help="First step index included in the Asian averaging window")
        asian_end_step = config("asian_end_step", None, help="Exclusive end step of the Asian averaging window; defaults to full path")
        config.done()

        # -------------------------
        # Load real paths
        # -------------------------
        data = np.load(data_path).astype(self.np_dtype)

        _log.verify(len(data.shape) == 3, "Expected 3D array (nSamples, nSteps, nFeatures), got %s", str(data.shape))
        _log.verify(data.shape[2] >= 5, "Expected at least 5 features, got %ld", data.shape[2])

        self.nSamples = int(data.shape[0])
        self.nSteps = int(data.shape[1])
        self.nInst = 2
        self.dt = float(dt)
        self.timeline = np.linspace(0.0, self.nSteps * self.dt, self.nSteps + 1, dtype=np.float32)

        spot = data[:, :, 0]
        call_price = data[:, :, 1]
        call_delta = data[:, :, 2]
        call_vega = data[:, :, 3]
        ivol = data[:, :, 4]

        # -------------------------
        # Build time features
        # -------------------------
        time_left = np.linspace(float(self.nSteps), 1.0, self.nSteps, endpoint=True, dtype=self.np_dtype) * self.dt
        sqrt_time_left = np.sqrt(time_left)

        time_left_2d = np.full((self.nSamples, self.nSteps), time_left[np.newaxis, :], dtype=self.np_dtype)
        sqrt_time_left_2d = np.full((self.nSamples, self.nSteps), sqrt_time_left[np.newaxis, :], dtype=self.np_dtype)

        # -------------------------
        # Instrument P&L increments
        # -------------------------
        # Spot increment
        dS = np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype)
        dS[:, :-1] = spot[:, 1:] - spot[:, :-1]
        dS[:, -1] = 0.0

        # Option increment
        dC = np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype)
        dC[:, :-1] = call_price[:, 1:] - call_price[:, :-1]
        dC[:, -1] = 0.0

        # -------------------------
        # Trading costs
        # -------------------------
        cost_dS = cost_s * np.abs(spot)
        cost_dC = cost_v * np.abs(call_vega) + cost_s * np.abs(call_delta) + cost_p * np.abs(call_price)

        dInsts = np.zeros((self.nSamples, self.nSteps, 2), dtype=self.np_dtype)
        cost = np.zeros((self.nSamples, self.nSteps, 2), dtype=self.np_dtype)
        price = np.zeros((self.nSamples, self.nSteps, 2), dtype=self.np_dtype)
        ubnd_a = np.zeros((self.nSamples, self.nSteps, 2), dtype=self.np_dtype)
        lbnd_a = np.zeros((self.nSamples, self.nSteps, 2), dtype=self.np_dtype)

        dInsts[:, :, 0] = dS
        dInsts[:, :, 1] = dC

        cost[:, :, 0] = cost_dS
        cost[:, :, 1] = cost_dC

        price[:, :, 0] = spot
        price[:, :, 1] = call_price

        ubnd_a[:, :, 0] = ubnd_as
        ubnd_a[:, :, 1] = ubnd_av

        lbnd_a[:, :, 0] = lbnd_as
        lbnd_a[:, :, 1] = lbnd_av

        # -------------------------
        # Terminal payoff of the short option position
        # -------------------------
        if strike_mode == "atm_start":
            # strike = initial spot of each path
            strike = spot[:, 0]
        elif strike_mode == "unit":
            # normalized world convention
            strike = np.ones((self.nSamples,), dtype=self.np_dtype)
        else:
            _log.throw("Unknown strike_mode '%s'", strike_mode)

        payoff, liability_underlying = _compute_short_call_payoff(
            spot=spot,
            strike=strike,
            liability_type=liability_type,
            asian_average_type=asian_average_type,
            asian_start_step=asian_start_step,
            asian_end_step=asian_end_step,
            dtype=self.np_dtype,
        )

        # -------------------------
        # Store world data
        # -------------------------
        self.data = pdct()

        self.data.market = pdct(
            hedges=dInsts,
            cost=cost,
            ubnd_a=ubnd_a,
            lbnd_a=lbnd_a,
            payoff=payoff,
        )

        self.data.features = pdct(
            per_step=pdct(
                cost=cost,
                price=price,
                ubnd_a=ubnd_a,
                lbnd_a=lbnd_a,
                time_left=time_left_2d,
                sqrt_time_left=sqrt_time_left_2d,
                spot=spot,
                ivol=ivol,
                call_price=call_price,
                call_delta=call_delta,
                call_vega=call_vega,
                cost_v=cost_dC,
            ),
            per_path=pdct(),
        )

        self.data.features.per_path[DIM_DUMMY] = (payoff * 0.0)[:, np.newaxis]
        self.data.features.per_path["strike"] = strike[:, np.newaxis]
        self.data.features.per_path["liability_underlying"] = liability_underlying[:, np.newaxis]

        assert_iter_not_is_nan(self.data, "data")

        self.tf_data = tf_dict(
            features=self.data.features,
            market=self.data.market,
            dtype=self.tf_dtype,
        )

        self.details = pdct(
            spot_all=spot,
            drift=np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype),
            rvol=np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype),
            strike=strike,
            liability_underlying=liability_underlying,
        )

        assert_iter_not_is_nan(self.details, "details")

        self.sample_weights = np.full((self.nSamples, 1), 1.0 / float(self.nSamples), dtype=self.np_dtype)
        self.tf_sample_weights = tf.constant(self.sample_weights, dtype=self.tf_dtype)
        self.sample_weights = self.sample_weights.reshape((self.nSamples,))
        self.tf_y = tf.zeros((self.nSamples,), dtype=self.tf_dtype)

        self.inst_names = ["spot", "ATM Call"]

    def clone(self, config_overwrite=Config(), **kwargs):
        config = self.config.copy()
        config.update(config_overwrite, **kwargs)
        return RealWorld_Spot_ATM(config)

    def plot(self, *args, **kwargs):
        print("RealWorld_Spot_ATM: plotting not implemented yet.")

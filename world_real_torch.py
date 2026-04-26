"""
Real-data world for the PyTorch Deep Hedging / ProtoHedge pipeline.
"""

from pathlib import Path

import numpy as np

from deephedging.base_torch import Config, DIM_DUMMY, Logger, assert_iter_not_is_nan, pdct

_log = Logger(__file__)


class _ConfigAdapter(dict):
    def __call__(self, key, default=None, *args, **kwargs):
        return self.get(key, default)

    def copy(self):
        return _ConfigAdapter(self)

    def input_dict(self):
        return dict(self)

    def usage_report(self):
        return {}

    def input_report(self):
        return {}

    def done(self):
        return None


def _as_config(config):
    if isinstance(config, dict):
        return _ConfigAdapter(config)
    return config


def _resolve_data_path(path):
    path = Path(path)
    repo_root = Path(__file__).resolve().parent
    candidates = [path]
    if not path.is_absolute():
        candidates.append(repo_root / path)
        candidates.append(repo_root / "Data" / path)
        candidates.append(repo_root / "Data" / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find real-data path '{path}'")


class RealWorld_Spot_ATM_Torch(object):
    """
    Torch equivalent of ``world_real.RealWorld_Spot_ATM``.

    Expected path array shape: ``[nSamples, nSteps, >=5]`` with feature order:
    spot, call_price, call_delta, call_vega, ivol.
    """

    def __init__(self, config: Config, dtype=np.float32):
        config = _as_config(config)
        self.np_dtype = np.dtype(dtype)
        self.config = config.copy()
        self.unique_id = "real_world_spot_atm_torch"

        data_path = config("data_path", "Data/training_paths.npy", str, help="Path to real-data training paths")
        cost_s = config("cost_s", 0.0002, float, help="Spot trading cost coefficient")
        cost_v = config("cost_v", 0.02, float, help="Option vega cost coefficient")
        cost_p = config("cost_p", 0.0005, float, help="Option price cost coefficient")
        ubnd_as = config("ubnd_as", 5.0, float, help="Upper bound for spot trades")
        lbnd_as = config("lbnd_as", -5.0, float, help="Lower bound for spot trades")
        ubnd_av = config("ubnd_av", 5.0, float, help="Upper bound for option trades")
        lbnd_av = config("lbnd_av", -5.0, float, help="Lower bound for option trades")
        dt = config("dt", 1.0 / 252.0, float, help="Time step size")
        strike_mode = config("strike_mode", "atm_start", str, help="Strike mode: atm_start or unit")
        normalize = config("normalize", False, bool, help="Normalize prices by each path's initial spot")
        hedge_mode = config("hedge_mode", "step", str, help="Hedge PnL mode: step or terminal")
        position_bounds = config("position_bounds", False, bool, help="Expose cumulative position bounds to the gym")
        ubnd_delta_s = config("ubnd_delta_s", 2.0, float, help="Upper cumulative spot position bound")
        lbnd_delta_s = config("lbnd_delta_s", -2.0, float, help="Lower cumulative spot position bound")
        ubnd_delta_v = config("ubnd_delta_v", 2.0, float, help="Upper cumulative option position bound")
        lbnd_delta_v = config("lbnd_delta_v", -2.0, float, help="Lower cumulative option position bound")
        samples = config("samples", None, help="Optional number of paths to load")
        seed = config("seed", 1234, int, help="Sampling seed")
        sample_start = config("sample_start", 0, int, help="First path index for deterministic slicing")
        shuffle = config("shuffle", True, bool, help="Whether to randomly subsample when samples is set")
        sample_indices = config("sample_indices", None, help="Explicit sample indices")
        if hasattr(config, "done"):
            config.done()

        data = np.load(_resolve_data_path(data_path)).astype(self.np_dtype)
        _log.verify(len(data.shape) == 3, "Expected 3D array, got %s", str(data.shape))
        _log.verify(data.shape[2] >= 5, "Expected at least 5 features, got %ld", data.shape[2])

        if sample_indices is not None:
            indices = np.asarray(sample_indices, dtype=np.int64)
            data = data[indices]
        elif samples is not None:
            samples = int(samples)
            _log.verify(samples > 0, "'samples' must be positive")
            _log.verify(samples <= data.shape[0], "'samples' cannot exceed available paths")
            if shuffle:
                rng = np.random.default_rng(int(seed))
                indices = rng.choice(data.shape[0], size=samples, replace=False)
                indices.sort()
                data = data[indices]
            else:
                start = int(sample_start)
                end = start + samples
                _log.verify(end <= data.shape[0], "Requested sample slice exceeds available paths")
                data = data[start:end]

        if not np.isfinite(data).all():
            raise ValueError("Real-data paths contain NaN or infinite values")

        self.nSamples = int(data.shape[0])
        self.nSteps = int(data.shape[1])
        self.nInst = 2
        self.dt = float(dt)
        self.normalize = bool(normalize)
        self.hedge_mode = str(hedge_mode).lower()
        self.position_bounds = bool(position_bounds)
        self.timeline = np.linspace(0.0, self.nSteps * self.dt, self.nSteps + 1, dtype=np.float32)

        spot_raw = data[:, :, 0]
        call_price_raw = data[:, :, 1]
        call_delta = data[:, :, 2]
        call_vega_raw = data[:, :, 3]
        ivol = data[:, :, 4]

        if normalize:
            scale = np.maximum(np.abs(spot_raw[:, :1]), np.asarray(1e-8, dtype=self.np_dtype))
            spot = spot_raw / scale
            call_price = call_price_raw / scale
            call_vega = call_vega_raw / scale
        else:
            scale = np.ones((self.nSamples, 1), dtype=self.np_dtype)
            spot = spot_raw
            call_price = call_price_raw
            call_vega = call_vega_raw

        time_left = np.linspace(float(self.nSteps), 1.0, self.nSteps, endpoint=True, dtype=self.np_dtype) * self.dt
        sqrt_time_left = np.sqrt(time_left)
        time_left_2d = np.full((self.nSamples, self.nSteps), time_left[np.newaxis, :], dtype=self.np_dtype)
        sqrt_time_left_2d = np.full((self.nSamples, self.nSteps), sqrt_time_left[np.newaxis, :], dtype=self.np_dtype)

        hedge_mode = self.hedge_mode
        if hedge_mode in ["step", "period", "one_step"]:
            dS = np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype)
            dS[:, :-1] = spot[:, 1:] - spot[:, :-1]
            dS[:, -1] = 0.0

            dC = np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype)
            dC[:, :-1] = call_price[:, 1:] - call_price[:, :-1]
            dC[:, -1] = 0.0
        elif hedge_mode in ["terminal", "maturity", "to_maturity"]:
            dS = spot[:, -1:] - spot
            dC = call_price[:, -1:] - call_price
        else:
            raise ValueError(f"Unknown hedge_mode '{hedge_mode}'")

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

        if strike_mode == "atm_start":
            strike = spot[:, 0]
        elif strike_mode == "unit":
            strike = np.ones((self.nSamples,), dtype=self.np_dtype)
        else:
            raise ValueError(f"Unknown strike_mode '{strike_mode}'")

        payoff = -np.maximum(spot[:, -1] - strike, 0.0).astype(self.np_dtype)

        self.data = pdct()
        self.data.market = pdct(
            hedges=dInsts,
            cost=cost,
            ubnd_a=ubnd_a,
            lbnd_a=lbnd_a,
            payoff=payoff,
        )
        if position_bounds:
            ubnd_delta = np.zeros((self.nSamples, self.nSteps, 2), dtype=self.np_dtype)
            lbnd_delta = np.zeros((self.nSamples, self.nSteps, 2), dtype=self.np_dtype)
            ubnd_delta[:, :, 0] = ubnd_delta_s
            ubnd_delta[:, :, 1] = ubnd_delta_v
            lbnd_delta[:, :, 0] = lbnd_delta_s
            lbnd_delta[:, :, 1] = lbnd_delta_v
            self.data.market.ubnd_delta = ubnd_delta
            self.data.market.lbnd_delta = lbnd_delta

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
                spot_raw=spot_raw,
                call_price_raw=call_price_raw,
                call_vega_raw=call_vega_raw,
                price_scale=np.repeat(scale, self.nSteps, axis=1),
            ),
            per_path=pdct(),
        )
        self.data.features.per_path[DIM_DUMMY] = (payoff * 0.0)[:, np.newaxis]
        assert_iter_not_is_nan(self.data, "data")

        self.torch_data = dict(features=self.data.features, market=self.data.market)
        self.tf_data = self.torch_data

        self.details = pdct(
            spot_all=spot,
            spot_raw=spot_raw,
            call_price_raw=call_price_raw,
            drift=np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype),
            rvol=np.zeros((self.nSamples, self.nSteps), dtype=self.np_dtype),
            ivol=ivol,
        )
        assert_iter_not_is_nan(self.details, "details")

        self.sample_weights = np.full((self.nSamples, 1), 1.0 / float(self.nSamples), dtype=self.np_dtype)
        self.torch_sample_weights = self.sample_weights.copy()
        self.sample_weights = self.sample_weights.reshape((self.nSamples,))
        self.torch_y = np.zeros((self.nSamples,), dtype=self.np_dtype)
        self.tf_sample_weights = self.torch_sample_weights
        self.tf_y = self.torch_y
        self.inst_names = ["spot", "ATM Call"]

    def clone(self, config_overwrite=Config(), **kwargs):
        if "seed" not in kwargs:
            kwargs["seed"] = int(np.random.randint(0, 0x7FFFFFFF))
        config = self.config.copy()
        if hasattr(config, "update"):
            config.update(config_overwrite, **kwargs)
        else:
            if isinstance(config_overwrite, dict):
                config.update(config_overwrite)
            config.update(kwargs)
        return RealWorld_Spot_ATM_Torch(config)

    def plot(self, *args, **kwargs):
        print("RealWorld_Spot_ATM_Torch: plotting not implemented yet.")

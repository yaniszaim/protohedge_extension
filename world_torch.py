"""
Deep Hedging example worlds, NumPy/PyTorch-friendly port.
"""

from collections.abc import Mapping
import math

import numpy as np
from scipy.stats import norm

from deephedging.base_torch import Logger, Config, pdct, assert_iter_not_is_nan, DIM_DUMMY
from cdxbasics.dynaplot import figure, colors_tableau
from cdxbasics.util import uniqueHash

_log = Logger(__file__)


class SimpleWorld_Spot_ATM(object):
    def __init__(self, config: Config, dtype=np.float32):
        if isinstance(config, dict):
            class ConfigAdapter(dict):
                def __call__(self, key, default=None, *args, **kwargs):
                    return self.get(key, default)

                def copy(self):
                    return ConfigAdapter(self)

                def input_dict(self):
                    return dict(self)

                def usage_report(self):
                    return {}

                def input_report(self):
                    return {}

            config = ConfigAdapter(config)

        self.np_dtype = np.dtype(dtype)
        self.unique_id = None
        self.config = config.copy()

        nSteps = config("steps", 10, int, help="Number of time steps")
        nSamples = config("samples", 1000, int, help="Number of samples")
        seed = config("seed", 2312414312, int, help="Random seed")
        nIvSteps = config("invar_steps", 5, int, help="Invariant warmup steps")
        dt = config("dt", 1.0 / 50.0, float, help="Time per timestep")
        cost_s = config("cost_s", 0.0002, float, help="Trading cost spot")
        ubnd_as = config("ubnd_as", 5.0, float, help="Upper bound spot trade")
        lbnd_as = config("lbnd_as", -5.0, float, help="Lower bound spot trade")
        bs_mode = config("black_scholes", False, bool, help="Use Black-Scholes world")
        no_svol = config("no_stoch_vol", False, bool, help="Turn off stochastic vol")
        no_sdrift = config("no_stoch_drift", False, bool, help="Turn off stochastic drift")

        strike = config("strike", 1.0, float, help="Relative strike")
        ttm_steps = config("ttm_steps", 4, int, help="Option time to maturity in steps")
        cost_v = config("cost_v", 0.02, float, help="Trading cost vega")
        cost_p = config("cost_p", 0.0005, float, help="Trading cost option")
        ubnd_av = config("ubnd_av", 5.0, float, help="Upper bound option trade")
        lbnd_av = config("lbnd_av", -5.0, float, help="Lower bound option trade")

        payoff_f = config("payoff", "atmcall", help="Payoff function or keyword")
        if payoff_f is None:
            payoff_f = np.zeros((nSamples,))
        elif isinstance(payoff_f, (int, float)):
            payoff_f = np.full((nSamples,), float(payoff_f))
        elif isinstance(payoff_f, str):
            if payoff_f == "atmcall":
                payoff_f = lambda spots: -np.maximum(spots[:, -1] - 1.0, 0.0)
            elif payoff_f == "atmput":
                payoff_f = lambda spots: -np.maximum(1.0 - spots[:, -1], 0.0)
            else:
                _log.throw("Unknown 'payoff' '%s'", payoff_f)

        drift = config("drift", 0.1, float, help="Mean drift")
        kappa_m = config("meanrev_drift", 1.0, float, help="Mean reversion drift")
        xi_m = config("drift_vol", 0.1, float, help="Vol of drift")
        rvol_init = config("rvol", 0.2, float, help="Initial realized vol")
        ivol_init = config("ivol", rvol_init, float, help="Initial implied vol")
        kappa_v = config("meanrev_rvol", 2.0, float, help="Mean reversion realized vol")
        kappa_i = config("meanrev_ivol", 0.1, float, help="Mean reversion implied vol")
        xi_v = config("volvol_rvol", 0.5, float, help="Vol of vol realized")
        xi_i = config("volvol_ivol", 0.5, float, help="Vol of vol implied")

        rho_ms = config("corr_ms", 0.5, float, help="Corr asset/mean")
        rho_vs = config("corr_vs", -0.7, float, help="Corr asset/vol")
        rho_vi = config("corr_vi", 0.8, float, help="Corr implied/realized vol")
        rho_vs_r = config("rcorr_vs", -0.5, float, help="Residual corr")

        self.usage_report = config.usage_report() if hasattr(config, "usage_report") else {}
        self.input_report = config.input_report() if hasattr(config, "input_report") else {}

        if bs_mode:
            strike = 0.0
            ttm_steps = 1
            nIvSteps = 0
            no_sdrift = True
            no_svol = True
        if no_sdrift:
            kappa_m = 0.0
            xi_m = 0.0
        if no_svol:
            kappa_v = 0.0
            kappa_i = 0.0
            xi_v = 0.0
            xi_i = 0.0

        sqrtDt = math.sqrt(dt)
        ttm_steps = ttm_steps if strike > 0.0 else 1
        ttm = ttm_steps * dt
        sqrtTTM = math.sqrt(ttm)
        xi_m = abs(xi_m)
        xi_v = abs(xi_v)
        xi_i = abs(xi_i)
        time_left = np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=self.np_dtype) * dt
        sqrt_time_left = np.sqrt(time_left)

        np.random.seed(seed)
        dW = np.random.normal(size=(nSamples, nSteps + nIvSteps + ttm_steps - 1, 4)).astype(self.np_dtype) * sqrtDt
        dW_s = dW[:, :, 0]
        dW_m = dW[:, :, 0] * rho_ms + math.sqrt(1.0 - rho_ms ** 2) * dW[:, :, 1]
        dW_v = dW[:, :, 0] * rho_vs + math.sqrt(1.0 - rho_vs ** 2) * dW[:, :, 2]
        dW_i = dW[:, :, 2] * rho_vi + math.sqrt(1.0 - rho_vi ** 2) * (
            dW[:, :, 0] * rho_vs_r + math.sqrt(1.0 - rho_vs_r ** 2) * dW[:, :, 3]
        )

        spot = np.zeros((nSamples, nSteps + nIvSteps + ttm_steps), dtype=self.np_dtype)
        rdrift = np.full((nSamples, nSteps + nIvSteps + ttm_steps), drift, dtype=self.np_dtype)
        rvol = np.full((nSamples, nSteps + nIvSteps + ttm_steps), rvol_init, dtype=self.np_dtype)
        ivol = np.full((nSamples, nSteps + nIvSteps + ttm_steps), ivol_init, dtype=self.np_dtype)

        spot[:, 0] = 1.0
        log_ivol_init = np.log(ivol_init)
        log_rvol = np.log(rvol_init)
        log_ivol = log_ivol_init
        mrdrift = 0.0 * dW_m[:, 0]
        expdriftdt = np.exp(drift * dt)
        bStochDrift = kappa_m != 0.0 or xi_m != 0.0
        bStochVol = kappa_v != 0.0 or xi_v != 0.0 or kappa_i != 0.0 or xi_i != 0.0

        for j in range(1, nSteps + nIvSteps + ttm_steps):
            spot[:, j] = spot[:, j - 1] * np.exp(
                rdrift[:, j - 1] * dt + rvol[:, j - 1] * dW_s[:, j - 1] - 0.5 * (rvol[:, j - 1] ** 2) * dt
            )
            spot[:, j] *= expdriftdt / np.mean(spot[:, j])

            if bStochDrift:
                mrdrift = mrdrift - kappa_m * mrdrift * dt + xi_m * dW_m[:, j - 1]
                mrdrift = np.exp(mrdrift * dt)
                mrdrift = np.log(mrdrift / np.mean(mrdrift)) / dt
                rdrift[:, j] = drift + mrdrift

            if bStochVol:
                log_rvol += kappa_v * (log_ivol - log_rvol) * dt + xi_v * dW_v[:, j - 1] - 0.5 * (xi_v ** 2) * dt
                log_ivol += kappa_i * (log_ivol_init - log_ivol) * dt + xi_i * dW_i[:, j - 1] - 0.5 * (xi_i ** 2) * dt
                rvol[:, j] = np.exp(log_rvol)
                ivol[:, j] = np.exp(log_ivol)

        spot = spot[:, nIvSteps:]
        rdrift = rdrift[:, nIvSteps:nIvSteps + nSteps]
        rvol = rvol[:, nIvSteps:nIvSteps + nSteps]
        ivol = ivol[:, nIvSteps:nIvSteps + nSteps]

        ixs = np.argsort(spot[:, nSteps])
        spot = spot[ixs, :]
        rdrift = rdrift[ixs, :]
        rvol = rvol[ixs, :]
        ivol = ivol[ixs, :]

        dS = spot[:, nSteps][:, np.newaxis] - spot[:, :nSteps]
        cost_dS = spot[:, :nSteps] * cost_s

        if strike <= 0.0:
            dInsts = dS[:, :, np.newaxis]
            cost = cost_dS[:, :, np.newaxis]
            price = spot[:, :nSteps]
            ubnd_a = np.full((nSamples, nSteps, 1), ubnd_as, dtype=self.np_dtype)
            lbnd_a = np.full((nSamples, nSteps, 1), lbnd_as, dtype=self.np_dtype)
            call_price = None
            call_delta = None
            call_vega = None
            cost_dC = None
        else:
            mat_spot = spot[:, ttm_steps:ttm_steps + nSteps]
            opt_spot = spot[:, :nSteps]
            payoffs = np.maximum(0.0, mat_spot - strike * opt_spot)
            d1 = (-np.log(strike) + 0.5 * ivol * ivol * ttm) / (ivol * sqrtTTM)
            d2 = d1 - ivol * sqrtTTM
            N1 = norm.cdf(d1)
            N2 = norm.cdf(d2)
            call_price = N1 * opt_spot - N2 * strike * opt_spot
            dC = payoffs - call_price
            call_delta = N1
            call_vega = opt_spot * norm.pdf(d1) * sqrtTTM
            cost_dC = cost_v * np.abs(call_vega) + cost_s * np.abs(call_delta) + cost_p * np.abs(call_price)

            dInsts = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            cost = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            price = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            ubnd_a = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            lbnd_a = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            dInsts[:, :, 0] = dS
            dInsts[:, :, 1] = dC
            cost[:, :, 0] = cost_dS
            cost[:, :, 1] = cost_dC
            price[:, :, 0] = spot[:, :nSteps]
            price[:, :, 1] = call_price
            ubnd_a[:, :, 0] = ubnd_as
            ubnd_a[:, :, 1] = ubnd_av
            lbnd_a[:, :, 0] = lbnd_as
            lbnd_a[:, :, 1] = lbnd_av

        if not isinstance(payoff_f, np.ndarray):
            payoff = payoff_f(spot[:, :nSteps + 2])
            py_feat = None
            if isinstance(payoff, Mapping):
                py_feat = np.asarray(payoff["features"])
                payoff = np.asarray(payoff["payoff"])
                py_feat = py_feat[:, 0] if len(py_feat) == 2 else py_feat
            else:
                payoff = np.asarray(payoff)
            payoff = payoff[:, 0] if payoff.shape == (nSamples, 1) else payoff
        else:
            payoff = payoff_f
            py_feat = None

        self.unique_id = uniqueHash([config.input_dict(), payoff_f, str(self.np_dtype)], parse_functions=True)

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
                time_left=np.full((nSamples, nSteps), time_left[np.newaxis, :], dtype=self.np_dtype),
                sqrt_time_left=np.full((nSamples, nSteps), sqrt_time_left[np.newaxis, :], dtype=self.np_dtype),
                spot=spot[:, :nSteps],
                ivol=ivol,
            ),
            per_path=pdct(),
        )

        if strike > 0.0:
            self.data.features.per_step.update(
                call_price=call_price,
                call_delta=call_delta,
                call_vega=call_vega,
                cost_v=cost_dC,
            )

        if py_feat is not None:
            self.data.features.per_step.payoff_features = py_feat

        self.data.features.per_path[DIM_DUMMY] = (payoff * 0.0)[:, np.newaxis]

        assert_iter_not_is_nan(self.data, "data")

        self.torch_data = dict(features=self.data.features, market=self.data.market)
        self.tf_data = self.torch_data

        self.details = pdct(
            spot_all=spot[:, :nSteps + 1],
            drift=rdrift,
            rvol=rvol,
        )
        assert_iter_not_is_nan(self.details, "details")

        self.sample_weights = np.full((nSamples, 1), 1.0 / float(nSamples), dtype=self.np_dtype)
        self.torch_sample_weights = self.sample_weights.copy()
        self.sample_weights = self.sample_weights.reshape((nSamples,))
        self.torch_y = np.zeros((nSamples,), dtype=self.np_dtype)
        self.tf_sample_weights = self.torch_sample_weights
        self.tf_y = self.torch_y

        self.nSteps = nSteps
        self.nSamples = nSamples
        self.nInst = 1 if strike <= 0.0 else 2
        self.dt = dt
        self.timeline = np.cumsum(np.linspace(0.0, nSteps, nSteps + 1, endpoint=True, dtype=np.float32)) * dt

        self.inst_names = ["spot"]
        if strike > 0.0:
            self.inst_names.append("ATM Call")

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
        return SimpleWorld_Spot_ATM(config)

    def plot(self, config=Config(), **kwargs):
        config.update(kwargs)
        col_size = config.fig("col_size", 5, int, "Figure column size")
        row_size = config.fig("row_size", 5, int, "Figure row size")
        plot_samples = config("plot_samples", 5, int, "Number of samples to plot")

        xSamples = np.linspace(0, self.nSamples, plot_samples, endpoint=False, dtype=int)
        timeline1 = np.cumsum(np.linspace(0.0, self.nSteps, self.nSteps + 1, endpoint=True, dtype=np.float32)) * self.dt
        timeline = timeline1[:-1]

        fig = figure(tight=True, col_size=col_size, row_size=row_size, col_nums=3)
        fig.suptitle(self.__class__.__name__, fontsize=16)

        ax = fig.add_plot()
        ax.set_title("Spot")
        ax.set_xlabel("Time")
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline1, self.details.spot_all[i, :], "-", color=color)

        ax = fig.add_plot()
        ax.set_title("Drift")
        ax.set_xlabel("Time")
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline, self.details.drift[i, :], "-", color=color)

        ax = fig.add_plot()
        ax.set_title("Realized Vol")
        ax.set_xlabel("Time")
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline, self.details.rvol[i, :], "-", color=color)

        return fig

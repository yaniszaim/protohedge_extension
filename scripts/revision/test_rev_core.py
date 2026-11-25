#!/usr/bin/env python3
"""Unit tests for the Phase 3 revision core (run: python -m unittest scripts/revision/test_rev_core.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402  (first: it fixes the thread count before torch loads)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import phase2_diagnostics as p2  # noqa: E402


class ConstantPolicy(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.tensor(value, dtype=torch.float32)

    def forward(self, x, return_weights=False):
        out = self.value.expand(x.shape[0], 2)
        return (out, None) if return_weights else out


class RevCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.val = {liab: rc.Episodes("SPY", liab, "val") for liab in ("european_call", "asian_call")}

    def test_test_split_is_locked(self):
        with self.assertRaises(RuntimeError):
            rc.Episodes("SPY", "european_call", "test")

    def test_residual_zero_correction_is_the_liability_delta_hedge(self):
        for liab, ep in self.val.items():
            data = ep.tensors("s0")
            got = rc.rollout(ConstantPolicy([0.0, 0.0]), data, "residual")["offset"].numpy()
            targets = np.zeros((ep.n, rc.N_STEPS, 2))
            targets[:, :, 0] = ep.np["liab_delta"]
            want = p2.simulate_targets(ep.world.data.market, targets)["offset"]
            np.testing.assert_allclose(got, want, atol=2e-6, err_msg=liab)

    def test_target_constant_is_buy_and_hold(self):
        ep = self.val["european_call"]
        data = ep.tensors("s0")
        got = rc.rollout(ConstantPolicy([1.0, 0.0]), data, "target")["offset"].numpy()
        want = p2.simulate_targets(ep.world.data.market, p2.constant_targets(ep.n, 1.0, 0.0))["offset"]
        np.testing.assert_allclose(got, want, atol=2e-6)

    def test_trade_mode_accumulates_and_respects_limits(self):
        ep = self.val["european_call"]
        r = rc.rollout(ConstantPolicy([0.3, -0.3]), ep.tensors("s0"), "trade", return_paths=True)
        pos = r["positions"].numpy()
        np.testing.assert_allclose(pos[0, :5, 0], [0.3, 0.6, 0.9, 1.0, 1.0], atol=1e-6)
        self.assertLessEqual(np.abs(pos).max(), 1.0 + 1e-6)

    def test_clamp_keeps_gradient(self):
        x = torch.tensor([2.0], requires_grad=True)
        rc.clamp_ste(x, -1.0, 1.0).sum().backward()
        self.assertEqual(float(x.grad), 1.0)

    def test_oce_optimum_equals_exact_cvar(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(4001) * 0.03
        for name in ("cvar50_offset", "cvar10_he"):
            obj = rc.Objective(name)
            opt = torch.optim.Adam(obj.parameters(), lr=1e-3)
            xt, zero = torch.tensor(x, dtype=torch.float32), torch.zeros(len(x))
            for _ in range(3000):
                opt.zero_grad()
                loss = obj(xt, zero)
                loss.backward()
                opt.step()
            self.assertAlmostEqual(-float(loss), obj.exact_value(x, np.zeros(len(x))), delta=2e-4)

    def test_exact_cvar_handles_ties(self):
        x = np.array([-1.0] + [0.0] * 9)
        self.assertAlmostEqual(float(p2.lower_cvar(x, 0.5)), -0.2)

    def test_features_use_no_future_information(self):
        ep = self.val["asian_call"]
        spot = ep.features["spot"]
        # running-average moneyness at step t depends only on fixings 1..t
        avg = np.array([spot[:, 1:t + 1].mean(axis=1) if t > 0 else np.zeros(ep.n) for t in range(rc.N_STEPS)]).T
        np.testing.assert_allclose(ep.features["asian_moneyness"], avg, atol=1e-5)
        self.assertTrue(np.all(ep.np["liab_delta"] >= 0) and np.all(ep.np["liab_delta"] <= 1))

    def test_prototypes_are_real_visited_states(self):
        ep = rc.Episodes("SPY", "european_call", "train")
        scaler = rc.Scaler(ep, "s0")
        torch.manual_seed(0)
        dh = rc.DHPolicy(scaler)
        data = ep.tensors("s0")
        bank = rc.extract_prototypes(dh, data, "target", scaler, k=10, seed=0)
        self.assertEqual(bank["prototypes_scaled"].shape, (10, 5))
        i, t = int(bank["source_path"][0]), int(bank["source_step"][0])
        np.testing.assert_allclose(bank["prototypes_raw"][0, :3], data["x"][i, t].numpy(), atol=1e-6)
        ph = rc.PHPolicy(scaler, bank["prototypes_scaled"])
        out, w = ph(torch.cat([data["x"][:7, 0], torch.zeros(7, 2)], dim=1), return_weights=True)
        self.assertEqual(tuple(w.shape), (7, 10))
        self.assertLessEqual(float(out.abs().max()), 1.0)


if __name__ == "__main__":
    unittest.main()

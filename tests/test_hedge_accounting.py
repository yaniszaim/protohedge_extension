import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deephedging.gym_torch import DeepHedgingGymTorch
from deephedging.hedge_accounting import STEP_RETURN_MARKER
from deephedging.real_data_analysis_torch import ARTIFACT_META, load_model_artifact
from deephedging.real_data_sweep_torch import _evaluate_spot_delta_rule
from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch


class _SequenceAgent(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.register_buffer("action_sequence", torch.as_tensor(actions, dtype=torch.float32))
        self.step = 0

    def forward(self, x, return_weights=False):
        action = self.action_sequence[self.step].expand(x.shape[0], -1)
        self.step += 1
        return (action, None) if return_weights else action


class _PassthroughObjective(nn.Module):
    def forward(self, payoff, pnl, cost):
        gains = payoff + pnl - cost
        return {
            "loss": -gains.mean(),
            "loss_path": -gains,
            "utility": gains,
            "utility0": payoff,
            "gains": gains,
        }


def _gym_data(hedges, step_returns):
    hedges = torch.as_tensor(hedges, dtype=torch.float32)[None, :, None]
    n_batch, n_steps, n_inst = hedges.shape
    market = {
        "hedges": hedges,
        "cost": torch.zeros_like(hedges),
        "ubnd_a": torch.full_like(hedges, 10.0),
        "lbnd_a": torch.full_like(hedges, -10.0),
        "payoff": torch.zeros(n_batch),
    }
    if step_returns:
        market[STEP_RETURN_MARKER] = torch.ones(n_batch)
    return {
        "market": market,
        "features": {
            "per_step": {"time_left": torch.zeros(n_batch, n_steps)},
            "per_path": {},
        },
    }


class HedgeAccountingTests(unittest.TestCase):
    def test_terminal_return_identity_matches_stepwise_inventory(self):
        rng = np.random.default_rng(7)
        prices = rng.normal(size=(5, 9, 2)).cumsum(axis=1)
        actions = rng.normal(size=(5, 8, 2))

        step_returns = np.diff(prices, axis=1)
        positions = np.cumsum(actions, axis=1)
        terminal_returns = prices[:, -1:, :] - prices[:, :-1, :]

        stepwise_pnl = np.sum(positions * step_returns, axis=(1, 2))
        original_pnl = np.sum(actions * terminal_returns, axis=(1, 2))
        np.testing.assert_allclose(original_pnl, stepwise_pnl, rtol=1e-12, atol=1e-12)

    def test_torch_gym_uses_inventory_for_one_step_returns(self):
        actions = [[1.0], [1.0], [-2.0]]
        gym = DeepHedgingGymTorch(
            agent=_SequenceAgent(actions),
            objective=_PassthroughObjective(),
            feature_names=["time_left"],
        )

        result = gym(_gym_data([10.0, 20.0, 0.0], step_returns=True))

        self.assertAlmostEqual(float(result["pnl"][0]), 50.0)
        self.assertIn("liability_offset", result)
        torch.testing.assert_close(
            result["liability_offset"],
            result["gains"],
        )
        np.testing.assert_allclose(
            result["deltas"][0, :, 0].detach().numpy(),
            np.array([1.0, 2.0, 0.0]),
        )

    def test_torch_gym_preserves_original_terminal_return_accounting(self):
        actions = [[1.0], [1.0], [-2.0]]
        gym = DeepHedgingGymTorch(
            agent=_SequenceAgent(actions),
            objective=_PassthroughObjective(),
            feature_names=["time_left"],
        )

        result = gym(_gym_data([30.0, 20.0, 0.0], step_returns=False))

        self.assertAlmostEqual(float(result["pnl"][0]), 50.0)

    def test_real_world_marks_only_one_step_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "paths.npy"
            np.save(data_path, self._sample_path())

            step_world = RealWorld_Spot_ATM_Torch(
                {"data_path": str(data_path), "hedge_mode": "step", "normalize": False}
            )
            terminal_world = RealWorld_Spot_ATM_Torch(
                {"data_path": str(data_path), "hedge_mode": "terminal", "normalize": False}
            )

        self.assertEqual(step_world.pnl_accounting, "inventory_step")
        self.assertIn(STEP_RETURN_MARKER, step_world.data.market)
        np.testing.assert_allclose(
            step_world.data.market.hedges[0, :, 0],
            np.array([10.0, 20.0]),
        )
        self.assertEqual(step_world.nSteps, 2)
        self.assertEqual(step_world.nObservations, 3)

        self.assertEqual(terminal_world.pnl_accounting, "trade_to_terminal")
        self.assertNotIn(STEP_RETURN_MARKER, terminal_world.data.market)
        np.testing.assert_allclose(
            terminal_world.data.market.hedges[0, :, 0],
            np.array([30.0, 20.0]),
        )

    def test_spot_delta_baseline_credits_carried_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "paths.npy"
            np.save(data_path, self._sample_path())
            _, result, _ = _evaluate_spot_delta_rule(
                data_path=data_path,
                indices=[0],
                world_kwargs={
                    "normalize": False,
                    "hedge_mode": "step",
                    "position_bounds": False,
                },
                band=0.0,
            )

        self.assertAlmostEqual(float(result["pnl"][0]), 30.0)
        np.testing.assert_allclose(
            result["deltas"][0, :, 0],
            np.array([1.0, 1.0]),
        )

    def test_legacy_real_step_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            metadata = {
                "config": {
                    "world": {
                        "world_type": "real",
                        "hedge_mode": "step",
                    }
                }
            }
            (artifact_dir / ARTIFACT_META).write_text(json.dumps(metadata))

            with self.assertRaisesRegex(RuntimeError, "Retrain this real-data model"):
                load_model_artifact(artifact_dir)

    @staticmethod
    def _sample_path():
        # spot, call price, call delta, call vega, implied volatility
        return np.array(
            [
                [
                    [100.0, 5.0, 1.0, 0.2, 0.2],
                    [110.0, 8.0, 1.0, 0.2, 0.2],
                    [130.0, 4.0, 1.0, 0.2, 0.2],
                ]
            ],
            dtype=np.float32,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from deephedging.agents_torch import ProtoHedgeAgent
from deephedging.softclip_torch import legacy_proto_softclip, tfp_softclip


class TfpSoftClipTests(unittest.TestCase):
    def test_matches_tfp_dhsoftclip_reference_values(self):
        actions = torch.tensor(
            [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
            dtype=torch.float64,
        )
        expected = torch.tensor(
            [
                -0.5081408516520007,
                -0.3071233883234654,
                -0.1939125957923358,
                -0.0754790049440601,
                0.0450470505396723,
                0.1643202238283312,
                0.3866914245933368,
            ],
            dtype=torch.float64,
        )
        actual = tfp_softclip(actions, lower=-1.0, upper=1.0)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_is_bounded_and_differentiable(self):
        actions = torch.linspace(-20.0, 20.0, 101, requires_grad=True)
        bounded = tfp_softclip(actions, lower=-1.0, upper=2.0)
        self.assertTrue(torch.all(bounded >= -1.0))
        self.assertTrue(torch.all(bounded <= 2.0))
        bounded.sum().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())

    def test_legacy_mode_remains_the_default(self):
        torch.manual_seed(7)
        agent = ProtoHedgeAgent(
            input_dim=3,
            num_prototypes=4,
            action_low=[-1.0],
            action_high=[1.0],
        )
        expected = legacy_proto_softclip(
            agent.prototype_actions_unbounded,
            lower=agent.action_low,
            upper=agent.action_high,
        )
        torch.testing.assert_close(agent._bounded_prototype_actions(), expected)

    def test_agent_can_select_exact_mode(self):
        agent = ProtoHedgeAgent(
            input_dim=3,
            num_prototypes=4,
            action_low=[-1.0],
            action_high=[1.0],
            softclip_mode="tfp_exact",
        )
        expected = tfp_softclip(
            agent.prototype_actions_unbounded,
            lower=agent.action_low,
            upper=agent.action_high,
        )
        torch.testing.assert_close(agent._bounded_prototype_actions(), expected)


if __name__ == "__main__":
    unittest.main()

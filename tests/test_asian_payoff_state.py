import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deephedging.base_torch import torchCast
from deephedging.hedge_accounting import HEDGE_ACCOUNTING_VERSION
from deephedging.panel_data_pipeline import (
    EPISODE_TIMING_VERSION,
    OPTION_PATH_VERSION,
    TEMPORAL_SPLIT_VERSION,
)
from deephedging.payoff_state import (
    ASIAN_PAYOFF_FEATURE,
    ASIAN_PAYOFF_STATE_VERSION,
    arithmetic_running_average,
    model_features_for_liability,
)
from deephedging.prototype_extraction_torch import (
    build_prototype_payload,
    extract_agent_feature_matrix,
    save_prototype_payload,
)
from deephedging.real_data_analysis_torch import (
    ARTIFACT_META,
    load_model_artifact,
    prototype_usage_table,
)
from deephedging.run_train_torch import (
    build_experiment_components,
    default_config,
)
from deephedging.world_real_torch import RealWorld_Spot_ATM_Torch


class AsianPayoffStateTests(unittest.TestCase):
    def test_world_exposes_running_average_used_by_payoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "paths.npy"
            np.save(data_path, self._sample_paths())
            world = self._world(data_path)

        running_average = np.asarray(
            world.data.features.per_step["asian_running_average"]
        )
        asian_moneyness = np.asarray(
            world.data.features.per_step[ASIAN_PAYOFF_FEATURE]
        )
        fixing_spot = self._sample_paths()[:, 1:, 0]
        expected_terminal_average = fixing_spot.mean(axis=1)
        expected_average = np.concatenate(
            [
                np.zeros((fixing_spot.shape[0], 1)),
                np.cumsum(fixing_spot[:, :-1], axis=1)
                / np.arange(1.0, fixing_spot.shape[1])[None, :],
            ],
            axis=1,
        )
        expected_moneyness = expected_average / 100.0

        np.testing.assert_allclose(running_average, expected_average)
        np.testing.assert_allclose(asian_moneyness, expected_moneyness)
        np.testing.assert_allclose(
            world.data.features.per_path["liability_underlying"][:, 0],
            expected_terminal_average,
        )
        np.testing.assert_allclose(
            world.data.market.payoff,
            -np.maximum(expected_terminal_average - 100.0, 0.0),
        )
        self.assertEqual(world.payoff_state_version, ASIAN_PAYOFF_STATE_VERSION)

    def test_same_spot_state_retains_distinct_asian_payoff_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "paths.npy"
            paths = self._sample_paths()
            np.save(data_path, paths)
            world = self._world(data_path, normalize=True)
            features = model_features_for_liability("asian_call")
            matrix, sorted_names = extract_agent_feature_matrix(
                world=world,
                result=None,
                feature_names=features,
            )

        # At step 2 both paths have the same spot, listed-call price, inventory,
        # and time remaining, but their accumulated averages differ.
        first = matrix[2]
        second = matrix[world.nSteps + 2]
        self.assertEqual(sorted_names[0], ASIAN_PAYOFF_FEATURE)
        self.assertEqual(matrix.shape[1], 6)
        self.assertNotAlmostEqual(float(first[0]), float(second[0]))
        np.testing.assert_allclose(first[1:], second[1:])

    def test_partial_fixing_window_is_observable_and_carried_forward(self):
        spot = self._sample_paths()[:1, :, 0]
        running, count, fraction = arithmetic_running_average(
            spot,
            start_step=1,
            end_step=4,
        )
        np.testing.assert_allclose(
            running[0],
            np.array([0.0, 80.0, 90.0, 280.0 / 3.0, 280.0 / 3.0]),
        )
        np.testing.assert_allclose(count, np.array([0.0, 1.0, 2.0, 3.0, 3.0]))
        np.testing.assert_allclose(
            fraction,
            np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0]),
        )

    def test_asian_vanilla_and_proto_use_same_six_dimensional_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "paths.npy"
            prototype_path = root / "asian_prototypes.pkl"
            np.save(data_path, self._sample_paths())
            source_world = self._world(data_path, normalize=True)
            features = model_features_for_liability("asian_call")
            payload = build_prototype_payload(
                world=source_world,
                result=None,
                n_prototypes=2,
                feature_names=features,
                random_state=7,
            )
            save_prototype_payload(payload, prototype_path)

            vanilla_cfg = self._experiment_config(
                data_path,
                agent_type="feed_forward",
                feature_names=features,
            )
            proto_cfg = self._experiment_config(
                data_path,
                agent_type="protopnet",
                feature_names=features,
                prototype_path=prototype_path,
            )
            vanilla = build_experiment_components(vanilla_cfg)
            proto = build_experiment_components(proto_cfg)

            self.assertIn(ASIAN_PAYOFF_FEATURE, vanilla["gym"].feature_names)
            self.assertIn(ASIAN_PAYOFF_FEATURE, proto["gym"].feature_names)
            self.assertEqual(vanilla["gym"].agent.input_dim, 6)
            self.assertEqual(proto["gym"].agent.input_dim, 6)
            self.assertEqual(
                payload["payoff_state_version"],
                ASIAN_PAYOFF_STATE_VERSION,
            )
            self.assertIn(ASIAN_PAYOFF_FEATURE, payload["feature_names"])

            result = proto["gym"].forward(
                torchCast(proto["world"].torch_data),
                training=False,
                return_paths=True,
            )
            bundle = {
                "metadata": {
                    "prototype_path": str(prototype_path),
                    "config": proto_cfg,
                },
                "world": proto["world"],
                "gym": proto["gym"],
            }
            top, _ = prototype_usage_table(bundle, result, top_n=2)
            self.assertIn(ASIAN_PAYOFF_FEATURE, top.columns)

    def test_asian_policy_without_running_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "paths.npy"
            np.save(data_path, self._sample_paths())
            cfg = self._experiment_config(
                data_path,
                agent_type="feed_forward",
                feature_names=["price", "delta", "time_left"],
            )
            with self.assertRaisesRegex(ValueError, "Asian-option policies"):
                build_experiment_components(cfg)

    def test_legacy_asian_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            metadata = {
                "hedge_accounting_version": HEDGE_ACCOUNTING_VERSION,
                "temporal_split_version": TEMPORAL_SPLIT_VERSION,
                "option_path_version": OPTION_PATH_VERSION,
                "episode_timing_version": EPISODE_TIMING_VERSION,
                "liability_type": "asian_call",
                "config": {
                    "world": {
                        "world_type": "real",
                        "hedge_mode": "step",
                        "liability_type": "asian_call",
                    },
                    "model": {
                        "feature_list": ["price", "delta", "time_left"],
                    },
                },
            }
            (artifact_dir / ARTIFACT_META).write_text(json.dumps(metadata))

            with self.assertRaisesRegex(RuntimeError, "observable Asian payoff state"):
                load_model_artifact(artifact_dir)

    @staticmethod
    def _sample_paths():
        # Both paths have identical current market observations from step 2
        # onward but different histories before that point.
        spots = np.array(
            [
                [100.0, 80.0, 100.0, 100.0, 100.0],
                [100.0, 120.0, 100.0, 100.0, 100.0],
            ],
            dtype=np.float32,
        )
        paths = np.zeros((2, 5, 5), dtype=np.float32)
        paths[:, :, 0] = spots
        paths[:, :, 1] = np.array([5.0, 6.0, 5.0, 4.0, 3.0], dtype=np.float32)
        paths[:, :, 2] = 0.5
        paths[:, :, 3] = 0.2
        paths[:, :, 4] = 0.25
        return paths

    @staticmethod
    def _world(data_path, normalize=False):
        return RealWorld_Spot_ATM_Torch(
            {
                "data_path": str(data_path),
                "sample_indices": [0, 1],
                "normalize": bool(normalize),
                "hedge_mode": "step",
                "liability_type": "asian_call",
                "asian_average_type": "arithmetic",
                "asian_start_step": 1,
                "asian_end_step": None,
                "payoff_state_version": ASIAN_PAYOFF_STATE_VERSION,
            }
        )

    @staticmethod
    def _experiment_config(
        data_path,
        agent_type,
        feature_names,
        prototype_path=None,
    ):
        cfg = default_config()
        cfg["world"].update(
            {
                "world_type": "real",
                "data_path": str(data_path),
                "sample_indices": [0],
                "val_sample_indices": [1],
                "samples": None,
                "normalize": True,
                "hedge_mode": "step",
                "liability_type": "asian_call",
                "payoff_state_version": ASIAN_PAYOFF_STATE_VERSION,
            }
        )
        cfg["model"].update(
            {
                "agent_type": agent_type,
                "feature_list": list(feature_names),
                "prototype_path": (
                    None if prototype_path is None else str(prototype_path)
                ),
                "prototype_init": None,
            }
        )
        return cfg


if __name__ == "__main__":
    unittest.main()

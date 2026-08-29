import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from deephedging.hedge_accounting import HEDGE_ACCOUNTING_VERSION
from deephedging.panel_data_pipeline import (
    EPISODE_TIMING_VERSION,
    OPTION_PATH_VERSION,
    RAW_FEATURE_ORDER,
    TEMPORAL_SPLIT_VERSION,
    build_chronological_episode_dataset,
    build_episode_tensor,
    load_chronological_splits,
    save_chronological_episode_artifacts,
)
from deephedging.real_data_analysis_torch import ARTIFACT_META, load_model_artifact
from deephedging.real_data_sweep_torch import split_indices


def _feature_frame(n_rows=120):
    dates = pd.bdate_range("2010-01-04", periods=n_rows)
    step = np.arange(n_rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "date": dates,
            "spot": 100.0 + step,
            "call_price": 5.0 + 0.01 * step,
            "call_delta": 0.5 + 0.0001 * step,
            "call_vega": 0.2 + 0.0001 * step,
            "ivol": 0.2 + 0.00001 * step,
        }
    )


class TemporalSplitTests(unittest.TestCase):
    def test_episode_builder_keeps_last_valid_window(self):
        features = _feature_frame(8)
        episodes = build_episode_tensor(features, n_steps=5)
        self.assertEqual(episodes.shape, (3, 6, len(RAW_FEATURE_ORDER)))

    def test_split_before_windows_prevents_cross_split_date_overlap(self):
        features = _feature_frame()
        episodes, metadata, splits, info = build_chronological_episode_dataset(
            features,
            n_steps=5,
            train_end_date=features.loc[59, "date"],
            validation_end_date=features.loc[89, "date"],
            embargo_steps=4,
        )

        self.assertEqual(info["split_version"], TEMPORAL_SPLIT_VERSION)
        self.assertEqual(info["episode_timing_version"], EPISODE_TIMING_VERSION)
        self.assertEqual(len(splits["train"]), 55)
        self.assertEqual(len(splits["val"]), 21)
        self.assertEqual(len(splits["test"]), 21)
        self.assertEqual(len(episodes), 97)

        source_rows = {}
        for split_name in ("train", "val", "test"):
            used = set()
            for row in metadata[metadata["split"] == split_name].itertuples():
                used.update(range(row.start_feature_row, row.end_feature_row + 1))
            source_rows[split_name] = used

        self.assertFalse(source_rows["train"] & source_rows["val"])
        self.assertFalse(source_rows["train"] & source_rows["test"])
        self.assertFalse(source_rows["val"] & source_rows["test"])
        self.assertEqual(min(source_rows["val"]) - max(source_rows["train"]), 5)
        self.assertEqual(min(source_rows["test"]) - max(source_rows["val"]), 5)

    def test_persisted_test_membership_is_seed_independent(self):
        features = _feature_frame()
        episodes, metadata, splits, info = build_chronological_episode_dataset(
            features,
            n_steps=5,
            train_end_date=features.loc[59, "date"],
            validation_end_date=features.loc[89, "date"],
            embargo_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode_path = root / "episodes" / "TEST_training_paths.npy"
            split_path = root / "episodes" / "TEST_chronological_splits.npz"
            metadata_path = root / "episode_metadata" / "TEST_episode_metadata.csv"
            contract_path = root / "contract_paths" / "TEST_contract_paths.csv"
            contract_path.parent.mkdir(parents=True)
            contract_path.write_text("episode_index,step,date,optionid,exdate\n")
            info.update(
                {
                    "option_path_version": OPTION_PATH_VERSION,
                    "contract_path_path": str(contract_path),
                }
            )
            metadata["option_path_version"] = OPTION_PATH_VERSION
            metadata["optionid"] = metadata["episode_index"] + 1
            metadata["option_exdate"] = (
                pd.to_datetime(metadata["end_date"]) + pd.Timedelta(days=30)
            ).dt.date.astype(str)
            metadata["contract_observation_count"] = 6
            metadata["contract_unique_optionids"] = 1
            metadata["contract_unique_exdates"] = 1
            metadata["contract_unique_strikes"] = 1
            metadata["contract_expires_after_episode"] = True
            save_chronological_episode_artifacts(
                episode_path,
                split_path,
                metadata_path,
                episodes,
                splits,
                metadata,
                info,
            )
            first, first_info, _ = load_chronological_splits(
                episode_path,
                split_path,
                metadata_path,
                samples=30,
            )
            second, second_info, _ = load_chronological_splits(
                episode_path,
                split_path,
                metadata_path,
                samples=30,
            )

        for split_name in ("train", "val", "test"):
            np.testing.assert_array_equal(first[split_name], second[split_name])
        self.assertEqual(first_info["split_version"], TEMPORAL_SPLIT_VERSION)
        self.assertEqual(first_info, second_info)

    def test_random_historical_split_api_is_disabled(self):
        with self.assertRaisesRegex(RuntimeError, "overlapping observations"):
            split_indices(100, train_frac=0.7, val_frac=0.15, seed=1)

    def test_legacy_random_split_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            metadata = {
                "hedge_accounting_version": HEDGE_ACCOUNTING_VERSION,
                "episode_timing_version": EPISODE_TIMING_VERSION,
                "config": {
                    "world": {
                        "world_type": "real",
                        "hedge_mode": "step",
                    }
                },
            }
            (artifact_dir / ARTIFACT_META).write_text(json.dumps(metadata))

            with self.assertRaisesRegex(RuntimeError, "fixed chronological pre-window split"):
                load_model_artifact(artifact_dir)

    def test_mixed_contract_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            metadata = {
                "hedge_accounting_version": HEDGE_ACCOUNTING_VERSION,
                "temporal_split_version": TEMPORAL_SPLIT_VERSION,
                "episode_timing_version": EPISODE_TIMING_VERSION,
                "config": {
                    "world": {
                        "world_type": "real",
                        "hedge_mode": "step",
                    }
                },
            }
            (artifact_dir / ARTIFACT_META).write_text(json.dumps(metadata))

            with self.assertRaisesRegex(RuntimeError, "contract-consistent"):
                load_model_artifact(artifact_dir)


if __name__ == "__main__":
    unittest.main()

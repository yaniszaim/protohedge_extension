import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from deephedging.real_data_sweep_torch import (
    _completed_artifact_checkpoint,
    _completed_baseline_checkpoint,
    _validate_resume_config,
    _validate_resume_metric_keys,
)


class SweepResumeTests(unittest.TestCase):
    def _artifact_metrics(self, artifact_dir, model="proto_a"):
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "split": split,
                    "seed": 1234,
                    "risk_measure": "cvar",
                    "artifact_dir": str(artifact_dir),
                }
                for split in ("train", "val")
            ]
        )

    def _write_artifact(self, artifact_dir, model="proto_a"):
        artifact_dir.mkdir(parents=True)
        metadata = {
            "model_name": model,
            "seed": 1234,
            "risk_measure": "cvar",
            "history": {"best_epoch": 17},
        }
        (artifact_dir / "artifact_metadata.json").write_text(
            json.dumps(metadata)
        )
        (artifact_dir / "gym_state.pt").write_bytes(b"saved-state")

    def test_complete_metric_pair_and_artifact_are_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifact"
            self._write_artifact(artifact_dir)
            metrics = self._artifact_metrics(artifact_dir)

            _validate_resume_metric_keys(metrics)
            checkpoint = _completed_artifact_checkpoint(
                metrics,
                model="proto_a",
                seed=1234,
                risk_measure="cvar",
            )

            self.assertIsNotNone(checkpoint)
            rows, saved_dir, metadata = checkpoint
            self.assertEqual(set(rows["split"]), {"train", "val"})
            self.assertEqual(saved_dir, artifact_dir)
            self.assertEqual(metadata["history"]["best_epoch"], 17)

    def test_metric_rows_without_complete_artifact_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "artifact_metadata.json").write_text("{}")
            metrics = self._artifact_metrics(artifact_dir)

            with self.assertRaisesRegex(RuntimeError, "artifact is incomplete"):
                _completed_artifact_checkpoint(
                    metrics,
                    model="proto_a",
                    seed=1234,
                    risk_measure="cvar",
                )

    def test_changed_experiment_configuration_is_rejected(self):
        previous = {"epochs": 800, "seeds": [1234, 2345, 3456]}
        requested = {"epochs": 400, "seeds": [1234, 2345, 3456]}

        with self.assertRaisesRegex(RuntimeError, "different experiment"):
            _validate_resume_config(previous, requested)

    def test_all_six_baseline_rows_are_required(self):
        rows = [
            {
                "model": model,
                "split": split,
                "seed": 1234,
                "risk_measure": None,
                "artifact_dir": None,
            }
            for model in ("unhedged", "spot_delta", "spot_delta_band")
            for split in ("train", "val")
        ]
        metrics = pd.DataFrame(rows)
        self.assertTrue(_completed_baseline_checkpoint(metrics, 1234))

        with self.assertRaisesRegex(RuntimeError, "partial or inconsistent"):
            _completed_baseline_checkpoint(metrics.iloc[:-1], 1234)


if __name__ == "__main__":
    unittest.main()

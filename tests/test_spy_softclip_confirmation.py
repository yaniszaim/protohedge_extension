import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_spy_softclip_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("spy_softclip_confirmation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SpySoftClipConfirmationTests(unittest.TestCase):
    def test_candidate_selection_is_restricted_and_validation_only(self):
        rows = [
            {
                "model": "faithful_k10",
                "prototype_source": "vanilla",
                "weighted_similarity": False,
                "learn_distance_feature_weights": False,
                "has_all_seed_runs": True,
                "passes_validation_robust_screen": True,
                "liability_offset_cvar05_avg": -0.10,
                "liability_offset_mean_avg": -0.02,
                "n_prototypes": 10,
            },
            {
                "model": "faithful_k25",
                "prototype_source": "vanilla",
                "weighted_similarity": False,
                "learn_distance_feature_weights": False,
                "has_all_seed_runs": True,
                "passes_validation_robust_screen": True,
                "liability_offset_cvar05_avg": -0.08,
                "liability_offset_mean_avg": -0.03,
                "n_prototypes": 25,
            },
            {
                "model": "expanded_but_better",
                "prototype_source": "spot_delta",
                "weighted_similarity": True,
                "learn_distance_feature_weights": False,
                "has_all_seed_runs": True,
                "passes_validation_robust_screen": True,
                "liability_offset_cvar05_avg": -0.01,
                "liability_offset_mean_avg": -0.01,
                "n_prototypes": 10,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            pd.DataFrame(rows).to_csv(
                task_dir / "validation_model_selection.csv", index=False
            )
            selection = MODULE._select_paper_faithful_candidate(task_dir)
        self.assertEqual(selection["selected_model"], "faithful_k25")
        self.assertEqual(selection["selection_split"], "validation")
        self.assertEqual(len(selection["eligible_candidates"]), 2)

    def test_block_bootstrap_is_paired(self):
        benchmark = np.arange(60, dtype=np.float64)
        candidate = benchmark + 0.25
        result = MODULE._bootstrap_difference(
            candidate,
            benchmark,
            n_bootstrap=200,
            block_length=10,
            confidence=0.95,
            random_state=5,
        )
        self.assertAlmostEqual(result["estimate"], 0.25)
        self.assertAlmostEqual(result["ci_low"], 0.25)
        self.assertAlmostEqual(result["ci_high"], 0.25)


if __name__ == "__main__":
    unittest.main()

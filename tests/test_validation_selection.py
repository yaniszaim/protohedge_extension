import unittest

import pandas as pd

from deephedging.real_data_analysis_torch import build_frontier_candidates
from deephedging.real_data_sweep_torch import (
    MODEL_SELECTION_VERSION,
    select_proto_configs_from_validation,
)


def _candidate_row(
    model,
    seed,
    split,
    mean,
    cvar,
    bound_occupancy=0.05,
):
    return {
        "model": model,
        "model_family": "proto",
        "seed": seed,
        "split": split,
        "risk_measure": "cvar",
        "prototype_source": "spot_delta",
        "n_prototypes": 10 if model == "proto_a" else 25,
        "weighted_similarity": False,
        "learn_distance_feature_weights": False,
        "liability_offset_mean": mean,
        "liability_offset_cvar05": cvar,
        "liability_offset_rmse": abs(mean) + 0.1,
        "liability_offset_downside_deviation": abs(cvar),
        "pct_at_any_position_bound": bound_occupancy,
        "pct_paths_touch_any_position_bound": bound_occupancy,
    }


class ValidationSelectionTests(unittest.TestCase):
    def _metrics(self):
        rows = []
        for seed in (11, 22):
            # A wins validation mean; B wins validation tail.
            rows.append(
                _candidate_row("proto_a", seed, "val", -0.10, -0.45)
            )
            rows.append(
                _candidate_row("proto_b", seed, "val", -0.20, -0.30)
            )
            # Test ordering is deliberately the opposite and must be ignored.
            rows.append(
                _candidate_row("proto_a", seed, "test", -10.0, -20.0)
            )
            rows.append(
                _candidate_row("proto_b", seed, "test", 10.0, 20.0)
            )
        return pd.DataFrame(rows)

    def test_test_outcomes_cannot_change_validation_selection(self):
        metrics = self._metrics()
        _, selected = select_proto_configs_from_validation(
            metrics,
            expected_seeds=2,
            max_bound_occupancy=0.50,
            max_path_touch_rate=0.50,
        )
        by_name = selected.set_index("selection")["model"].to_dict()
        self.assertEqual(by_name["best_screened_proto_mean"], "proto_a")
        self.assertEqual(by_name["best_screened_proto_cvar05"], "proto_b")
        self.assertTrue((selected["selection_split"] == "validation").all())
        self.assertTrue(
            (
                selected["model_selection_version"]
                == MODEL_SELECTION_VERSION
            ).all()
        )

        changed = metrics.copy()
        test_mask = changed["split"] == "test"
        changed.loc[test_mask, "liability_offset_mean"] *= -1000.0
        changed.loc[test_mask, "liability_offset_cvar05"] *= -1000.0
        _, selected_after_change = select_proto_configs_from_validation(
            changed,
            expected_seeds=2,
            max_bound_occupancy=0.50,
            max_path_touch_rate=0.50,
        )
        self.assertEqual(
            selected_after_change.set_index("selection")["model"].to_dict(),
            by_name,
        )

    def test_validation_bound_screen_is_applied_before_selection(self):
        metrics = self._metrics()
        extra = []
        for seed in (11, 22):
            row = _candidate_row(
                "proto_c",
                seed,
                "val",
                mean=100.0,
                cvar=100.0,
                bound_occupancy=0.95,
            )
            row["n_prototypes"] = 50
            extra.append(row)
        metrics = pd.concat([metrics, pd.DataFrame(extra)], ignore_index=True)

        summary, selected = select_proto_configs_from_validation(
            metrics,
            expected_seeds=2,
            max_bound_occupancy=0.50,
            max_path_touch_rate=0.50,
        )
        rejected = summary[summary["model"] == "proto_c"].iloc[0]
        self.assertFalse(rejected["passes_validation_robust_screen"])
        self.assertNotIn("proto_c", set(selected["model"]))

    def test_analysis_accepts_only_rows_marked_as_validation_selected(self):
        summary = pd.DataFrame(
            [
                {
                    "model": "proto_a",
                    "model_family": "proto",
                    "selected_for": "best_screened_proto_mean",
                    "liability_offset_mean_avg": -100.0,
                },
                {
                    "model": "proto_b",
                    "model_family": "proto",
                    "selected_for": "",
                    "liability_offset_mean_avg": 100.0,
                },
            ]
        )
        selected = build_frontier_candidates(summary)
        self.assertEqual(selected["model"].tolist(), ["proto_a"])
        self.assertEqual(
            selected["selection_split"].tolist(),
            ["validation"],
        )


if __name__ == "__main__":
    unittest.main()

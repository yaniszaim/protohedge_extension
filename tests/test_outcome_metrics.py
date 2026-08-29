import unittest

import numpy as np

from deephedging.outcome_metrics import (
    OUTCOME_DEFINITION_VERSION,
    circular_block_bootstrap_indices,
    paired_circular_block_bootstrap,
    paired_panel_circular_block_bootstrap,
    paired_panel_seed_circular_block_bootstrap,
    paired_seed_circular_block_bootstrap,
    summarize_liability_offset,
)


class OutcomeMetricTests(unittest.TestCase):
    def test_liability_offset_metrics_are_explicit_and_consistent(self):
        offset = np.array([-2.0, -1.0, 0.0, 3.0])
        metrics = summarize_liability_offset(offset)

        self.assertEqual(
            metrics["outcome_definition"],
            OUTCOME_DEFINITION_VERSION,
        )
        self.assertFalse(metrics["premium_included"])
        self.assertAlmostEqual(metrics["liability_offset_mean"], 0.0)
        self.assertAlmostEqual(
            metrics["liability_offset_rmse"],
            np.sqrt(14.0 / 4.0),
        )
        self.assertAlmostEqual(
            metrics["liability_offset_downside_deviation"],
            np.sqrt(5.0 / 4.0),
        )
        self.assertAlmostEqual(metrics["negative_offset_rate"], 0.5)
        self.assertNotIn("gains_mean", metrics)
        self.assertNotIn("shortfall_prob", metrics)

    def test_circular_blocks_are_locally_consecutive(self):
        samples = circular_block_bootstrap_indices(
            n_observations=7,
            block_length=3,
            n_bootstrap=5,
            random_state=11,
        )
        self.assertEqual(samples.shape, (5, 7))
        for sample in samples:
            self.assertTrue(np.all((sample[:2] + 1) % 7 == sample[1:3]))
            self.assertTrue(np.all((sample[3:5] + 1) % 7 == sample[4:6]))

    def test_paired_bootstrap_uses_shared_temporal_resamples(self):
        benchmark = np.linspace(-2.0, 1.0, 60)
        candidate = benchmark + 0.25
        rows = paired_circular_block_bootstrap(
            candidate,
            benchmark,
            block_length=10,
            n_bootstrap=200,
            random_state=17,
        )
        by_metric = {row["metric"]: row for row in rows}
        mean_row = by_metric["liability_offset_mean_difference"]

        self.assertAlmostEqual(mean_row["estimate"], 0.25)
        self.assertAlmostEqual(mean_row["ci_low"], 0.25)
        self.assertAlmostEqual(mean_row["ci_high"], 0.25)
        self.assertEqual(mean_row["bootstrap_method"], "paired_circular_block")

    def test_panel_bootstrap_uses_equal_group_weighting(self):
        candidate = {
            "large": np.ones(100),
            "small": np.full(10, -1.0),
        }
        benchmark = {
            "large": np.zeros(100),
            "small": np.zeros(10),
        }
        rows = paired_panel_circular_block_bootstrap(
            candidate,
            benchmark,
            block_length=5,
            n_bootstrap=50,
            random_state=9,
        )
        by_metric = {row["metric"]: row for row in rows}
        mean_row = by_metric["liability_offset_mean_difference"]

        # Equal ticker/group weighting gives (1 + -1) / 2 = 0 rather than
        # the observation-weighted value 90 / 110.
        self.assertAlmostEqual(mean_row["estimate"], 0.0)
        self.assertEqual(mean_row["n_groups"], 2)
        self.assertEqual(
            mean_row["aggregation"],
            "equal_weight_across_groups",
        )

    def test_seed_aware_bootstrap_matches_average_of_seed_metrics(self):
        benchmark = np.array(
            [[-3.0, -2.0, -1.0, 0.0], [-1.0, -0.5, 0.0, 0.5]]
        )
        candidate = np.array(
            [[-2.5, -1.5, -0.5, 0.5], [-0.8, -0.3, 0.2, 0.7]]
        )
        rows = paired_seed_circular_block_bootstrap(
            candidate,
            benchmark,
            block_length=2,
            n_bootstrap=20,
            random_state=3,
        )
        mean_row = {
            row["metric"]: row for row in rows
        }["liability_offset_mean_difference"]
        self.assertAlmostEqual(mean_row["estimate"], 0.35)
        self.assertEqual(mean_row["n_seeds"], 2)
        self.assertEqual(
            mean_row["bootstrap_method"],
            "paired_circular_block_seed_average",
        )

    def test_panel_seed_bootstrap_averages_seeds_then_groups(self):
        candidate = {
            "a": np.array([[1.0, 1.0], [3.0, 3.0]]),
            "b": np.array([[-2.0, -2.0], [0.0, 0.0]]),
        }
        benchmark = {
            "a": np.zeros((2, 2)),
            "b": np.zeros((2, 2)),
        }
        rows = paired_panel_seed_circular_block_bootstrap(
            candidate,
            benchmark,
            block_length=2,
            n_bootstrap=20,
            random_state=4,
        )
        mean_row = {
            row["metric"]: row for row in rows
        }["liability_offset_mean_difference"]
        self.assertAlmostEqual(mean_row["estimate"], 0.5)
        self.assertEqual(mean_row["n_groups"], 2)
        self.assertEqual(mean_row["n_seeds"], 2)
        self.assertEqual(mean_row["aggregation"], "seed_average_then_equal_group")


if __name__ == "__main__":
    unittest.main()

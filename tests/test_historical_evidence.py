import unittest

import numpy as np

from deephedging.historical_evidence import (
    _paired_metrics,
    frozen_oce_cvar50,
    oce_cvar50_values,
)


class HistoricalEvidenceTests(unittest.TestCase):
    def test_frozen_oce_matches_pathwise_mean(self):
        values = np.array([-3.0, -1.0, 1.0, 4.0])
        pathwise = oce_cvar50_values(values, threshold_y=1.0)
        np.testing.assert_allclose(pathwise, [-5.0, -1.0, -1.0, -1.0])
        self.assertAlmostEqual(frozen_oce_cvar50(values, 1.0), -2.0)

    def test_optimized_oce_equals_lower_half_mean_without_ties(self):
        values = np.array([-4.0, -2.0, 1.0, 3.0])
        # Any threshold between -1 and 2 is an optimum; y=0 is convenient.
        self.assertAlmostEqual(frozen_oce_cvar50(values, 0.0), -3.0)

    def test_spot_comparison_signs_treat_higher_outcome_as_better(self):
        benchmark = np.array([-2.0, -1.0, 0.0, 1.0])
        candidate = benchmark + 0.5
        metrics = _paired_metrics(candidate, benchmark, tail_probability=0.5)
        self.assertAlmostEqual(
            metrics["liability_offset_mean_difference"], 0.5
        )
        self.assertAlmostEqual(
            metrics["liability_offset_cvar50_difference"], 0.5
        )


if __name__ == "__main__":
    unittest.main()

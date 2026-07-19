from __future__ import annotations

import math
import unittest

from analysis.stories_phase3.compile import calculate_primary_metrics
from src.run.experiment.stories.quantization.run import canonical_conditions, expected_record_count


class StoriesPhase3Tests(unittest.TestCase):
    def test_canonical_matrix_has_expected_size(self) -> None:
        self.assertEqual(len(canonical_conditions()), 27)
        self.assertEqual(expected_record_count(), 290)

    def test_capability_recovery(self) -> None:
        metric = calculate_primary_metrics(1.0, 2.0, 1.0, 1.7, [1.0] * 4, [1.0] * 4)
        self.assertAlmostEqual(metric["absolute_capability_recovery"], 0.3)
        self.assertEqual(metric["verdict"], "capability_recovery")

    def test_ordinary_degradation_can_preserve_gap(self) -> None:
        metric = calculate_primary_metrics(1.0, 2.0, 1.1, 2.1, [1.0] * 4, [1.05] * 4)
        self.assertAlmostEqual(metric["isolation_gap_erosion"], 0.0)
        self.assertLess(metric["absolute_capability_recovery"], 0)
        self.assertEqual(metric["verdict"], "robust")

    def test_gap_erosion_without_improvement(self) -> None:
        metric = calculate_primary_metrics(1.0, 2.0, 1.4, 2.0, [1.0] * 4, [1.0] * 4)
        self.assertEqual(metric["absolute_capability_recovery"], 0.0)
        self.assertAlmostEqual(metric["isolation_gap_erosion"], 0.4)
        self.assertEqual(metric["verdict"], "isolation_erosion_without_recovery")

    def test_negative_recovery_is_not_clipped(self) -> None:
        metric = calculate_primary_metrics(1.0, 2.0, 1.0, 2.2, [1.0] * 4, [1.0] * 4)
        self.assertAlmostEqual(metric["absolute_capability_recovery"], -0.2)

    def test_utility_guard_and_nonfinite(self) -> None:
        degraded = calculate_primary_metrics(1.0, 2.0, 1.0, 1.5, [1.0] * 4, [1.1] * 4)
        self.assertTrue(degraded["utility_guard_failed"])
        self.assertEqual(degraded["verdict"], "inconclusive_due_to_general_degradation")
        nonfinite = calculate_primary_metrics(1.0, 2.0, math.nan, 1.5, [1.0] * 4, [1.0] * 4)
        self.assertTrue(nonfinite["utility_guard_failed"])


if __name__ == "__main__":
    unittest.main()

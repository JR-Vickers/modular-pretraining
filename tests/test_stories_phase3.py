from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from analysis.stories_phase2.common import DATA_LABELS, PRIMARY_AUX_LABEL
from analysis.stories_phase3.compile import build_headline_series, calculate_primary_metrics
from analysis.stories_phase3.plot import render_headline_figure
from src.run.experiment.stories.quantization.run import (
    DEADLINE_OFF,
    canonical_conditions,
    expected_record_count,
)


ALL_GROUPS = ("core_mlp", "aux_modules", "attention", "embeddings")
DENSE_GROUPS = ("core_mlp", "attention", "embeddings")


def synthetic_headline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    precision_factors = ((None, 1.0), (8, 1.1), (6, 1.2), (4, 1.3))

    def add(kind: str, bit: int | None, profile: str, label: str, loss: float) -> None:
        selected_groups = ALL_GROUPS if kind == "gram" else DENSE_GROUPS
        rows.append({
            "identity": {
                "model_kind": kind,
                "bit_width": bit,
                "granularity": "fp32" if bit is None else "per_channel",
                "selected_groups": [] if bit is None else list(selected_groups),
                "profile_id": profile,
                "data_label": label,
            },
            "loss": loss,
        })

    for kind, profile, offset in (
        ("gram", "all_on", 0.0),
        ("baseline", "dense", 1.0),
        ("filtered", "dense", 2.0),
    ):
        for bit, factor in precision_factors:
            for label_index, label in enumerate(DATA_LABELS):
                add(kind, bit, profile, label, factor * (1.0 + offset + label_index))

    on_losses = {None: 1.0, 8: 1.2, 6: 0.8, 4: 1.4}
    off_losses = {None: 2.0, 8: 1.7, 6: 2.1, 4: 2.2}
    for row in rows:
        identity = row["identity"]
        if (identity["model_kind"] == "gram" and identity["profile_id"] == "all_on"
                and identity["data_label"] == PRIMARY_AUX_LABEL):
            row["loss"] = on_losses[identity["bit_width"]]
        if (identity["model_kind"] == "gram" and identity["bit_width"] == 8
                and identity["profile_id"] == "all_on" and identity["data_label"] == "core"):
            row["loss"] += 1.0
    for bit, loss in off_losses.items():
        add("gram", bit, DEADLINE_OFF, PRIMARY_AUX_LABEL, loss)
    return rows


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
        self.assertEqual(nonfinite["verdict"], "inconclusive_due_to_general_degradation")

    def test_headline_series_has_signed_unclipped_values_and_fp32_zeroes(self) -> None:
        headline = build_headline_series(synthetic_headline_rows())

        self.assertEqual(headline["precision_order"], ["FP32", "int8", "int6", "int4"])
        self.assertEqual(
            headline["raw_deadline_loss"]["gram_deadline_off"],
            [2.0, 1.7, 2.1, 2.2],
        )
        self.assertEqual(headline["signed_percent"]["capability_recovery"][0], 0.0)
        self.assertEqual(headline["signed_percent"]["isolation_erosion"][0], 0.0)
        self.assertAlmostEqual(headline["signed_percent"]["capability_recovery"][1], 30.0)
        self.assertAlmostEqual(headline["signed_percent"]["capability_recovery"][2], -10.0)
        self.assertAlmostEqual(headline["signed_percent"]["isolation_erosion"][1], 50.0)
        self.assertAlmostEqual(headline["signed_percent"]["isolation_erosion"][2], -30.0)

        retained = headline["mean_retained_topic_relative_degradation_percent"]
        gram_fp32_retained_mean = sum(
            1.0 + index for index, label in enumerate(DATA_LABELS)
            if label != PRIMARY_AUX_LABEL
        ) / 4
        self.assertAlmostEqual(
            retained["gram_all_on"][1],
            10.0 + 25.0 / gram_fp32_retained_mean,
        )
        for values in (retained["dense_baseline"], retained["deadline_filtered"]):
            self.assertAlmostEqual(values[0], 0.0)
            self.assertAlmostEqual(values[1], 10.0)
            self.assertAlmostEqual(values[2], 20.0)
            self.assertAlmostEqual(values[3], 30.0)
        self.assertAlmostEqual(retained["gram_all_on"][0], 0.0)
        self.assertAlmostEqual(retained["gram_all_on"][2], 20.0)
        self.assertAlmostEqual(retained["gram_all_on"][3], 30.0)

    def test_headline_figure_renders_nonempty_png_headlessly(self) -> None:
        report = {
            "manifest": {"smoke": True},
            "headline_series": build_headline_series(synthetic_headline_rows()),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "headline.png"
            render_headline_figure(report, output)
            data = output.read_bytes()

        self.assertGreater(len(data), 1_000)
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from analysis.stories.smoke_check import check_smoke
from src.model.moe import MoETransformer
from src.run.experiment.stories.smoke.run import (
    DEFAULT_AUX_TOKENS,
    DEFAULT_CORE_TOKENS,
    routed_model_config,
    split_token_budget,
)


class SmokeConfigTests(unittest.TestCase):
    def test_default_budget_split(self):
        self.assertEqual(
            split_token_budget(10_000_000),
            (DEFAULT_CORE_TOKENS, DEFAULT_AUX_TOKENS),
        )

    def test_small_shape_exact_parameter_count_and_aux_width(self):
        model = MoETransformer(
            routed_model_config("small"), ["core", "a", "b", "c", "d"]
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5_918_464)
        self.assertEqual(model.blocks[0].moe.experts[1].c_fc.out_features, 128)


class SmokeCheckerTests(unittest.TestCase):
    def _fixture(self, root: Path, own_delta: float = 1.0):
        routed = root / "routed"
        routed.mkdir()
        losses = {
            "train": {"core": [(i, 5.0 - i * 0.1) for i in range(20)]},
            "val": {"core": [(0, 4.0)]},
        }
        (routed / "losses.pkl").write_bytes(pickle.dumps(losses))
        labels = ["core", "a", "b", "c", "d"]
        active_losses = {label: 2.0 for label in labels}
        entries = []
        profiles = [labels] + [sorted(set(labels) - {aux}) for aux in labels[1:]]
        for profile in profiles:
            ablated = next((aux for aux in labels[1:] if aux not in profile), None)
            for label in labels:
                delta = own_delta if ablated == label else (0.1 if ablated else 0.0)
                entries.append({
                    "function": "do_eval",
                    "data_label": label,
                    "expert_labels": profile,
                    "loss": active_losses[label] + delta,
                })
        (root / "stats.jsonl").write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries)
        )

    def test_checker_accepts_all_four_selective_ablations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            summary = check_smoke(root)
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["all_four_auxiliaries_passed"])

    def test_checker_rejects_nonselective_ablations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root, own_delta=0.05)
            summary = check_smoke(root)
            self.assertFalse(summary["passed"])


if __name__ == "__main__":
    unittest.main()

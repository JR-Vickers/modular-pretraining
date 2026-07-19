from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from analysis.stories_phase2.common import DATA_LABELS, evaluation_profiles, read_jsonl
from analysis.stories_phase2.compile import calculate_gate
from analysis.stories_phase2.evaluate import generate_records, validate_evaluation_records
from analysis.stories_phase2.verify import EXPECTED_MODEL_PARAMS, discover_run, validate_run


class StoriesPhase2Tests(unittest.TestCase):
    def _make_baseline_fixture(self, root: Path, nonfinite: bool = False) -> Path:
        run = root / "seed_1/20260101"
        stage = run / "baseline"
        stage.mkdir(parents=True)
        config = {
            "stages": [{"name": "baseline", "retain_targets": None}],
            "model": {
                "arch": "base", "ctx_len": 256, "vocab_size": 4096,
                "num_layers": 8, "num_heads": 8, "num_key_value": 2,
                "embed_dim": 512, "mlp_dim": 2048,
            },
            "run": {
                "seed": 1, "device": "mps", "dtype": "torch.float32", "compile": False,
                "micro_batch_size": 16, "accumulation_steps": 8,
                "effective_batch_size": 128, "nominal_token_budget": 547_853_673,
                "model_shape": "paper",
            },
        }
        (run / "config.json").write_text(json.dumps(config))
        (stage / "stage.json").write_text(json.dumps({"completed": True}))
        torch.save({"model": {"weight": torch.ones(1)}, "step": 2, "total_steps": 2}, stage / "checkpoint.pth")
        train = {label: np.asarray([[1.0, float("nan") if nonfinite else 1.0]]) for label in DATA_LABELS}
        with (stage / "losses.pkl").open("wb") as handle:
            pickle.dump({"train": train, "val": {}}, handle)
        rows = [
            {"function": "do_eval", "data_label": label, "expert_labels": None, "loss": 1.0}
            for label in DATA_LABELS
        ]
        (run / "stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return run

    def test_evaluation_profiles_are_five_unique_masks(self) -> None:
        profiles = evaluation_profiles()
        self.assertEqual(len(profiles), 5)
        self.assertEqual(len({item["profile_id"] for item in profiles}), 5)
        self.assertEqual(len({tuple(item["expert_labels"]) for item in profiles}), 5)

    def test_generate_records_produces_25_unique_records(self) -> None:
        references = {}
        for profile in evaluation_profiles()[:2]:
            for index, label in enumerate(DATA_LABELS):
                references[(profile["profile_id"], label)] = 1.0 + index / 10

        def evaluate(experts: list[str], label: str) -> float:
            profile = next(item for item in evaluation_profiles() if item["expert_labels"] == experts)
            return references.get((profile["profile_id"], label), 2.0)

        rows, reproduction = generate_records(evaluate, references)
        self.assertTrue(reproduction["passed"])
        validate_evaluation_records(rows)

    def test_reference_failure_stops_before_new_masks(self) -> None:
        references = {
            (profile["profile_id"], label): 1.0
            for profile in evaluation_profiles()[:2]
            for label in DATA_LABELS
        }
        rows, reproduction = generate_records(lambda _experts, _label: 1.1, references)
        self.assertFalse(reproduction["passed"])
        self.assertEqual(len(rows), 10)

    def test_read_jsonl_reports_invalid_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.jsonl"
            path.write_text('{"ok": true}\nnot-json\n')
            with self.assertRaisesRegex(ValueError, ":2"):
                read_jsonl(path)

    def test_discovery_selects_newest_completed_matching_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            for run_id, completed in (("20260101", True), ("20260102", False), ("20260103", True)):
                run = root / "seed_1" / run_id
                (run / "routed").mkdir(parents=True)
                (run / "config.json").write_text(json.dumps({"stages": [{"name": "routed"}]}))
                (run / "routed/stage.json").write_text(json.dumps({"completed": completed}))
            self.assertEqual(discover_run(root, "gram").name, "20260103")

    def test_gate_calculations(self) -> None:
        all_on = {label: 1.0 for label in DATA_LABELS}
        deadline_off = {label: 1.01 for label in DATA_LABELS}
        deadline_off["a-deadline-or-time-limit"] = 1.4
        filtered = {label: 1.0 for label in DATA_LABELS}
        filtered["a-deadline-or-time-limit"] = 1.5
        gate = calculate_gate(all_on, deadline_off, filtered)
        self.assertTrue(gate["passed"])
        self.assertTrue(all(gate["conditions"].values()))

    def test_nonfinite_reference_fails_record_validation(self) -> None:
        rows = []
        for profile in evaluation_profiles():
            for label in DATA_LABELS:
                rows.append({"profile_id": profile["profile_id"], "data_label": label, "loss": 1.0})
        rows[-1]["loss"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_evaluation_records(rows)

    def test_artifact_validation_accepts_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._make_baseline_fixture(Path(directory))
            with mock.patch.dict(EXPECTED_MODEL_PARAMS, {"baseline": 1}):
                report = validate_run(run, "baseline")
            self.assertTrue(report["passed"], report["failures"])

    def test_artifact_validation_rejects_nonfinite_training_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._make_baseline_fixture(Path(directory), nonfinite=True)
            with mock.patch.dict(EXPECTED_MODEL_PARAMS, {"baseline": 1}):
                report = validate_run(run, "baseline")
            self.assertFalse(report["passed"])
            self.assertTrue(any("non-finite" in failure for failure in report["failures"]))


if __name__ == "__main__":
    unittest.main()

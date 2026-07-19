"""Evaluate all five GRAM capability profiles from a completed checkpoint.

The source checkpoint is read-only. A smoke run uses 128 sequences per label and
writes below ``smoke/``; the default evaluates each complete test loader.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from analysis.stories_phase2.common import (
    DATA_LABELS,
    PRIMARY_AUX_LABEL,
    evaluation_profiles,
    profile_id,
    read_jsonl,
    retained_key,
    write_jsonl,
)
from analysis.stories_phase2.verify import DEFAULT_RESULTS_ROOT, discover_run, validate_run
from src.model.config import RoutedModelConfig
from src.model.moe import MoETransformer
from src.run.eval import eval_loss
from src.run.experiment.config import GetStoriesConfig
from src.run.util.config import setup


REPRODUCTION_TOLERANCE = 1e-5


def original_reference_losses(gram_dir: Path) -> dict[tuple[str, str], float]:
    all_on = tuple(sorted(DATA_LABELS))
    deadline_off = tuple(sorted(set(DATA_LABELS) - {PRIMARY_AUX_LABEL}))
    wanted = {all_on: profile_id(None), deadline_off: profile_id(PRIMARY_AUX_LABEL)}
    references: dict[tuple[str, str], float] = {}
    for row in read_jsonl(gram_dir / "stats.jsonl"):
        retained = retained_key(row.get("expert_labels"))
        if row.get("function") == "do_eval" and retained in wanted:
            references[(wanted[retained], row["data_label"])] = float(row["loss"])
    expected = 2 * len(DATA_LABELS)
    if len(references) != expected:
        raise ValueError(f"Expected {expected} original reference losses, found {len(references)}")
    return references


def validate_evaluation_records(rows: list[dict[str, Any]]) -> None:
    expected_profiles = {item["profile_id"] for item in evaluation_profiles()}
    keys = {(row.get("profile_id"), row.get("data_label")) for row in rows}
    expected_keys = {(profile, label) for profile in expected_profiles for label in DATA_LABELS}
    if len(rows) != 25 or keys != expected_keys:
        raise ValueError("Evaluation must contain exactly 25 unique profile/topic records")
    if any(not torch.isfinite(torch.tensor(row.get("loss", float("nan")))) for row in rows):
        raise ValueError("Evaluation records contain a non-finite loss")


def generate_records(
    evaluate_one: Callable[[list[str], str], float],
    references: dict[tuple[str, str], float],
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate reference profiles first, then accept new masks on reproduction."""
    profiles = evaluation_profiles()
    reference_ids = {profile_id(None), profile_id(PRIMARY_AUX_LABEL)}
    ordered = [item for item in profiles if item["profile_id"] in reference_ids]
    ordered += [item for item in profiles if item["profile_id"] not in reference_ids]
    rows: list[dict[str, Any]] = []
    reproduction: dict[str, Any] = {"tolerance": tolerance, "comparisons": [], "passed": True}
    for profile_index, profile in enumerate(ordered):
        if profile_index == 2 and not reproduction["passed"]:
            break
        for label in DATA_LABELS:
            loss = float(evaluate_one(profile["expert_labels"], label))
            row = {**profile, "data_label": label, "loss": loss}
            rows.append(row)
            key = (profile["profile_id"], label)
            if key in references:
                original = references[key]
                difference = abs(loss - original)
                passed = difference <= tolerance
                reproduction["comparisons"].append({
                    "profile_id": profile["profile_id"],
                    "data_label": label,
                    "original_loss": original,
                    "rerun_loss": loss,
                    "absolute_difference": difference,
                    "passed": passed,
                })
                reproduction["passed"] = reproduction["passed"] and passed
    if reproduction["passed"]:
        validate_evaluation_records(rows)
    return rows, reproduction


def _build_runtime(gram_dir: Path, output_dir: Path, device: str, max_sequences: int | None):
    source_config = json.loads((gram_dir / "config.json").read_text())
    stage_model = source_config["stages"][0]["model"]
    config = GetStoriesConfig()
    config.stages = []
    config.data.core.method = "total"
    config.data.core.limit = 1.0
    config.data.aux.method = "total"
    config.data.aux.limit = 1.0
    config.run.res_root = output_dir.parent
    config.run.experiment_id = output_dir.name
    config.run.seed = source_config["run"]["seed"]
    config.run.device = device
    config.run.dtype = "float32"
    config.run.compile = False
    config.run.cleanup_distributed = False
    config.run.target_effective_batch_size = -1
    config.run.micro_batch_size = source_config["run"]["micro_batch_size"]
    config.run.accumulation_steps = source_config["run"]["accumulation_steps"]
    config.run.limit_eval_sequences = max_sequences is not None
    if max_sequences is not None:
        config.run.max_num_test_sequences = max_sequences
    config = setup(config)
    model_config = RoutedModelConfig(**{
        key: value for key, value in stage_model.items()
        if key not in {"tokenizer"} and key in RoutedModelConfig.__dataclass_fields__
    })
    model_config.tokenizer = config.model.tokenizer
    checkpoint = torch.load(
        gram_dir / "routed/checkpoint.pth", map_location="cpu", weights_only=False, mmap=True
    )
    model = MoETransformer(model_config, labels=list(DATA_LABELS))
    model.load_state_dict(checkpoint["model"])
    model = model.to(config.run.device, dtype=torch.float32)
    return config, model, int(checkpoint["step"])


def run_evaluation(
    gram_dir: Path,
    output_dir: Path,
    device: str = "mps",
    max_sequences: int | None = None,
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> dict[str, Any]:
    verification = validate_run(gram_dir, "gram")
    if not verification["passed"]:
        raise ValueError("GRAM run verification failed: " + "; ".join(verification["failures"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    references = original_reference_losses(gram_dir) if max_sequences is None else {}
    config, model, checkpoint_step = _build_runtime(gram_dir, output_dir, device, max_sequences)

    def evaluate_one(expert_labels: list[str], data_label: str) -> float:
        return eval_loss(model, config, data_label, expert_labels=expert_labels)

    rows, reproduction = generate_records(evaluate_one, references, tolerance)
    reproduction["applicable"] = max_sequences is None
    for row in rows:
        row.update({"source_run_id": gram_dir.name, "checkpoint_step": checkpoint_step})
    records_path = output_dir / "stats.jsonl"
    write_jsonl(records_path, rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": gram_dir.name,
        "source_run_dir": str(gram_dir.resolve()),
        "source_checkpoint": str((gram_dir / "routed/checkpoint.pth").resolve()),
        "source_checkpoint_step": checkpoint_step,
        "device": str(config.run.device),
        "dtype": str(config.run.dtype),
        "micro_batch_size": config.run.micro_batch_size,
        "max_sequences_per_label": max_sequences,
        "full_test": max_sequences is None,
        "profiles": evaluation_profiles(),
        "record_count": len(rows),
        "records_path": str(records_path.resolve()),
        "reproduction": reproduction,
        "accepted": reproduction["passed"] and len(rows) == 25,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["accepted"]:
        raise RuntimeError("Reference profiles did not reproduce; new masks were not accepted")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gram-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--smoke", action="store_true", help="Evaluate 128 sequences per label below smoke/")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--tolerance", type=float, default=REPRODUCTION_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gram_dir = (args.gram_dir or discover_run(args.results_root, "gram")).resolve()
    max_sequences = 128 if args.smoke else args.max_sequences
    canonical = args.results_root / "evaluations" / gram_dir.name
    output_dir = (args.output_dir or (canonical / "smoke" if args.smoke else canonical)).resolve()
    manifest = run_evaluation(gram_dir, output_dir, args.device, max_sequences, args.tolerance)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

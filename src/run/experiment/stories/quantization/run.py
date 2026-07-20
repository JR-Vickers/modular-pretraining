"""Run the fixed Phase 3 GRAM quantization matrix.

Equivalent fp32 conditions are imported from accepted Phase 2 results. Every
new result is an atomic, identity-addressed JSON file, making interruption and
resumption safe without changing the scientific matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from analysis.stories_phase2.common import DATA_LABELS, PRIMARY_AUX_LABEL, evaluation_profiles, profile_id, read_jsonl
from analysis.stories_phase2.verify import DEFAULT_RESULTS_ROOT, discover_run, resolve_artifact_dir, validate_run
from src.model.base import BaseTransformer
from src.model.config import ModelConfig, RoutedModelConfig
from src.model.moe import MoETransformer
from src.run.eval import eval_loss
from src.run.experiment.config import GetStoriesConfig
from src.run.quantization import GROUPS, quantize_model_copy
from src.run.util.config import setup


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/stories_phase3"
BITS = (8, 6, 4)
DEADLINE_OFF = profile_id(PRIMARY_AUX_LABEL)


@dataclass(frozen=True)
class Condition:
    model_kind: str
    bit_width: int | None
    granularity: str
    selected_groups: tuple[str, ...]
    profile_ids: tuple[str, ...]
    evidence: str


def canonical_conditions() -> list[Condition]:
    all_profiles = tuple(item["profile_id"] for item in evaluation_profiles())
    primary_profiles = ("all_on", DEADLINE_OFF)
    conditions = [Condition("gram", None, "fp32", (), all_profiles, "primary")]
    conditions.extend(Condition("gram", bit, "per_channel", GROUPS, all_profiles, "primary") for bit in BITS)
    conditions.extend(
        Condition("gram", bit, "per_channel", (group,), primary_profiles, "group_diagnostic")
        for bit in BITS for group in GROUPS
    )
    for kind in ("baseline", "filtered"):
        conditions.append(Condition(kind, None, "fp32", (), ("dense",), "control"))
        dense_groups = tuple(group for group in GROUPS if group != "aux_modules")
        conditions.extend(Condition(kind, bit, "per_channel", dense_groups, ("dense",), "control") for bit in BITS)
    conditions.extend(Condition("gram", bit, "per_tensor", GROUPS, primary_profiles, "sensitivity") for bit in BITS)
    return conditions


def expected_record_count() -> int:
    return sum(len(condition.profile_ids) * len(DATA_LABELS) for condition in canonical_conditions())


def condition_identity(condition: Condition, profile: dict[str, Any], data_label: str,
                       source_run_id: str, checkpoint_step: int, checkpoint_sha256: str,
                       max_sequences: int | None) -> dict[str, Any]:
    return {
        "model_kind": condition.model_kind,
        "source_run_id": source_run_id,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "bit_width": condition.bit_width,
        "granularity": condition.granularity,
        "selected_groups": list(condition.selected_groups),
        "profile_id": profile["profile_id"],
        "expert_mask": profile["expert_mask"],
        "data_label": data_label,
        "max_sequences": max_sequences,
    }


def identity_key(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _profile(profile_value: str) -> dict[str, Any]:
    if profile_value == "dense":
        return {"profile_id": "dense", "expert_labels": [], "expert_mask": []}
    source = next(item for item in evaluation_profiles() if item["profile_id"] == profile_value)
    labels = source["expert_labels"]
    return {**source, "expert_mask": [1 if label in labels else 0 for label in DATA_LABELS]}


def _runtime_config(source_config: dict[str, Any], output_dir: Path, device: str,
                    max_sequences: int | None):
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
    return setup(config)


def _load_model(kind: str, run_dir: Path, output_dir: Path, device: str,
                max_sequences: int | None):
    verification = validate_run(run_dir, kind)
    if not verification["passed"]:
        raise ValueError(f"{kind} verification failed: {'; '.join(verification['failures'])}")
    source_config = json.loads((run_dir / "config.json").read_text())
    config = _runtime_config(source_config, output_dir, device, max_sequences)
    artifact = resolve_artifact_dir(run_dir, kind, source_config)
    checkpoint_path = artifact / "checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if kind == "gram":
        model_values = source_config["stages"][0]["model"]
        model_config = RoutedModelConfig(**{
            key: value for key, value in model_values.items()
            if key != "tokenizer" and key in RoutedModelConfig.__dataclass_fields__
        })
        model_config.tokenizer = config.model.tokenizer
        model = MoETransformer(model_config, labels=list(DATA_LABELS))
    else:
        model_values = source_config["model"]
        model_config = ModelConfig(**{
            key: value for key, value in model_values.items()
            if key != "tokenizer" and key in ModelConfig.__dataclass_fields__
        })
        model_config.tokenizer = config.model.tokenizer
        model = BaseTransformer(model_config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(config.run.device, dtype=torch.float32).eval()
    return config, model, int(checkpoint["step"]), checkpoint_path, _sha256(checkpoint_path)


def _close_config_loaders(config: Any) -> None:
    """Release memmap-backed shard files before constructing another model config."""
    loaders = getattr(getattr(config, "run", None), "loaders", {})
    closed: set[int] = set()
    for split_loaders in loaders.values():
        for loader in split_loaders.values():
            dataset = getattr(loader, "dataset", None)
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            tokens = getattr(dataset, "tokens", None)
            mmap_handle = getattr(tokens, "_mmap", None)
            if mmap_handle is not None and id(mmap_handle) not in closed:
                mmap_handle.close()
                closed.add(id(mmap_handle))
    if hasattr(config, "run"):
        config.run.loaders = {}


def _phase2_fp32_rows(kind: str, run_dir: Path, results_root: Path) -> list[dict[str, Any]]:
    if kind == "gram":
        evaluation_dir = results_root / "evaluations" / run_dir.name
        manifest = json.loads((evaluation_dir / "manifest.json").read_text())
        if not manifest.get("accepted") or not manifest.get("full_test") or not manifest["reproduction"]["passed"]:
            raise ValueError("Full runs require accepted Phase 2 fp32 reproduction within 1e-5")
        return read_jsonl(evaluation_dir / "stats.jsonl")
    return [row for row in read_jsonl(run_dir / "stats.jsonl") if row.get("function") == "do_eval"]


def _write_record(records_dir: Path, identity: dict[str, Any], record: dict[str, Any]) -> bool:
    path = records_dir / f"{identity_key(identity)}.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("identity") != identity:
            raise ValueError(f"Identity collision at {path}")
        complete = (
            isinstance(existing.get("loss"), (int, float))
            and math.isfinite(existing["loss"])
            and existing.get("git_commit")
            and existing.get("source_checkpoint")
            and "quantization_statistics" in existing
        )
        if complete:
            return False
    _atomic_json(path, record)
    return True


def run_matrix(results_root: Path = DEFAULT_RESULTS_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT,
               device: str = "mps", max_sequences: int | None = None,
               condition_indices: Iterable[int] | None = None) -> dict[str, Any]:
    runs = {kind: discover_run(results_root, kind) for kind in ("gram", "baseline", "filtered")}
    result_name = "full" if max_sequences is None else ("smoke" if max_sequences == 128 else f"max_sequences_{max_sequences}")
    output_dir = output_root / runs["gram"].name / result_name
    records_dir = output_dir / "conditions"
    commit = _git_commit()
    all_conditions = canonical_conditions()
    selected_indices = set(range(len(all_conditions)) if condition_indices is None else condition_indices)
    if not selected_indices <= set(range(len(all_conditions))):
        raise ValueError("Condition index is outside the canonical matrix")
    created = skipped = 0
    runtimes: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    try:
        for index, condition in enumerate(all_conditions):
            if index not in selected_indices:
                continue
            if condition.model_kind not in runtimes:
                for kind, runtime in list(runtimes.items()):
                    current_hash = _sha256(runtime[3])
                    if current_hash != runtime[4]:
                        raise RuntimeError(f"Source checkpoint changed during evaluation: {runtime[3]}")
                    checkpoint_hashes[kind] = current_hash
                    _close_config_loaders(runtime[0])
                    del runtimes[kind]
                runtimes[condition.model_kind] = _load_model(
                    condition.model_kind, runs[condition.model_kind], output_dir,
                    device, max_sequences,
                )
            config, source_model, step, checkpoint_path, checkpoint_hash = runtimes[condition.model_kind]
            if condition.bit_width is None:
                if max_sequences is not None:
                    model = source_model
                    quant_stats = None
                    imported_rows = None
                else:
                    model = source_model
                    quant_stats = None
                    imported_rows = _phase2_fp32_rows(condition.model_kind, runs[condition.model_kind], results_root)
            else:
                model, quant_stats = quantize_model_copy(
                    source_model, condition.bit_width, condition.granularity, condition.selected_groups
                )
                model = model.to(config.run.device).eval()
                imported_rows = None
            for profile_id_value in condition.profile_ids:
                profile = _profile(profile_id_value)
                for data_label in DATA_LABELS:
                    identity = condition_identity(condition, profile, data_label, runs[condition.model_kind].name,
                                                  step, checkpoint_hash, max_sequences)
                    path = records_dir / f"{identity_key(identity)}.json"
                    if path.is_file():
                        existing = json.loads(path.read_text())
                        if existing.get("identity") != identity:
                            raise ValueError(f"Identity collision at {path}")
                        complete = (
                            isinstance(existing.get("loss"), (int, float))
                            and math.isfinite(existing["loss"])
                            and existing.get("git_commit")
                            and existing.get("source_checkpoint")
                            and "quantization_statistics" in existing
                        )
                        if complete:
                            skipped += 1
                            continue
                    if imported_rows is None:
                        experts = profile["expert_labels"] if condition.model_kind == "gram" else None
                        loss = float(eval_loss(model, config, data_label, expert_labels=experts))
                        provenance = "evaluated"
                    else:
                        if condition.model_kind == "gram":
                            source_row = next(row for row in imported_rows if row["profile_id"] == profile_id_value
                                              and row["data_label"] == data_label)
                        else:
                            source_row = next(row for row in imported_rows if row["data_label"] == data_label)
                        loss = float(source_row["loss"])
                        provenance = "phase2_reuse"
                    record = {
                        "identity": identity,
                        "condition_index": index,
                        "evidence": condition.evidence,
                        "loss": loss,
                        "expert_labels": profile["expert_labels"],
                        "source_checkpoint": str(checkpoint_path.resolve()),
                        "git_commit": commit,
                        "device": str(config.run.device),
                        "dtype": "torch.float32",
                        "quantization_statistics": quant_stats,
                        "provenance": provenance,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    created += int(_write_record(records_dir, identity, record))
            if condition.bit_width is not None:
                del model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    finally:
        for kind, runtime in runtimes.items():
            _close_config_loaders(runtime[0])
            checkpoint_path = runtime[3]
            original_hash = runtime[4]
            current_hash = _sha256(checkpoint_path)
            if current_hash != original_hash:
                raise RuntimeError(f"Source checkpoint changed during evaluation: {checkpoint_path}")
            checkpoint_hashes[kind] = current_hash
        runtimes.clear()
    record_count = len(list(records_dir.glob("*.json")))
    expected = expected_record_count() if condition_indices is None else None
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gram_run_id": runs["gram"].name,
        "source_runs": {kind: str(path.resolve()) for kind, path in runs.items()},
        "source_checkpoint_sha256": checkpoint_hashes,
        "git_commit": commit,
        "device": device,
        "dtype": "torch.float32",
        "max_sequences_per_label": max_sequences,
        "smoke": max_sequences == 128,
        "canonical_condition_count": len(all_conditions),
        "canonical_record_count": expected_record_count(),
        "selected_condition_indices": sorted(selected_indices),
        "expected_selected_record_count": expected,
        "record_count": record_count,
        "created_records": created,
        "skipped_records": skipped,
        "complete": expected is not None and record_count == expected,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke", action="store_true", help="Evaluate 128 sequences per label")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--condition-index", type=int, action="append", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke and args.max_sequences is not None:
        raise ValueError("--smoke and --max-sequences are mutually exclusive")
    max_sequences = 128 if args.smoke else args.max_sequences
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("--max-sequences must be positive")
    manifest = run_matrix(args.results_root.resolve(), args.output_root.resolve(), args.device,
                          max_sequences, args.condition_index)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Validate and compile the canonical Phase 3 matrix to JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analysis.stories_phase2.common import AUX_LABELS, DATA_LABELS, PRIMARY_AUX_LABEL
from src.run.experiment.stories.quantization.run import (
    DEADLINE_OFF,
    DEFAULT_OUTPUT_ROOT,
    canonical_conditions,
    expected_record_count,
)


RETAINED_LABELS = tuple(label for label in DATA_LABELS if label != PRIMARY_AUX_LABEL)


def calculate_primary_metrics(
    on_fp32: float,
    off_fp32: float,
    on_quantized: float,
    off_quantized: float,
    retained_fp32: list[float],
    retained_quantized: list[float],
) -> dict[str, Any]:
    required = [on_fp32, off_fp32, on_quantized, off_quantized, *retained_fp32, *retained_quantized]
    finite = all(math.isfinite(value) for value in required)
    gap_fp32 = off_fp32 - on_fp32
    gap_quantized = off_quantized - on_quantized
    if finite and gap_fp32 != 0:
        recovery = (off_fp32 - off_quantized) / gap_fp32
        erosion = 1 - gap_quantized / gap_fp32
    else:
        recovery = float("nan")
        erosion = float("nan")
    retained_fp32_mean = statistics.mean(retained_fp32) if retained_fp32 else float("nan")
    retained_quantized_mean = statistics.mean(retained_quantized) if retained_quantized else float("nan")
    retained_relative_change = (
        retained_quantized_mean / retained_fp32_mean - 1
        if retained_fp32_mean != 0 else float("nan")
    )
    utility_failure = not finite or not math.isfinite(retained_relative_change) or retained_relative_change >= 0.10
    if utility_failure:
        verdict = "inconclusive_due_to_general_degradation"
    elif recovery >= 0.20:
        verdict = "capability_recovery"
    elif erosion >= 0.20:
        verdict = "isolation_erosion_without_recovery"
    else:
        verdict = "robust"
    return {
        "all_on_fp32_loss": on_fp32,
        "off_fp32_loss": off_fp32,
        "all_on_quantized_loss": on_quantized,
        "off_quantized_loss": off_quantized,
        "fp32_isolation_gap": gap_fp32,
        "quantized_isolation_gap": gap_quantized,
        "absolute_capability_recovery": recovery,
        "isolation_gap_erosion": erosion,
        "mean_retained_fp32_loss": retained_fp32_mean,
        "mean_retained_quantized_loss": retained_quantized_mean,
        "mean_retained_relative_change": retained_relative_change,
        "utility_guard_failed": utility_failure,
        "verdict": verdict,
    }


def load_and_validate(result_dir: Path, require_complete: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((result_dir / "manifest.json").read_text())
    rows = [json.loads(path.read_text()) for path in sorted((result_dir / "conditions").glob("*.json"))]
    identities = [json.dumps(row.get("identity"), sort_keys=True) for row in rows]
    failures: list[str] = []
    if len(identities) != len(set(identities)):
        failures.append("duplicate condition identities")
    if require_complete and len(rows) != expected_record_count():
        failures.append(f"expected {expected_record_count()} records, found {len(rows)}")
    if require_complete and not manifest.get("complete"):
        failures.append("manifest does not mark the canonical matrix complete")
    expected_by_index = {
        index: len(condition.profile_ids) * len(DATA_LABELS)
        for index, condition in enumerate(canonical_conditions())
    }
    actual_by_index = {
        index: sum(row.get("condition_index") == index for row in rows)
        for index in expected_by_index
    }
    if require_complete and actual_by_index != expected_by_index:
        failures.append("records do not match the canonical per-condition matrix")
    for row in rows:
        identity = row.get("identity", {})
        required = ("model_kind", "source_run_id", "checkpoint_step", "checkpoint_sha256",
                    "bit_width", "granularity", "selected_groups", "profile_id", "expert_mask",
                    "data_label", "max_sequences")
        if any(key not in identity for key in required):
            failures.append("record has incomplete identity")
            continue
        if identity["data_label"] not in DATA_LABELS:
            failures.append(f"unknown data label {identity['data_label']}")
        expected_mask_length = len(DATA_LABELS) if identity["model_kind"] == "gram" else 0
        if len(identity["expert_mask"]) != expected_mask_length or any(value not in (0, 1) for value in identity["expert_mask"]):
            failures.append("invalid expert mask")
        if not isinstance(row.get("loss"), (int, float)) or not math.isfinite(row["loss"]):
            failures.append("non-finite loss")
        if len(identity["checkpoint_sha256"]) != 64 or not row.get("git_commit"):
            failures.append("invalid provenance")
    if failures:
        raise ValueError("Phase 3 validation failed: " + "; ".join(sorted(set(failures))))
    return manifest, rows


def _find(rows: list[dict[str, Any]], model_kind: str, bit_width: int | None,
          granularity: str, selected_groups: tuple[str, ...], profile: str,
          label: str) -> dict[str, Any]:
    matches = [row for row in rows if row["identity"]["model_kind"] == model_kind
               and row["identity"]["bit_width"] == bit_width
               and row["identity"]["granularity"] == granularity
               and tuple(row["identity"]["selected_groups"]) == selected_groups
               and row["identity"]["profile_id"] == profile
               and row["identity"]["data_label"] == label]
    if len(matches) != 1:
        raise ValueError(f"Expected one matching record, found {len(matches)}")
    return matches[0]


def compile_results(result_dir: Path) -> dict[str, Any]:
    manifest, rows = load_and_validate(result_dir)
    all_groups = ("core_mlp", "aux_modules", "attention", "embeddings")
    dense_groups = ("core_mlp", "attention", "embeddings")
    on32 = _find(rows, "gram", None, "fp32", (), "all_on", PRIMARY_AUX_LABEL)["loss"]
    off32 = _find(rows, "gram", None, "fp32", (), DEADLINE_OFF, PRIMARY_AUX_LABEL)["loss"]
    on4 = _find(rows, "gram", 4, "per_channel", all_groups, "all_on", PRIMARY_AUX_LABEL)["loss"]
    off4 = _find(rows, "gram", 4, "per_channel", all_groups, DEADLINE_OFF, PRIMARY_AUX_LABEL)["loss"]
    retained32 = [_find(rows, "gram", None, "fp32", (), "all_on", label)["loss"] for label in RETAINED_LABELS]
    retained4 = [_find(rows, "gram", 4, "per_channel", all_groups, "all_on", label)["loss"] for label in RETAINED_LABELS]
    primary = calculate_primary_metrics(on32, off32, on4, off4, retained32, retained4)

    normalized: dict[str, dict[str, Any]] = {}
    for label in (PRIMARY_AUX_LABEL, "alien-encounters"):
        off_profile = f"leave_out__{label}"
        label_on32 = _find(rows, "gram", None, "fp32", (), "all_on", label)["loss"]
        label_off32 = _find(rows, "gram", None, "fp32", (), off_profile, label)["loss"]
        normalized[label] = {}
        for bit in (8, 6, 4):
            label_on = _find(rows, "gram", bit, "per_channel", all_groups, "all_on", label)["loss"]
            label_off = _find(rows, "gram", bit, "per_channel", all_groups, off_profile, label)["loss"]
            normalized[label][str(bit)] = calculate_primary_metrics(
                label_on32, label_off32, label_on, label_off, retained32, retained32
            )

    degradations: list[dict[str, Any]] = []
    for kind, groups, profile in (("gram", all_groups, "all_on"), ("baseline", dense_groups, "dense"),
                                  ("filtered", dense_groups, "dense")):
        for bit in (8, 6, 4):
            for label in DATA_LABELS:
                fp = _find(rows, kind, None, "fp32", (), profile, label)["loss"]
                quantized = _find(rows, kind, bit, "per_channel", groups, profile, label)["loss"]
                degradations.append({"model_kind": kind, "bit_width": bit, "data_label": label,
                                     "fp32_loss": fp, "quantized_loss": quantized,
                                     "signed_change": quantized - fp,
                                     "relative_change": quantized / fp - 1})
    filtered_distances = []
    for bit in (8, 6, 4):
        gram_off = _find(rows, "gram", bit, "per_channel", all_groups, DEADLINE_OFF, PRIMARY_AUX_LABEL)["loss"]
        filtered = _find(rows, "filtered", bit, "per_channel", dense_groups, "dense", PRIMARY_AUX_LABEL)["loss"]
        filtered_distances.append({"bit_width": bit, "gram_deadline_off_loss": gram_off,
                                   "filtered_loss": filtered, "signed_distance": gram_off - filtered,
                                   "absolute_distance": abs(gram_off - filtered)})

    diagnostic_losses = [row for row in rows if row["evidence"] == "group_diagnostic"]
    sensitivity = [row for row in rows if row["evidence"] == "sensitivity"]
    quantization_error = []
    seen_error_conditions: set[tuple[Any, ...]] = set()
    for row in rows:
        stats = row["quantization_statistics"]
        if stats is None:
            continue
        identity = row["identity"]
        key = (identity["model_kind"], identity["bit_width"], identity["granularity"],
               tuple(identity["selected_groups"]))
        if key in seen_error_conditions:
            continue
        seen_error_conditions.add(key)
        quantization_error.append({
            "model_kind": identity["model_kind"],
            "bit_width": identity["bit_width"],
            "granularity": identity["granularity"],
            "selected_groups": identity["selected_groups"],
            "overall": stats["overall"],
            "per_group": stats["per_group"],
        })
    secondary_raw = {
        label: [
            {"bit_width": bit, "all_on_loss": _find(rows, "gram", bit, "per_channel", all_groups, "all_on", label)["loss"],
             "off_loss": _find(rows, "gram", bit, "per_channel", all_groups, f"leave_out__{label}", label)["loss"]}
            for bit in (8, 6, 4)
        ] for label in ("bygone-eras", "cultural-traditions")
    }
    report = {
        "manifest": manifest,
        "primary_int4_verdict": primary,
        "normalized_isolation": normalized,
        "secondary_raw_isolation": secondary_raw,
        "same_bit_filtered_distance": filtered_distances,
        "all_on_degradation": degradations,
        "quantization_error": quantization_error,
        "group_diagnostic_losses": [
            {"bit_width": row["identity"]["bit_width"],
             "selected_groups": row["identity"]["selected_groups"],
             "profile_id": row["identity"]["profile_id"],
             "data_label": row["identity"]["data_label"], "loss": row["loss"]}
            for row in diagnostic_losses
        ],
        "per_tensor_sensitivity": [
            {"bit_width": row["identity"]["bit_width"],
             "profile_id": row["identity"]["profile_id"],
             "data_label": row["identity"]["data_label"], "loss": row["loss"]}
            for row in sensitivity
        ],
        "group_diagnostic_record_count": len(diagnostic_losses),
        "per_tensor_sensitivity_record_count": len(sensitivity),
    }
    (result_dir / "phase3_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (result_dir / "phase3_records.csv").open("w", newline="") as handle:
        fields = ["model_kind", "bit_width", "granularity", "selected_groups", "profile_id",
                  "expert_mask", "data_label", "loss", "source_run_id", "checkpoint_step",
                  "checkpoint_sha256", "git_commit", "provenance"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            identity = row["identity"]
            writer.writerow({**{field: identity.get(field) for field in fields},
                             "selected_groups": ";".join(identity["selected_groups"]),
                             "expert_mask": ";".join(map(str, identity["expert_mask"])),
                             "loss": row["loss"], "git_commit": row["git_commit"],
                             "provenance": row["provenance"]})
    markdown = [
        "# Phase 3 Quantization Summary", "",
        f"**Pre-registered int4 verdict: {primary['verdict'].replace('_', ' ')}.**", "",
        "| Metric | Value |", "|---|---:|",
        f"| FP32 deadline isolation gap | {primary['fp32_isolation_gap']:+.8f} |",
        f"| Int4 deadline isolation gap | {primary['quantized_isolation_gap']:+.8f} |",
        f"| Absolute capability recovery | {primary['absolute_capability_recovery']:+.2%} |",
        f"| Isolation-gap erosion | {primary['isolation_gap_erosion']:+.2%} |",
        f"| Mean retained loss change | {primary['mean_retained_relative_change']:+.2%} |",
        f"| Utility guard failed | {primary['utility_guard_failed']} |", "",
        "Ratios are signed and un-clipped. Deadline and alien normalized results are in `phase3_report.json`; bygone and cultural are reported there as raw secondary curves. Per-tensor conditions are sensitivity evidence only.", "",
        "Limitations: one seed, a 26M dense-core/32.57M GRAM model, synthetic SimpleStories data, and fake weight quantization rather than packed integer deployment.",
    ]
    (result_dir / "phase3_summary.md").write_text("\n".join(markdown) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gram-run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    if result_dir is None:
        if args.gram_run_id is None:
            candidates = sorted(path.parent for path in args.output_root.glob("*/full/manifest.json"))
            if not candidates:
                raise FileNotFoundError("No full Phase 3 result found")
            result_dir = candidates[-1]
        else:
            result_dir = args.output_root / args.gram_run_id / "full"
    report = compile_results(result_dir.resolve())
    print(json.dumps(report["primary_int4_verdict"], indent=2))


if __name__ == "__main__":
    main()

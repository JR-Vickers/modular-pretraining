"""Compile verified Phase 2 runs and checkpoint-only GRAM evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from analysis.stories_phase2.common import (
    AUX_LABELS,
    DATA_LABELS,
    PRIMARY_AUX_LABEL,
    profile_id,
    read_jsonl,
    retained_key,
)
from analysis.stories_phase2.evaluate import validate_evaluation_records
from analysis.stories_phase2.verify import DEFAULT_RESULTS_ROOT, verify_phase2


def _losses_by_label(path: Path) -> dict[str, float]:
    rows = [row for row in read_jsonl(path) if row.get("function") == "do_eval"]
    losses: dict[str, float] = {}
    for row in rows:
        if row.get("expert_labels") is None:
            losses[row["data_label"]] = float(row["loss"])
    if set(losses) != set(DATA_LABELS):
        raise ValueError(f"Expected one dense evaluation for each data label in {path}")
    return losses


def calculate_gate(
    all_on: dict[str, float],
    deadline_ablated: dict[str, float],
    filtered: dict[str, float],
) -> dict[str, Any]:
    forget_effect = deadline_ablated[PRIMARY_AUX_LABEL] - all_on[PRIMARY_AUX_LABEL]
    retained = [label for label in DATA_LABELS if label != PRIMARY_AUX_LABEL]
    retained_changes = [abs(deadline_ablated[label] - all_on[label]) for label in retained]
    median_off_topic = statistics.median(retained_changes)
    mean_retained = statistics.mean(retained_changes)
    all_on_distance = abs(all_on[PRIMARY_AUX_LABEL] - filtered[PRIMARY_AUX_LABEL])
    ablated_distance = abs(deadline_ablated[PRIMARY_AUX_LABEL] - filtered[PRIMARY_AUX_LABEL])
    conditions = {
        "primary_forget_effect": forget_effect > 0,
        "selectivity": forget_effect > median_off_topic,
        "filter_alignment": ablated_distance < all_on_distance,
        "retain_preservation": mean_retained < forget_effect,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "forget_effect": forget_effect,
        "median_absolute_off_topic_change": median_off_topic,
        "mean_absolute_retained_change": mean_retained,
        "all_on_filter_distance": all_on_distance,
        "ablated_filter_distance": ablated_distance,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compile_phase2(results_root: Path, evaluation_dir: Path | None = None) -> dict[str, Any]:
    verification = verify_phase2(results_root)
    if not verification["passed"]:
        raise ValueError("Training-run verification failed: " + "; ".join(verification["failures"]))
    runs = verification["runs"]
    gram_dir = Path(runs["gram"]["run_dir"])
    evaluation_dir = evaluation_dir or results_root / "evaluations" / gram_dir.name
    manifest = json.loads((evaluation_dir / "manifest.json").read_text())
    rows = read_jsonl(evaluation_dir / "stats.jsonl")
    validate_evaluation_records(rows)
    if not manifest.get("accepted") or not manifest.get("full_test"):
        raise ValueError("Compiler requires an accepted full-test evaluation manifest")
    eval_losses = {(row["profile_id"], row["data_label"]): float(row["loss"]) for row in rows}
    all_on = {label: eval_losses[(profile_id(None), label)] for label in DATA_LABELS}
    ablated = {
        aux: {label: eval_losses[(profile_id(aux), label)] for label in DATA_LABELS}
        for aux in AUX_LABELS
    }
    baseline = _losses_by_label(Path(runs["baseline"]["run_dir"]) / "stats.jsonl")
    filtered = _losses_by_label(Path(runs["filtered"]["run_dir"]) / "stats.jsonl")
    primary_rows: list[dict[str, Any]] = []
    sources = (
        ("baseline", baseline),
        ("gram_all_on", all_on),
        ("gram_deadline_ablated", ablated[PRIMARY_AUX_LABEL]),
        ("deadline_filtered", filtered),
    )
    for condition, losses in sources:
        for label in DATA_LABELS:
            primary_rows.append({"condition": condition, "data_label": label, "loss": losses[label]})
    ablation_rows: list[dict[str, Any]] = []
    for aux in AUX_LABELS:
        retained_changes = [
            abs(ablated[aux][label] - all_on[label]) for label in DATA_LABELS if label != aux
        ]
        for label in DATA_LABELS:
            ablation_rows.append({
                "ablated_module": aux,
                "data_label": label,
                "is_own_topic": label == aux,
                "all_on_loss": all_on[label],
                "ablated_loss": ablated[aux][label],
                "signed_loss_change": ablated[aux][label] - all_on[label],
                "absolute_loss_change": abs(ablated[aux][label] - all_on[label]),
                "mean_absolute_retained_change": statistics.mean(retained_changes),
            })
    gate = calculate_gate(all_on, ablated[PRIMARY_AUX_LABEL], filtered)
    gate["verification_passed"] = verification["passed"]
    gate["passed"] = gate["passed"] and verification["passed"]
    results_root.mkdir(parents=True, exist_ok=True)
    _write_csv(results_root / "phase2_primary.csv", ["condition", "data_label", "loss"], primary_rows)
    _write_csv(
        results_root / "phase2_ablations.csv",
        ["ablated_module", "data_label", "is_own_topic", "all_on_loss", "ablated_loss",
         "signed_loss_change", "absolute_loss_change", "mean_absolute_retained_change"],
        ablation_rows,
    )
    status = "PASS" if gate["passed"] else "FAIL"
    summary = [
        f"# Phase 2 Summary\n",
        f"**Gate: {status}.** {'Proceed to quantization.' if gate['passed'] else 'Stop before quantization.'}\n",
        "## Inputs\n",
        f"- GRAM: `{runs['gram']['run_dir']}`",
        f"- Baseline: `{runs['baseline']['run_dir']}`",
        f"- Deadline-filtered: `{runs['filtered']['run_dir']}`",
        f"- Checkpoint-only evaluation: `{evaluation_dir.resolve()}`\n",
        "All runs use seed 1, eager FP32 on MPS, micro-batch 16, accumulation 8, effective batch 128, and the paper model shape.\n",
        "## Gate metrics\n",
        "| Condition | Value | Result |",
        "|---|---:|:---:|",
        f"| Deadline forget effect | {gate['forget_effect']:+.8f} | {'Pass' if gate['conditions']['primary_forget_effect'] else 'Fail'} |",
        f"| Median absolute off-topic change | {gate['median_absolute_off_topic_change']:.8f} | {'Pass' if gate['conditions']['selectivity'] else 'Fail'} |",
        f"| Filter distance: all-on / ablated | {gate['all_on_filter_distance']:.8f} / {gate['ablated_filter_distance']:.8f} | {'Pass' if gate['conditions']['filter_alignment'] else 'Fail'} |",
        f"| Mean absolute retained change | {gate['mean_absolute_retained_change']:.8f} | {'Pass' if gate['conditions']['retain_preservation'] else 'Fail'} |\n",
        "## Primary losses\n",
        "| Label | Baseline | GRAM all-on | Deadline off | Filtered |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in DATA_LABELS:
        summary.append(
            f"| {label} | {baseline[label]:.8f} | {all_on[label]:.8f} | "
            f"{ablated[PRIMARY_AUX_LABEL][label]:.8f} | {filtered[label]:.8f} |"
        )
    summary.extend(["\n## All leave-one-out effects\n", "| Module | Own-topic delta | Mean absolute retained delta |", "|---|---:|---:|"])
    for aux in AUX_LABELS:
        retained = [abs(ablated[aux][label] - all_on[label]) for label in DATA_LABELS if label != aux]
        summary.append(f"| {aux} | {ablated[aux][aux] - all_on[aux]:+.8f} | {statistics.mean(retained):.8f} |")
    (results_root / "phase2_summary.md").write_text("\n".join(summary) + "\n")
    report = {"status": status, "gate": gate, "verification": verification, "evaluation_manifest": manifest}
    (results_root / "phase2_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--evaluation-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compile_phase2(args.results_root, args.evaluation_dir)
    print(json.dumps(report["gate"], indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Validate the Phase 1 loss trend and four auxiliary-ablation effects."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
from pathlib import Path


def _load_losses(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _loss_windows(losses: dict) -> tuple[float, float, int]:
    observations = []
    for values in losses.get("train", {}).values():
        observations.extend((int(step), float(loss)) for step, loss in values)
    observations.sort(key=lambda item: item[0])
    if not observations:
        raise ValueError("No training-loss observations found")
    if not all(math.isfinite(loss) for _, loss in observations):
        raise ValueError("Training losses contain a non-finite value")
    for split in losses.values():
        for values in split.values():
            if not all(math.isfinite(float(loss)) for _, loss in values):
                raise ValueError("Recorded losses contain a non-finite value")
    window = max(1, math.ceil(len(observations) * 0.1))
    first = statistics.median(loss for _, loss in observations[:window])
    final = statistics.median(loss for _, loss in observations[-window:])
    return first, final, len(observations)


def _load_eval_profiles(stats_path: Path) -> tuple[list[str], dict[frozenset[str], dict[str, float]]]:
    entries = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    eval_entries = [entry for entry in entries if entry.get("function") == "do_eval"]
    if not eval_entries:
        raise ValueError("No evaluation entries found")
    profiles: dict[frozenset[str], dict[str, float]] = {}
    all_labels = set()
    for entry in eval_entries:
        loss = float(entry["loss"])
        if not math.isfinite(loss):
            raise ValueError("Evaluation losses contain a non-finite value")
        active = frozenset(entry.get("expert_labels") or [])
        profiles.setdefault(active, {})[entry["data_label"]] = loss
        all_labels.update(active)
    labels = ["core"] + sorted(all_labels - {"core"})
    return labels, profiles


def check_smoke(run_dir: Path) -> dict:
    losses_paths = sorted(run_dir.glob("routed*/losses.pkl"))
    if len(losses_paths) != 1:
        raise ValueError(f"Expected one routed losses.pkl, found {len(losses_paths)}")
    first, final, num_observations = _loss_windows(_load_losses(losses_paths[0]))
    labels, profiles = _load_eval_profiles(run_dir / "stats.jsonl")
    active = frozenset(labels)
    if active not in profiles:
        raise ValueError("Missing all-experts-active evaluation profile")

    auxiliary_results = {}
    for auxiliary in labels[1:]:
        ablated = active - {auxiliary}
        if ablated not in profiles:
            raise ValueError(f"Missing ablation profile for {auxiliary}")
        missing_topics = (
            (set(labels) - set(profiles[active]))
            | (set(labels) - set(profiles[ablated]))
        )
        if missing_topics:
            raise ValueError(f"Missing evaluation topics: {sorted(missing_topics)}")
        deltas = {
            topic: profiles[ablated][topic] - profiles[active][topic]
            for topic in labels
        }
        retain_deltas = [deltas[topic] for topic in labels if topic != auxiliary]
        retain_median = statistics.median(retain_deltas)
        passed = deltas[auxiliary] > 0 and deltas[auxiliary] > retain_median
        auxiliary_results[auxiliary] = {
            "own_topic_signed_loss_increase": deltas[auxiliary],
            "other_topics_and_core_median_signed_delta": retain_median,
            "passed": passed,
        }

    loss_decreased = final < first
    all_auxiliaries_passed = all(item["passed"] for item in auxiliary_results.values())
    passed = loss_decreased and len(auxiliary_results) == 4 and all_auxiliaries_passed
    return {
        "passed": passed,
        "finite_losses": True,
        "training_loss": {
            "num_observations": num_observations,
            "first_10_percent_median": first,
            "final_10_percent_median": final,
            "decreased": loss_decreased,
        },
        "auxiliary_ablations": auxiliary_results,
        "all_four_auxiliaries_passed": len(auxiliary_results) == 4 and all_auxiliaries_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        summary = check_smoke(args.run_dir)
    except Exception as exc:
        summary = {"passed": False, "error": str(exc)}
    output = args.run_dir / "smoke_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

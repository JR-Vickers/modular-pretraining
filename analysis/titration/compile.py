#!/usr/bin/env python3
"""
Compile titration eval results into titration.csv with compute ratios.

Source: results/titration/<method>/800M/seed_N/<timestamp>/stats.jsonl (the
newest timestamped run per method/seed). Each row is the test
cross-entropy of GRAM (``grmoe``) or FT-LoRA (``lora``) at a float forward-mask
weight -- the *titration level* ``t`` in [0, 1] -- on one aux capability,
evaluated either on that aux's own domain or on core. ``t = 0`` ablates the aux
module, ``t = 1`` fully enables it.

CR is computed per seed against that seed's own 800M baseline learning curves at
results/scaling/realistic/base/800M/seed_N/<ts>/baseline/losses.pkl, using the
shared ``analysis.common.compile`` machinery (log-space power-law fit, pure
inverse). Output columns include the ``titration`` level.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.common.compile import build_baseline, compute_cr, latest_baseline_ts, pool_baselines


RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
TITRATION_ROOT = RESULTS_ROOT / "titration"
BASE_ROOT = RESULTS_ROOT / "scaling" / "realistic" / "base" / "800M"
DEFAULT_OUT = Path(__file__).resolve().parent / "titration.csv"

SIZE = "800M"
SEEDS = ("seed_1", "seed_2", "seed_3")
METHODS = ("grmoe", "lora")


def label_class(data_label: str) -> str:
    """Each titration row evaluates either core or the titrated aux capability."""
    return "core" if data_label == "core" else "aux"


def latest_stats(method: str, seed: str) -> Path | None:
    """Newest run's stats.jsonl for (method, seed) under the timestamped layout
    results/titration/<method>/800M/seed_N/<timestamp>/stats.jsonl. "Newest" =
    lexicographically-largest timestamp dir, so reruns supersede rather than
    accumulate."""
    seed_dir = TITRATION_ROOT / method / SIZE / seed
    if not seed_dir.is_dir():
        return None
    cands = [d for d in seed_dir.iterdir()
             if d.is_dir() and (d / "stats.jsonl").exists()]
    if not cands:
        return None
    return max(cands, key=lambda d: d.name) / "stats.jsonl"


def load_baselines() -> dict[str, dict]:
    """Build the per-seed 800M baseline used to normalize CR."""
    out: dict[str, dict] = {}
    for seed in SEEDS:
        seed_dir = BASE_ROOT / seed
        ts = latest_baseline_ts(seed_dir) if seed_dir.is_dir() else None
        base = build_baseline(ts) if ts is not None else None
        if base is None:
            print(f"  skip {seed}: no baseline under {seed_dir}")
            continue
        out[seed] = base
        print(f"  {seed}: baseline {ts.name}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    baselines = pool_baselines(load_baselines())

    # One stats.jsonl per (method, seed) -- the newest timestamped run. Each row
    # carries its seed; normalize each against that seed's baseline.
    all_rows: list[dict] = []
    for method in METHODS:
        for seed in SEEDS:
            stats_path = latest_stats(method, seed)
            if stats_path is None:
                print(f"  skip {method}/{seed}: no stats.jsonl")
                continue
            base = baselines.get(seed)
            if base is None:
                continue
            print(f"  {method}/{seed}: {stats_path.parent.name}")
            for line in open(stats_path):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                dl = rec["data_label"]
                loss = rec["loss"]
                all_rows.append({
                    "method": rec["name"],
                    "model_size": SIZE,
                    "num_params": base["num_params"],
                    "seed": seed,
                    "aux": rec["aux"],
                    "titration": rec["titration"],
                    "data_label": dl,
                    "label_class": label_class(dl),
                    "loss": loss,
                    "compute_ratio": compute_cr(loss, dl, base),
                    "source": stats_path.relative_to(RESULTS_ROOT).as_posix(),
                })

    all_rows.sort(key=lambda r: (
        r["method"], r["seed"], r["aux"], r["titration"], r["data_label"],
    ))

    fieldnames = [
        "method", "model_size", "num_params", "seed", "aux", "titration",
        "data_label", "label_class", "loss", "compute_ratio", "source",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

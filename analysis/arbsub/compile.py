#!/usr/bin/env python3
"""
Compile arbsub (eval_arbsub) results into a single CSV with compute ratios.

Source: results/arbsub/<size>/seed_N/<method>/stats.jsonl
  where <method> ∈ {grmoe, lora} and <size> is iterated over whatever sizes
  exist (currently 800M only).

CR is computed against the seed_1 baseline from
results/scaling/realistic/base/<size>/seed_1/<latest_ts>/baseline/losses.pkl
(power-law fit with linear extrapolation past the last training step). The
same baseline curve is used for all seeds so values are directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.common.compile import (
    build_baseline,
    pool_baselines,
    compute_cr,
    label_class,
    latest_baseline_ts,
    model_size_sort_key,
)


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
ARBSUB_ROOT = EXPERIMENT_ROOT / "results" / "arbsub"
SCALING_BASE_ROOT = (EXPERIMENT_ROOT / "results" / "scaling" / "realistic"
                     / "base")
DEFAULT_OUT = SCRIPT_DIR / "arbsub.csv"

METHODS = ("grmoe", "lora")
SEEDS = ("seed_1", "seed_2", "seed_3")


# ---------------------------------------------------------------------------
# Per-(size, seed) baseline registry: each seed normalized against its own
# baseline. CR math: common.compile.
# ---------------------------------------------------------------------------

def fit_baseline(size: str, seed: str) -> dict | None:
    seed_dir = SCALING_BASE_ROOT / size / seed
    if not seed_dir.is_dir():
        return None
    ts_dir = latest_baseline_ts(seed_dir)
    if ts_dir is None:
        return None
    return build_baseline(ts_dir)


# ---------------------------------------------------------------------------
# Row collection.
# ---------------------------------------------------------------------------

def collect_rows(size: str) -> list[dict]:
    rows: list[dict] = []
    size_dir = ARBSUB_ROOT / size
    # pooled curve + common denominator across this size's seeds
    bases = pool_baselines({seed: fit_baseline(size, seed) for seed in SEEDS})
    for seed in SEEDS:
        base = bases.get(seed)
        if base is None:
            print(f"  skip {size}/{seed}: no baseline")
            continue
        for method in METHODS:
            stats_path = size_dir / seed / method / "stats.jsonl"
            if not stats_path.exists():
                continue
            with open(stats_path) as f:
                for line in f:
                    rec = json.loads(line)
                    loss = rec["loss"]
                    retained = rec.get("retained", [])
                    rows.append({
                        "method": method,
                        "model_size": size,
                        "num_params": base["num_params"],
                        "seed": seed,
                        "retained": "+".join(sorted(retained)),
                        "data_label": rec["data_label"],
                        "label_class": label_class(
                            rec["data_label"], retained, bool(rec.get("elicited"))),
                        "loss": loss,
                        "compute_ratio": compute_cr(loss, rec["data_label"], base),
                        "source": stats_path.relative_to(RESULTS_ROOT).as_posix(),
                    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not ARBSUB_ROOT.is_dir():
        raise SystemExit(f"No arbsub results under {ARBSUB_ROOT}")

    sizes = sorted(
        [d.name for d in ARBSUB_ROOT.iterdir() if d.is_dir()],
        key=model_size_sort_key,
    )
    print(f"Sizes: {sizes}")

    all_rows: list[dict] = []
    for size in sizes:
        rs = collect_rows(size)
        print(f"{size}: {len(rs)} rows")
        all_rows.extend(rs)

    all_rows.sort(key=lambda r: (
        model_size_sort_key(r["model_size"]), r["seed"], r["method"],
        r["retained"], r["data_label"],
    ))

    fieldnames = [
        "method", "model_size", "num_params", "seed", "retained",
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

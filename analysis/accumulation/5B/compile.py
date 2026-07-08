#!/usr/bin/env python3
"""
Compile 5B accumulation-mode results into a CSV with compute ratios.

Each method (filtering, grmoe, lora) has two 5B/seed_1 result dirs:
  - run_1     -> heterogeneous accumulation ("het")
  - <ts dir>  -> uniform accumulation       ("uni")
Both are emitted, tagged in the new `accumulation` column.

Baseline is taken only from base/5B/seed_1/run_1, and CR is always computed
against that baseline's losses.pkl curve (power-law fit with linear
extrapolation past the last training step).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.common.compile import (
    build_baseline,
    compute_cr,
    label_class,
)


RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
RESULTS_ROOT = RESULTS_DIR / "scaling" / "realistic"
DEFAULT_OUT = Path(__file__).resolve().parent / "accumulation.csv"

METHODS = ("filtering", "grmoe", "lora")
SIZE = "5B"
SEED = "seed_1"
RETAINED_FILTER = ("core", "papers-biology")
HET_DIR = "run_1"  # run_1 -> "het"; any other ts_dir -> "uni"


def accumulation_tag(ts_dir_name: str) -> str:
    return "het" if ts_dir_name == HET_DIR else "uni"


# ---------------------------------------------------------------------------
# Baseline: base/5B/seed_1/run_1 only. CR math: common.compile.
# ---------------------------------------------------------------------------

def load_baseline() -> dict:
    ts_dir = RESULTS_ROOT / "base" / SIZE / SEED / HET_DIR
    base = build_baseline(ts_dir)
    if base is None:
        raise FileNotFoundError(f"missing baseline files under {ts_dir}")
    return base


# ---------------------------------------------------------------------------
# Method rows: emit both ts_dirs (het + uni) per method.
# ---------------------------------------------------------------------------

def collect_method_rows(method: str, base: dict) -> list[dict]:
    seed_dir = RESULTS_ROOT / method / SIZE / SEED
    rows: list[dict] = []
    if not seed_dir.is_dir():
        return rows
    for ts_dir in sorted(seed_dir.iterdir()):
        if not ts_dir.is_dir() or not (ts_dir / "stats.jsonl").exists():
            continue
        tag = accumulation_tag(ts_dir.name)
        stats_path = ts_dir / "stats.jsonl"
        with open(stats_path) as f:
            for line in f:
                rec = json.loads(line)
                if tuple(sorted(rec.get("retained", []))) != RETAINED_FILTER:
                    continue
                loss = rec["loss"]
                rows.append({
                    "method": method,
                    "model_size": SIZE,
                    "num_params": base["num_params"],
                    "seed": SEED,
                    "accumulation": tag,
                    "retained": "+".join(RETAINED_FILTER),
                    "data_label": rec["data_label"],
                    "label_class": label_class(
                        rec["data_label"], rec["retained"], rec["elicited"]),
                    "loss": loss,
                    "compute_ratio": compute_cr(loss, rec["data_label"], base),
                    "source": stats_path.relative_to(RESULTS_DIR).as_posix(),
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print("Fitting baseline from base/5B/seed_1/run_1 ...")
    base = load_baseline()

    all_rows: list[dict] = []
    for method in METHODS:
        print(f"Collecting {method} ...")
        rs = collect_method_rows(method, base)
        tags = sorted({r["accumulation"] for r in rs})
        print(f"  {len(rs)} rows, accumulation tags: {tags}")
        all_rows.extend(rs)

    all_rows.sort(key=lambda r: (
        r["method"], r["accumulation"], r["data_label"], r["label_class"],
    ))

    fieldnames = [
        "method", "model_size", "num_params", "seed", "accumulation",
        "retained", "data_label", "label_class", "loss", "compute_ratio",
        "source",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compile 800M realistic-setting results into a CSV with compute ratios.

Sources: results/scaling/realistic/{base,coreftaux,filtering,grmoe,lora,maxent}/800M/.
Seeds 1-3. Keeps rows for EVERY retain variant per (method, seed) so the
downstream aggregation can average over all retain variants per
label_class. For filtering, each retain variant lives in its own ts_dir,
so we walk latest-first and pick the latest ts_dir per retain set.

CR is computed against the seed_1 baseline losses.pkl
(power-law fit with linear extrapolation past the last training step).
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
    model_size_sort_key,
)


REPO_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
RESULTS_ROOT = REPO_RESULTS_ROOT / "scaling" / "realistic"
DEFAULT_OUT = Path(__file__).resolve().parent / "realistic.csv"

METHODS = ("base", "coreftaux", "filtering", "grmoe", "lora", "maxent")
SEEDS = ("seed_1", "seed_2", "seed_3")
SIZE = "800M"
# Methods restricted to seed_1 with only the most-recent ts_dir.
SEED1_LATEST_ONLY = set()
# Methods that span all seeds but use only the OLDEST ts_dir per seed.
ALL_SEEDS_OLDEST_TS = set()
# Explicit ts_dir pin per (method, seed). When set, overrides the
# latest/oldest selection and uses exactly this ts_dir for that seed.
TS_PINS: dict[str, dict[str, str]] = {
    "coreftaux": {
        "seed_1": "20260526204750347853",
        "seed_2": "20260527062635525021",
        "seed_3": "20260527160602990792",
    },
    "maxent": {  # most recent maxent batch (20260527), all 3 seeds
        "seed_1": "20260527045134832649",
        "seed_2": "20260527045406108760",
        "seed_3": "20260527122922310033",
    },
}


# ---------------------------------------------------------------------------
# Baseline registry: per seed. CR math lives in common.compile.
# ---------------------------------------------------------------------------

def ts_dirs_latest_first(seed_dir: Path) -> list[Path]:
    candidates = [d for d in seed_dir.iterdir()
                  if d.is_dir() and (d / "stats.jsonl").exists()]
    return sorted(candidates, key=lambda d: d.name, reverse=True)


def load_baselines() -> dict[str, dict]:
    """Return {seed: {"curves", "max_steps", "ref_se", "num_params"}}."""
    size_dir = RESULTS_ROOT / "base" / SIZE
    out: dict[str, dict] = {}
    for seed in SEEDS:
        seed_dir = size_dir / seed
        if not seed_dir.is_dir():
            continue
        ts_dir = next(
            (d for d in ts_dirs_latest_first(seed_dir)
             if (d / "baseline" / "losses.pkl").exists()),
            None,
        )
        if ts_dir is None:
            continue
        base = build_baseline(ts_dir)
        if base is not None:
            out[seed] = base
    return out


# ---------------------------------------------------------------------------
# Method row collection.
# ---------------------------------------------------------------------------

def collect_method_rows(method: str, baselines: dict,
                        baseline_seed: str | None = None) -> list[dict]:
    size_dir = RESULTS_ROOT / method / SIZE
    rows: list[dict] = []
    if not size_dir.is_dir():
        return rows
    # maxent is restricted to seed_1 / latest ts_dir only.
    # coreftaux spans all seeds but uses only the OLDEST ts_dir per seed.
    if method in SEED1_LATEST_ONLY:
        seeds = ("seed_1",)
    else:
        seeds = SEEDS
    for seed in seeds:
        seed_dir = size_dir / seed
        if not seed_dir.is_dir():
            continue
        base = baselines.get(baseline_seed or seed)
        if base is None:
            continue

        # Walk ts_dirs latest-first; keep one row-set per distinct retain
        # variant (the latest ts_dir wins). For most methods all retain
        # variants live in a single ts_dir; for `filtering` they're split
        # across separate ts_dirs.
        ts_dirs = ts_dirs_latest_first(seed_dir)
        pin = TS_PINS.get(method, {}).get(seed)
        if pin is not None:
            ts_dirs = [d for d in ts_dirs if d.name == pin]
            if not ts_dirs:
                print(f"  WARN {method}/{SIZE}/{seed}: pinned ts {pin} not found")
        elif method in SEED1_LATEST_ONLY:
            ts_dirs = ts_dirs[:1]
        elif method in ALL_SEEDS_OLDEST_TS:
            ts_dirs = ts_dirs[-1:]  # oldest = last in latest-first order
        picked_by_retained: dict[tuple, list[dict]] = {}
        picked_path_by_retained: dict[tuple, Path] = {}
        for ts_dir in ts_dirs:
            stats_path = ts_dir / "stats.jsonl"
            ts_rows: dict[tuple, list[dict]] = {}
            with open(stats_path) as f:
                for line in f:
                    rec = json.loads(line)
                    key = tuple(sorted(rec.get("retained", [])))
                    ts_rows.setdefault(key, []).append(rec)
            for key, recs in ts_rows.items():
                if key not in picked_by_retained:
                    picked_by_retained[key] = recs  # first = latest wins
                    picked_path_by_retained[key] = stats_path
        if not picked_by_retained:
            print(f"  skip {method}/{SIZE}/{seed}: no rows")
            continue

        for key, recs in picked_by_retained.items():
            stats_path = picked_path_by_retained[key]
            source = stats_path.relative_to(REPO_RESULTS_ROOT).as_posix()
            for rec in recs:
                loss = rec["loss"]
                rows.append({
                    "method": method,
                    "model_size": SIZE,
                    "num_params": base["num_params"],
                    "seed": seed,
                    "retained": "+".join(sorted(rec.get("retained", []))),
                    "data_label": rec["data_label"],
                    "label_class": label_class(
                        rec["data_label"], rec["retained"], rec["elicited"]),
                    "loss": loss,
                    "compute_ratio": compute_cr(loss, rec["data_label"], base),
                    "source": source,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-seed", type=str, default="",
                        help="By default each run is normalized against its "
                             "OWN seed's baseline. Pass a seed name (e.g. "
                             "seed_1) to pin every run to that seed instead.")
    args = parser.parse_args()

    print(f"Fitting baselines for {SIZE} per seed ...")
    baselines = pool_baselines(load_baselines())
    print(f"  {len(baselines)} baselines: {sorted(baselines.keys())}")
    if args.baseline_seed:
        print(f"  using {args.baseline_seed} baseline for ALL method runs")

    all_rows: list[dict] = []
    for method in METHODS:
        print(f"Collecting {method} ...")
        rs = collect_method_rows(method, baselines, args.baseline_seed)
        print(f"  {len(rs)} rows")
        all_rows.extend(rs)

    all_rows.sort(key=lambda r: (
        r["method"], r["seed"], r["data_label"], r["label_class"],
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

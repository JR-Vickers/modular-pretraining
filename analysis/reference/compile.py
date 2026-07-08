#!/usr/bin/env python3
"""
Compile 200M reference-setting results into a CSV with compute ratios.

Source: results/reference/200M/{seed}/{ts_dir}/. Unlike the `realistic`
setting (where each method lives in its own top-level dir), the reference
runs pack every stage into a SINGLE run dir:

    {ts_dir}/stats.jsonl   # rows for baseline / filtering / routed stages
    {ts_dir}/baseline/losses.pkl

so the "method" is the stats.jsonl stage name. We emit one CSV row per
stats.jsonl eval record for the `filtering` and `routed` stages, keeping
both elicited and non-elicited rows (mirrors realistic/compile.py).

CR is computed against the baseline `losses.pkl` of the baseline-seed run
(power-law fit with linear extrapolation past the last training step) --
identical math to analysis/realistic/compile.py.
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
)


RESULTS_ROOT = (Path(__file__).resolve().parents[2]
                / "results" / "reference")
DEFAULT_OUT = Path(__file__).resolve().parent / "reference.csv"

# Stages emitted as "methods" in the CSV (top row / bottom row of the plot).
METHODS = ("filtering", "routed")
SEEDS = ("seed_1", "seed_2", "seed_3")
SIZE = "200M"

# Explicit ts_dir pin per (method, seed). When set, that method's rows for
# that seed are read from exactly this ts_dir instead of the default latest
# run-with-baseline. Used to pull a GRAM-only rerun in from its own dir; the
# baseline (and other methods) still come from the default run, so CR stays
# pinned to the original baseline.
TS_PINS: dict[str, dict[str, str]] = {
    "routed": {  # GRAM rerun for seed_1 (routed stage only)
        "seed_1": "20260622174855196949",
    },
}


# ---------------------------------------------------------------------------
# Run-dir helpers + per-seed baseline registry. CR math: common.compile.
# ---------------------------------------------------------------------------

def ts_dirs_latest_first(seed_dir: Path) -> list[Path]:
    candidates = [d for d in seed_dir.iterdir()
                  if d.is_dir() and (d / "stats.jsonl").exists()]
    return sorted(candidates, key=lambda d: d.name, reverse=True)


def run_dir_for_seed(seed: str) -> Path | None:
    """Latest ts_dir for a seed that has both stats.jsonl and a baseline pkl."""
    seed_dir = RESULTS_ROOT / SIZE / seed
    if not seed_dir.is_dir():
        return None
    return next(
        (d for d in ts_dirs_latest_first(seed_dir)
         if (d / "baseline" / "losses.pkl").exists()),
        None,
    )


def load_baselines() -> dict[str, dict]:
    """Return {seed: {"curves", "max_steps", "ref_se", "num_params"}}."""
    out: dict[str, dict] = {}
    for seed in SEEDS:
        ts_dir = run_dir_for_seed(seed)
        if ts_dir is None:
            continue
        base = build_baseline(ts_dir)
        if base is not None:
            out[seed] = base
    return out


# ---------------------------------------------------------------------------
# Method (== stage) row collection.
# ---------------------------------------------------------------------------

def method_run_dir(method: str, seed: str) -> Path | None:
    """ts_dir to read `method`'s rows from for `seed`.

    Honors TS_PINS[method][seed] (an exact ts_dir, which need not contain a
    baseline); otherwise falls back to the default latest run-with-baseline.
    """
    pin = TS_PINS.get(method, {}).get(seed)
    if pin is not None:
        d = RESULTS_ROOT / SIZE / seed / pin
        if (d / "stats.jsonl").exists():
            return d
        print(f"  WARN {method}/{seed}: pinned ts {pin} not found")
        return None
    return run_dir_for_seed(seed)


def collect_method_rows(method: str, baselines: dict,
                        baseline_seed: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        ts_dir = method_run_dir(method, seed)
        if ts_dir is None:
            continue
        base = baselines.get(baseline_seed or seed)
        if base is None:
            continue

        n_seed = 0
        stats_path = ts_dir / "stats.jsonl"
        with open(stats_path) as f:
            for line in f:
                rec = json.loads(line)
                if (rec.get("stage") or {}).get("name") != method:
                    continue
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
                    "source": stats_path.relative_to(
                        RESULTS_ROOT.parent).as_posix(),
                })
                n_seed += 1
        if n_seed == 0:
            print(f"  skip {method}/{SIZE}/{seed}: no rows")
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
        if args.baseline_seed not in baselines:
            raise SystemExit(
                f"baseline-seed {args.baseline_seed!r} has no baseline; "
                f"available: {sorted(baselines)}")
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

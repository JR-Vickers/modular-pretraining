#!/usr/bin/env python3
"""
Compile scaling experiment results into scaling.csv with compute ratios.

Sources: results/scaling/realistic/{base,filtering,grmoe,lora}/.
Emits every eval row across all retain configs (seeds 1-3); selecting a single
config (e.g. core+papers-biology) is left to the plot step. For 5B, only the
"run_1" subdir is used (older timestamp dirs are ignored).

CR is computed per seed against that seed's own baseline losses.pkl at each
model size (power-law fit with linear extrapolation past the last training
step). Pass --baseline-seed to pin all runs to one seed's baseline instead.
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


RESULTS_ROOT = (Path(__file__).resolve().parents[2]
                / "results" / "scaling" / "realistic")
DEFAULT_OUT = Path(__file__).resolve().parent / "scaling.csv"

METHODS = ("filtering", "grmoe", "lora")
SEEDS = ("seed_1", "seed_2", "seed_3")
RUN_1_DIR = "run_1"


# ---------------------------------------------------------------------------
# Baseline registry: per (model_size, seed). CR math lives in common.compile.
# ---------------------------------------------------------------------------

def ts_dirs_latest_first(seed_dir: Path, *, is_5b: bool) -> list[Path]:
    # 5B always uses ONLY the run_1 subdir (heterogeneous accumulation),
    # ignoring any older timestamp dirs. Other sizes use every ts dir.
    if is_5b:
        candidates = [
            d for d in seed_dir.iterdir()
            if d.is_dir() and d.name == RUN_1_DIR
            and (d / "stats.jsonl").exists()
        ]
    else:
        candidates = [
            d for d in seed_dir.iterdir()
            if d.is_dir() and (d / "stats.jsonl").exists()
        ]
    return sorted(candidates, key=lambda d: d.name, reverse=True)


def load_baselines() -> dict[tuple[str, str], dict]:
    """Return {(size, seed): {"curves", "max_steps", "ref_se", "num_params"}}."""
    base_dir = RESULTS_ROOT / "base"
    out: dict[tuple[str, str], dict] = {}
    for size_dir in sorted(base_dir.iterdir(), key=lambda d: model_size_sort_key(d.name)):
        if not size_dir.is_dir():
            continue
        size = size_dir.name
        is_5b = (size == "5B")
        for seed in SEEDS:
            seed_dir = size_dir / seed
            if not seed_dir.is_dir():
                continue
            ts_dir = next(
                (d for d in ts_dirs_latest_first(seed_dir, is_5b=is_5b)
                 if (d / "baseline" / "losses.pkl").exists()),
                None,
            )
            if ts_dir is None:
                continue
            base = build_baseline(ts_dir)
            if base is not None:
                out[(size, seed)] = base
    return out


# ---------------------------------------------------------------------------
# Method row collection.
# ---------------------------------------------------------------------------

def collect_method_rows(method: str, baselines: dict) -> list[dict]:
    method_dir = RESULTS_ROOT / method
    rows: list[dict] = []
    if not method_dir.exists():
        return rows
    repo_results = RESULTS_ROOT.parents[1]   # .../arXiv-Codebase/results
    for size_dir in sorted(method_dir.iterdir(), key=lambda d: model_size_sort_key(d.name)):
        if not size_dir.is_dir():
            continue
        size = size_dir.name
        is_5b = (size == "5B")
        for seed in SEEDS:
            seed_dir = size_dir / seed
            if not seed_dir.is_dir():
                continue
            base = baselines.get((size, seed))
            if base is None:
                continue

            # Emit every eval row across all retain configs. Walk ts_dirs
            # latest-first and dedup by retain set (latest ts wins); for
            # `filtering`, distinct retain sets live in distinct ts_dirs, so
            # this gathers all of them. Plot-time filtering (e.g. to the
            # core+papers-biology config) belongs downstream, not here.
            picked_by_retained: dict[tuple, list[dict]] = {}
            src_by_retained: dict[tuple, Path] = {}
            for ts_dir in ts_dirs_latest_first(seed_dir, is_5b=is_5b):
                stats_path = ts_dir / "stats.jsonl"
                ts_rows: dict[tuple, list[dict]] = {}
                with open(stats_path) as f:
                    for line in f:
                        rec = json.loads(line)
                        key = tuple(sorted(rec.get("retained", [])))
                        ts_rows.setdefault(key, []).append(rec)
                for key, recs in ts_rows.items():
                    if key not in picked_by_retained:  # first (latest) wins
                        picked_by_retained[key] = recs
                        src_by_retained[key] = stats_path
            if not picked_by_retained:
                print(f"  skip {method}/{size}/{seed}: no rows")
                continue

            for key, recs in picked_by_retained.items():
                source = src_by_retained[key].relative_to(repo_results).as_posix()
                for rec in recs:
                    loss = rec["loss"]
                    rows.append({
                        "method": method,
                        "model_size": size,
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
    args = parser.parse_args()

    print("Fitting baselines per (size, seed) ...")
    baselines = pool_baselines(load_baselines(), group_fn=lambda k: k[0])
    print(f"  {len(baselines)} baselines: "
          f"{sorted(baselines.keys(), key=lambda k: (model_size_sort_key(k[0]), k[1]))}")

    all_rows: list[dict] = []
    for method in METHODS:
        print(f"Collecting {method} ...")
        rs = collect_method_rows(method, baselines)
        print(f"  {len(rs)} rows")
        all_rows.extend(rs)

    all_rows.sort(key=lambda r: (
        r["method"], model_size_sort_key(r["model_size"]), r["seed"],
        r["data_label"], r["label_class"],
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

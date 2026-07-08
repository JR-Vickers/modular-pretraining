#!/usr/bin/env python3
"""
Compile partial-labeling experiment results into a CSV with compute ratios.

Source: results/partial/<size>/<seed>/<timestamp>/. Each timestamp dir
contains a baseline + routed_01 (MoE) + routed_02 (LoRA) + filtering stage.
CR is computed against the per-(size, seed) baseline curve fit from
baseline/losses.pkl.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.common.compile import (
    build_baseline, compute_cr, label_class, model_size_sort_key, pool_baselines,
)


REPO_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
RESULTS_ROOT = REPO_RESULTS_ROOT / "partial"
DEFAULT_OUT = Path(__file__).resolve().parent / "partial.csv"


# ---------------------------------------------------------------------------
# Per-size baseline registry (seed_1). CR math lives in common.compile.
# ---------------------------------------------------------------------------

def latest_ts_dir(seed_dir: Path) -> Path | None:
    cands = [d for d in seed_dir.iterdir()
             if d.is_dir() and (d / "stats.jsonl").exists()]
    return max(cands, key=lambda d: d.name) if cands else None


def all_ts_dirs(seed_dir: Path) -> list[Path]:
    return sorted([d for d in seed_dir.iterdir()
                   if d.is_dir() and (d / "stats.jsonl").exists()],
                  key=lambda d: d.name)


def baseline_ts_dir(seed_dir: Path) -> Path | None:
    for d in all_ts_dirs(seed_dir):
        if (d / "baseline" / "losses.pkl").exists():
            return d
    return None


# ---------------------------------------------------------------------------
# Stage row extraction.
# ---------------------------------------------------------------------------

def stage_dir_name(stage: dict) -> str | None:
    """e.g. 'routed_01', 'filtering'. Strips the 'core' suffix on filtering."""
    res = stage.get("res_dir") or ""
    if not res:
        return None
    name = Path(res).name
    if stage.get("name") == "filtering" and name == "core":
        return Path(res).parent.name
    return name


def collect_rows(ts_dir: Path, size: str, seed: str, base: dict) -> list[dict]:
    rows = []
    exp_id = ts_dir.name
    stats_path = ts_dir / "stats.jsonl"
    source = stats_path.relative_to(REPO_RESULTS_ROOT).as_posix()
    with open(stats_path) as f:
        for line in f:
            rec = json.loads(line)
            stage = rec.get("stage") or {}
            if stage.get("name") == "baseline":
                continue
            retained = rec.get("retained") or []
            loss = rec["loss"]
            data_label = rec["data_label"]
            rows.append({
                "exp_id": exp_id,
                "method": stage.get("name"),
                "stage_dir": stage_dir_name(stage),
                "arch": (stage.get("model") or {}).get("arch"),
                "model_size": size,
                "num_params": base["num_params"],
                "seed": seed,
                "retained": "+".join(sorted(retained)),
                "data_label": data_label,
                "label_class": label_class(
                    data_label, retained, bool(rec.get("elicited"))),
                "label_prc": stage.get("label_prc"),
                "aux_route_prc": stage.get("aux_route_prc"),
                "loss": loss,
                "compute_ratio": compute_cr(loss, data_label, base),
                "source": source,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for size_dir in sorted(RESULTS_ROOT.iterdir(),
                           key=lambda d: model_size_sort_key(d.name)):
        if not size_dir.is_dir():
            continue
        seed_dirs = [d for d in sorted(size_dir.iterdir()) if d.is_dir()]
        # pooled curve + common denominator across this size's seeds
        raw = {}
        for seed_dir in seed_dirs:
            bt = baseline_ts_dir(seed_dir)
            raw[seed_dir.name] = build_baseline(bt) if bt is not None else None
        bases = pool_baselines(raw)
        for seed_dir in seed_dirs:
            base = bases.get(seed_dir.name)
            if base is None:
                print(f"  skip {size_dir.name}/{seed_dir.name}: no baseline")
                continue
            for ts_dir in all_ts_dirs(seed_dir):
                rs = collect_rows(ts_dir, size_dir.name, seed_dir.name, base)
                print(f"{size_dir.name}/{seed_dir.name}/{ts_dir.name}: {len(rs)} rows")
                all_rows.extend(rs)

    all_rows.sort(key=lambda r: (
        model_size_sort_key(r["model_size"]), r["seed"], r["exp_id"],
        r["stage_dir"] or "", r["retained"], r["data_label"],
    ))

    fieldnames = [
        "exp_id", "method", "stage_dir", "arch",
        "model_size", "num_params", "seed",
        "retained", "data_label", "label_class",
        "label_prc", "aux_route_prc",
        "loss", "compute_ratio",
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

#!/usr/bin/env python3
"""
Compile arch/100M results into arch.csv.

Sources: results/arch/100M/seed_{1,2,3}/<ts_dir>/. For each seed we fit a
power law on that seed's own baseline/losses.pkl and convert every loss in
stats.jsonl into a compute ratio against that same per-seed baseline.

Rows include the baseline itself (method='baseline') plus the four
architecture/training configurations: GR-MoE / GR-LoRA / FT-MoE / FT-LoRA.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.common.compile import (
    build_baseline, compute_cr, label_class, pool_baselines,
)


ARCH_DIR = Path(__file__).resolve().parents[2] / "results" / "arch" / "100M"
RESULTS_ROOT = ARCH_DIR.parents[1]
DEFAULT_OUT = Path(__file__).resolve().parent / "arch.csv"


def config_name(arch: str | None, aux_route_prc) -> str:
    gr = "GR" if aux_route_prc is not None else "FT"
    a = "MoE" if arch == "moe" else "LoRA"
    return f"{gr}-{a}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    all_rows: list[dict] = []
    # collect each seed's latest ts, then pool the curve + common denominator
    seed_ts: dict = {}
    for seed_dir in sorted(ARCH_DIR.iterdir()):
        if not seed_dir.is_dir():
            continue
        ts_dirs = [d for d in seed_dir.iterdir()
                   if d.is_dir() and (d / "stats.jsonl").exists()]
        if not ts_dirs:
            print(f"  skip {seed_dir.name}: no ts_dir with stats.jsonl")
            continue
        seed_ts[seed_dir.name] = sorted(ts_dirs, key=lambda d: d.name)[-1]
    bases = pool_baselines({s: build_baseline(ts) for s, ts in seed_ts.items()})
    for seed_name, ts in seed_ts.items():
        base = bases.get(seed_name)
        if base is None:
            print(f"  skip {seed_name}: no baseline in {ts.name}")
            continue
        print(f"Processing {seed_name}/{ts.name}")

        stats_path = ts / "stats.jsonl"
        with open(stats_path) as f:
            for line in f:
                rec = json.loads(line)
                st = rec.get("stage") or {}
                if st.get("name") == "baseline":
                    method = "baseline"
                else:
                    method = config_name((st.get("model") or {}).get("arch"),
                                         st.get("aux_route_prc"))
                lc = label_class(rec["data_label"], rec["retained"],
                                 bool(rec.get("elicited")))
                cr = compute_cr(rec["loss"], rec["data_label"], base)
                all_rows.append({
                    "method": method,
                    "seed": seed_name,
                    "retained": "+".join(sorted(rec.get("retained", []))),
                    "data_label": rec["data_label"],
                    "label_class": lc,
                    "elicited": bool(rec.get("elicited")),
                    "loss": rec["loss"],
                    "compute_ratio": cr,
                    "source": stats_path.relative_to(RESULTS_ROOT).as_posix(),
                })

    all_rows.sort(key=lambda r: (r["method"], r["seed"],
                                  r["data_label"], r["label_class"]))
    fieldnames = ["method", "seed", "retained", "data_label",
                  "label_class", "elicited", "loss", "compute_ratio", "source"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

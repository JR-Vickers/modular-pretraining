#!/usr/bin/env python3
"""
Compile the auxnum experiment into auxnum.csv with compute ratios.

The auxnum sweep asks how routing quality changes as the number of auxiliary
categories grows. Each ``num_aux`` is a self-contained stories run
(``src/run/experiment/auxnum/run.py``): a baseline stage (no routing) followed
by an unordered-routed stage that sweeps retain configs (core + each single
aux) and evaluates every category, with and without elicitation FT.

Source (newest timestamp dir per num_aux/seed)::

    results/auxnum/num_aux_<N>/seed_<S>/<ts>/stats.jsonl       # eval rows
    results/auxnum/num_aux_<N>/seed_<S>/<ts>/baseline/losses.pkl

CR is computed per seed against that ``num_aux`` run's OWN baseline -- each
num_aux trains on a different number of aux categories, so baselines are not
shared across num_aux. Within a num_aux the three seeds share a pooled curve +
common denominator (``analysis.common.compile.pool_baselines``), matching every
other experiment's CR methodology.

Each routed eval row is bucketed by ``analysis.common.compile.label_class``:

    core             data_label == "core"          (always retained)
    retain           data_label in the retained set
    forget           a forgotten aux, not elicited
    elicited_forget  a forgotten aux, after elicitation FT

These are the four lines in ``paper/figures/aux_scaling.tex``; ``plot.py``
aggregates this CSV into that figure with x = number of aux categories.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from analysis.common.compile import (
    build_baseline,
    compute_cr,
    label_class,
    latest_baseline_ts,
    pool_baselines,
)


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
AUXNUM_ROOT = RESULTS_ROOT / "auxnum"
DEFAULT_OUT = SCRIPT_DIR / "auxnum.csv"

SEEDS = ("seed_1", "seed_2", "seed_3")


def num_aux_values(root: Path) -> list[int]:
    """Sorted list of N for every ``num_aux_<N>`` dir under ``root``."""
    out: list[int] = []
    if not root.is_dir():
        return out
    for d in root.iterdir():
        m = re.fullmatch(r"num_aux_(\d+)", d.name)
        if m and d.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def collect_rows(num_aux: int, root: Path) -> list[dict]:
    """All routed eval rows for one num_aux, with CR against its own baseline.

    The three seeds of this num_aux share a pooled curve + common denominator,
    so their CRs are directly comparable; each seed's routed rows are then
    normalized against that shared reference.
    """
    rows: list[dict] = []

    # Per-seed baselines for THIS num_aux, then pool across the seeds.
    ts_by_seed: dict[str, Path | None] = {}
    bases: dict[str, dict | None] = {}
    for seed in SEEDS:
        seed_dir = root / f"num_aux_{num_aux}" / seed
        ts = latest_baseline_ts(seed_dir) if seed_dir.is_dir() else None
        ts_by_seed[seed] = ts
        bases[seed] = build_baseline(ts) if ts is not None else None
    bases = pool_baselines(bases)

    for seed in SEEDS:
        base = bases.get(seed)
        ts = ts_by_seed.get(seed)
        if base is None or ts is None:
            print(f"  skip num_aux_{num_aux}/{seed}: no usable baseline")
            continue
        stats_path = ts / "stats.jsonl"
        print(f"  num_aux_{num_aux}/{seed}: {ts.name}")
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (rec.get("stage") or {}).get("name") != "routed":
                    continue
                dl = rec["data_label"]
                retained = rec.get("retained", [])
                cr = compute_cr(rec["loss"], dl, base)
                if cr is None:
                    continue
                rows.append({
                    "num_aux": num_aux,
                    "num_params": base["num_params"],
                    "seed": seed,
                    "retained": "+".join(sorted(retained)),
                    "data_label": dl,
                    "label_class": label_class(
                        dl, retained, bool(rec.get("elicited"))),
                    "elicited": bool(rec.get("elicited")),
                    "loss": rec["loss"],
                    "compute_ratio": cr,
                    "source": stats_path.relative_to(RESULTS_ROOT).as_posix(),
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subdir", default=None,
        help="Subdirectory under results/auxnum/ to compile (e.g. 'v2', 'v3', "
             "'alt'). Default: results/auxnum/ itself. Output defaults to "
             "auxnum.csv regardless, so pass -o to keep variants separate.")
    parser.add_argument(
        "--alt", action="store_true",
        help="Shorthand for --subdir alt that also writes auxnum_alt.csv.")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    subdir = args.subdir if args.subdir is not None else ("alt" if args.alt else None)
    root = AUXNUM_ROOT / subdir if subdir else AUXNUM_ROOT
    default_name = "auxnum_alt.csv" if (args.alt and args.subdir is None) else "auxnum.csv"
    output = args.output or (SCRIPT_DIR / default_name)

    if not root.is_dir():
        raise SystemExit(f"No auxnum results under {root}")

    nvals = num_aux_values(root)
    print(f"root: {root}")
    print(f"num_aux values: {nvals}")

    all_rows: list[dict] = []
    for n in nvals:
        rs = collect_rows(n, root)
        print(f"num_aux_{n}: {len(rs)} rows")
        all_rows.extend(rs)

    all_rows.sort(key=lambda r: (
        r["num_aux"], r["seed"], r["retained"], r["data_label"], r["elicited"],
    ))

    fieldnames = [
        "num_aux", "num_params", "seed", "retained", "data_label",
        "label_class", "elicited", "loss", "compute_ratio", "source",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compile the Simple Stories hyperparameter sweep into stories.csv (with CR).

Source (newest timestamp dir per seed)::

    results/sweep/stories/seed_<S>/<ts>/stats.jsonl        # eval rows
    results/sweep/stories/seed_<S>/<ts>/baseline/losses.pkl

Each run (``src/run/experiment/sweep/run_stories.py``) trains one baseline
stage followed by 16 routed stages that together form THREE one-knob sweeps:

  robust_prc      GRAM core robustness  (UnorderedConfig, arch=moe, arp fixed 0.3)
  aux_route_prc   GRAM aux spread       (UnorderedConfig, arch=moe, robust fixed 0.5)
  core_aux_ratio  FT-LoRA core:aux split(OrderedConfig,  arch=lora)

We recover which sweep a routed stage belongs to from its own stage config
(arch + which knob is held fixed); the three are mutually exclusive for this
experiment (robust never uses robust=0.5; aux_route never uses arp=0.3).

CR is computed per seed against that seed's OWN in-run baseline; the three seeds
share a pooled curve + common denominator (``analysis.common.compile`` machinery),
so CRs are directly comparable across sweep points and seeds. Each routed eval
row is bucketed into core / retain / forget / elicited_forget by
``analysis.common.compile.label_class`` -- the four lines plot.py draws per pane.
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
    latest_baseline_ts,
    pool_baselines,
)


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[2]            # .../arXiv-Codebase
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
STORIES_ROOT = RESULTS_ROOT / "sweep" / "stories"
DEFAULT_OUT = SCRIPT_DIR / "stories.csv"

SEEDS = ("seed_1", "seed_2", "seed_3")

# The robust sweep holds aux_route_prc at this value while varying robust_prc;
# the aux_route sweep holds robust_prc at 0.5 while varying aux_route_prc. These
# never collide (robust sweep never uses robust=0.5; aux_route never uses
# arp=0.3), so they disambiguate the two moe sweeps. Matches run_stories.py.
ROBUST_SWEEP_FIXED_ARP = 0.3


def classify_sweep(stage: dict) -> tuple[str, float] | None:
    """Return (sweep_name, x_value) for a routed stage, or None if unrecognized.

    sweep_name is the knob being swept; x_value is its setting for this stage.
    """
    model = stage.get("model") or {}
    arch = model.get("arch")
    if arch == "lora":
        x = stage.get("core_aux_ratio")
        return ("core_aux_ratio", x) if x is not None else None
    if arch == "moe":
        arp = stage.get("aux_route_prc")
        rp = stage.get("robust_prc")
        if arp is not None and abs(arp - ROBUST_SWEEP_FIXED_ARP) < 1e-9:
            return ("robust_prc", rp) if rp is not None else None
        return ("aux_route_prc", arp) if arp is not None else None
    return None


def collect_rows(seed: str, base: dict, ts: Path) -> list[dict]:
    """All routed eval rows for one seed, CR'd against the pooled baseline."""
    rows: list[dict] = []
    stats_path = ts / "stats.jsonl"
    source = stats_path.relative_to(RESULTS_ROOT).as_posix()
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stage = rec.get("stage") or {}
            if stage.get("name") != "routed":
                continue
            swept = classify_sweep(stage)
            if swept is None:
                continue
            sweep, x = swept
            dl = rec["data_label"]
            retained = rec.get("retained") or []
            cr = compute_cr(rec["loss"], dl, base)
            if cr is None:
                continue
            rows.append({
                "sweep": sweep,
                "x": x,
                "seed": seed,
                "arch": (stage.get("model") or {}).get("arch"),
                "robust_prc": stage.get("robust_prc"),
                "aux_route_prc": stage.get("aux_route_prc"),
                "core_aux_ratio": stage.get("core_aux_ratio"),
                "data_label": dl,
                "label_class": label_class(
                    dl, retained, bool(rec.get("elicited"))),
                "elicited": bool(rec.get("elicited")),
                "retained": "+".join(sorted(retained)),
                "loss": rec["loss"],
                "compute_ratio": cr,
                "source": source,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not STORIES_ROOT.is_dir():
        raise SystemExit(f"No sweep/stories results under {STORIES_ROOT}")

    # Per-seed baseline = latest run that actually trained one; pooled across
    # seeds (shared curve + denominator). Reused for every routed row of that
    # seed, including later runs that add sweep points without a new baseline.
    bases: dict[str, dict | None] = {}
    for seed in SEEDS:
        seed_dir = STORIES_ROOT / seed
        ts = latest_baseline_ts(seed_dir) if seed_dir.is_dir() else None
        bases[seed] = build_baseline(ts) if ts is not None else None
    bases = pool_baselines(bases)

    all_rows: list[dict] = []
    for seed in SEEDS:
        base = bases.get(seed)
        if base is None:
            print(f"  skip {seed}: no usable baseline")
            continue
        seed_dir = STORIES_ROOT / seed
        # Merge routed rows across ALL timestamp dirs for this seed: a later run
        # may add only a few sweep points (e.g. extra robust_prc values) without
        # retraining a baseline -- those rows reuse the baseline above. Process
        # oldest->newest so a newer rerun of the same point supersedes.
        ts_dirs = sorted(
            (d for d in seed_dir.iterdir()
             if d.is_dir() and (d / "stats.jsonl").exists()),
            key=lambda d: d.name,
        )
        merged: dict[tuple, dict] = {}
        for ts in ts_dirs:
            for r in collect_rows(seed, base, ts):
                key = (r["sweep"], r["x"], r["data_label"],
                       r["retained"], r["elicited"])
                merged[key] = r
        print(f"  {seed}: {len(ts_dirs)} ts dir(s) -> {len(merged)} merged rows")
        all_rows.extend(merged.values())

    all_rows.sort(key=lambda r: (
        r["sweep"], r["seed"], r["x"], r["data_label"], r["elicited"],
    ))

    fieldnames = [
        "sweep", "x", "seed", "arch", "robust_prc", "aux_route_prc",
        "core_aux_ratio", "data_label", "label_class", "elicited",
        "retained", "loss", "compute_ratio", "source",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

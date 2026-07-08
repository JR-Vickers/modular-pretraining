#!/usr/bin/env python3
"""
Compile the best-trial MaxEnt BO results (loss + compute ratio) for the
realistic 800M study, across seeds 1-3.

For each (seed, retain_target) study under
``results/optimize/maxent/realistic/800M/seed_N/<retain_key>/`` we read
``bo_summary.json`` to find the best trial, then pull that trial's eval rows
(``bo_trial_NNN/stats.jsonl``) for the loss per data_label.

CR is normalized per seed: each seed's rows are divided by that seed's own
800M baseline under results/scaling/realistic/base/800M/seed_N/ (power-law fit
with linear extrapolation past the last training step, shared
``analysis.common.compile`` machinery).

Output CSV at ``analysis/optimize/maxent/realistic/maxent_realistic.csv``.

Usage:
    python analysis/optimize/maxent/realistic/compile.py
    python analysis/optimize/maxent/realistic/compile.py -o /tmp/foo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[3]   # .../arXiv-Codebase
sys.path.insert(0, str(EXPERIMENT_ROOT))

from analysis.common.compile import (                   # noqa: E402
    build_baseline,
    compute_cr,
    compute_ppl_ratio,
    latest_baseline_ts,
)


MODEL_SIZE = "800M"
SEEDS = ("seed_1", "seed_2", "seed_3")

RESULTS = EXPERIMENT_ROOT / "results"
STUDY_ROOT = RESULTS / "optimize" / "maxent" / "realistic" / MODEL_SIZE
# Each seed's CR is normalized against that seed's own 800M baseline.
BASE_DIR = RESULTS / "scaling" / "realistic" / "base" / MODEL_SIZE

DEFAULT_OUT = SCRIPT_DIR / "maxent_realistic.csv"


def fit_baselines() -> dict[str, dict]:
    """{seed: baseline} from base/800M/seed_N (each seed's own baseline)."""
    out: dict[str, dict] = {}
    for seed in SEEDS:
        seed_dir = BASE_DIR / seed
        if not seed_dir.is_dir():
            continue
        ts = latest_baseline_ts(seed_dir)
        if ts is None:
            continue
        base = build_baseline(ts)
        if base is not None:
            out[seed] = base
    return out


def label_class(data_label: str, retained: list[str], elicited: bool = False) -> str:
    if data_label == "core":
        return "core"
    if data_label in retained:
        return "retain"
    return "elicited_forget" if elicited else "forget"


def _eval_rows(stats_path: Path) -> list[dict]:
    """Every do_eval row from a stats.jsonl (content filtering — e.g. by
    elicited — is left to the plot step, not baked into the CSV)."""
    rows = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("function") != "do_eval":
                continue
            rows.append(rec)
    return rows


def _best_trial_number(summary: dict) -> int:
    best_v = summary["best_value"]
    return next(t["number"] for t in summary["all_trials"] if t["value"] == best_v)


def collect(baselines: dict[str, dict]) -> list[dict]:
    """Best-trial loss + CR rows across all (seed, retain_key) studies,
    each seed normalized against its own baseline."""
    rows: list[dict] = []

    for seed in SEEDS:
        seed_dir = STUDY_ROOT / seed
        if not seed_dir.is_dir():
            continue
        base = baselines.get(seed)
        if base is None:
            print(f"  skip {seed}: no baseline")
            continue
        num_params = base["num_params"]
        for study_dir in sorted(seed_dir.iterdir()):
            summary_path = study_dir / "bo_summary.json"
            if not study_dir.is_dir() or not summary_path.exists():
                continue
            key = study_dir.name
            summary = json.loads(summary_path.read_text())
            best_n = _best_trial_number(summary)
            trial_dir = study_dir / f"bo_trial_{best_n:03d}"
            stats_path = trial_dir / "stats.jsonl"
            if not stats_path.exists():
                print(f"  WARN: missing {stats_path}")
                continue

            best_lr = summary["best_params"]["lr"]
            best_ar = summary["best_params"]["alpha_retain"]
            source = stats_path.relative_to(RESULTS).as_posix()
            for rec in _eval_rows(stats_path):
                dl = rec["data_label"]
                retained = sorted(rec.get("retained") or [])
                elicited = bool(rec.get("elicited", False))
                loss = rec["loss"]
                rows.append({
                    "method": "maxent_bo",
                    "model_size": MODEL_SIZE,
                    "num_params": num_params,
                    "seed": seed,
                    "retain_key": key,
                    "experiment_id": trial_dir.name,
                    "retained": "+".join(retained),
                    "data_label": dl,
                    "label_class": label_class(dl, retained, elicited),
                    "elicited": elicited,
                    "loss": loss,
                    "compute_ratio": compute_cr(loss, dl, base),
                    "ppl_ratio": compute_ppl_ratio(loss, dl, base),
                    "lr": best_lr,
                    "alpha_retain": best_ar,
                    "trial_number": best_n,
                    "bo_score": summary["best_value"],
                    "source": source,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Fitting per-seed baselines from {BASE_DIR}")
    baselines = fit_baselines()
    if not baselines:
        raise SystemExit(f"no baselines found under {BASE_DIR}")
    print(f"  baselines: {sorted(baselines.keys())}")

    print(f"Collecting best-trial rows from {STUDY_ROOT} ...")
    rows = collect(baselines)
    print(f"  {len(rows)} rows")

    rows.sort(key=lambda r: (
        r["seed"], r["retain_key"], r["label_class"], r["data_label"],
    ))

    fieldnames = [
        "method", "model_size", "num_params", "seed", "retain_key",
        "experiment_id", "retained", "data_label", "label_class", "elicited",
        "loss", "compute_ratio", "ppl_ratio",
        "lr", "alpha_retain", "trial_number", "bo_score", "source",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  seeds: {sorted({r['seed'] for r in rows})}")
    print(f"  studies: {sorted({r['retain_key'] for r in rows})}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compile the SimpleStories results into a CSV with compute ratios, mirroring
analysis/realistic/compile.py.

Sources: results/stories/seed_{1,2,3}/<ts>/ with:
  - baseline/losses.pkl  (reference val curves per data_label, for CR)
  - stats.jsonl          (per-(stage, data_label, retained, elicited) losses)

Two methods are emitted:
  - filtering : the per-retain-set filtering models
                (retained = core, core+<story>, ...)
  - grmoe     : the single GRAM model (routed_01) evaluated with each module
                active (same retained sets)

CR is computed against this seed's own baseline losses.pkl
(power-law fit with linear extrapolation past the last training step).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.common.compile import (
    build_baseline, compute_cr, label_class, pool_baselines,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_ROOT = RESULTS_DIR / "stories"
OUT = Path(__file__).resolve().parent / "stories.csv"

SEEDS = ("seed_1", "seed_2", "seed_3")
MODEL_SIZE = "1.25M"
# routed runs are distinguished by model arch.
ARCH_TO_METHOD = {"moe": "grmoe", "lora": "lora", "demix": "demix"}


def baseline_ts_for(seed_dir: Path) -> Path | None:
    """ts_dir that holds baseline/losses.pkl (the original full run)."""
    cands = [d for d in seed_dir.iterdir()
             if d.is_dir() and (d / "baseline" / "losses.pkl").exists()]
    return sorted(cands, key=lambda d: d.name)[0] if cands else None


def alt_gram_ts_for(seed_dir: Path, arp: float | None = None) -> Path | None:
    """ts_dir for a new-hyperparam GRAM run: a standalone `routed` subdir (vs
    the original run's routed_01/02/03). If `arp` is given, pick the run whose
    aux_route_prc matches it; otherwise the newest such ts."""
    cands = [d for d in seed_dir.iterdir()
             if d.is_dir() and (d / "routed").is_dir()
             and (d / "stats.jsonl").exists()]
    cands.sort(key=lambda d: d.name, reverse=True)
    if arp is None:
        return cands[0] if cands else None
    for d in cands:
        try:
            v = json.load(open(d / "routed" / "stage.json"))["stage"].get("aux_route_prc")
        except (KeyError, json.JSONDecodeError, FileNotFoundError):
            v = None
        if v is not None and abs(float(v) - arp) < 1e-9:
            return d
    return None


def res_basename(rec: dict) -> str:
    return Path((rec.get("stage") or {}).get("res_dir", "")).name


def maxent_ts_for(seed_dir: Path) -> Path | None:
    """ts_dir holding the maxent runs (has a maxent_01 subdir)."""
    cands = [d for d in seed_dir.iterdir()
             if d.is_dir() and (d / "maxent_01").is_dir()
             and (d / "stats.jsonl").exists()]
    return sorted(cands, key=lambda d: d.name, reverse=True)[0] if cands else None


def coreftaux_ts_for(seed_dir: Path) -> Path | None:
    """ts_dir holding the FT-Full run (has a coreftaux subdir)."""
    cands = [d for d in seed_dir.iterdir()
             if d.is_dir() and (d / "coreftaux").is_dir()
             and (d / "stats.jsonl").exists()]
    return sorted(cands, key=lambda d: d.name, reverse=True)[0] if cands else None


def method_of(rec: dict) -> str | None:
    st = rec.get("stage") or {}
    name = st.get("name")
    if name == "filtering":
        return "filtering"
    if name == "maxent":
        return "maxent"
    if name == "coreftaux":
        return "coreftaux"
    if name == "routed":
        arch = (st.get("model") or {}).get("arch")
        return ARCH_TO_METHOD.get(arch)
    return None


def collect_stats(stats_path: Path, seed: str, num_params, base,
                  only_method: str | None = None,
                  skip_method: str | None = None) -> list[dict]:
    rows = []
    with open(stats_path) as f:
        for line in f:
            rec = json.loads(line)
            method = method_of(rec)
            if method is None:
                continue
            if only_method is not None and method != only_method:
                continue
            if skip_method is not None and method == skip_method:
                continue
            retained = rec.get("retained", [])
            lab = rec["data_label"]
            rows.append({
                "method": method,
                "model_size": MODEL_SIZE,
                "num_params": num_params,
                "seed": seed,
                "retained": "+".join(sorted(retained)),
                "data_label": lab,
                "label_class": label_class(lab, retained,
                                           rec.get("elicited", False)),
                "loss": rec["loss"],
                "compute_ratio": compute_cr(rec["loss"], lab, base),
                "source": stats_path.relative_to(RESULTS_DIR).as_posix(),
            })
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gram", choices=("std", "routed01"), default="std",
                    help="std (default): GRAM from a standalone `routed` run, "
                         "selected by --gram-arp (default 0.3 — the current "
                         "production setting). routed01: original routed_01 "
                         "(aux_route_prc=0.5). Baseline + filtering always come "
                         "from the original (baseline-bearing) ts.")
    ap.add_argument("--gram-arp", type=float, default=0.3,
                    help="With --gram std, select the standalone routed run by "
                         "aux_route_prc (e.g. 0.2, 0.3). Default 0.3.")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or (OUT.parent / "stories.csv")

    rows: list[dict] = []
    # gather per-seed baseline ts, then pool the curve + common denominator
    seed_info: dict = {}
    for seed in SEEDS:
        seed_dir = RESULTS_ROOT / seed
        if not seed_dir.is_dir():
            continue
        bts = baseline_ts_for(seed_dir)
        if bts is None:
            print(f"  skip {seed}: no baseline ts")
            continue
        seed_info[seed] = (seed_dir, bts)
    bases = pool_baselines({s: build_baseline(bt) for s, (sd, bt) in seed_info.items()})
    for seed, (seed_dir, bts) in seed_info.items():
        base = bases.get(seed)
        if base is None:
            print(f"  skip {seed}: baseline fit failed")
            continue
        num_params = base["num_params"]

        if args.gram == "routed01":
            rows += collect_stats(bts / "stats.jsonl", seed, num_params, base)
        else:
            # filtering / lora / demix from the original ts; grmoe from the
            # standalone routed run selected by aux_route_prc.
            rows += collect_stats(bts / "stats.jsonl", seed, num_params,
                                  base, skip_method="grmoe")
            ats = alt_gram_ts_for(seed_dir, args.gram_arp)
            if ats is None:
                print(f"  WARN {seed}: no standalone GRAM ts found "
                      f"(arp={args.gram_arp})")
            else:
                rows += collect_stats(ats / "stats.jsonl", seed, num_params,
                                      base, only_method="grmoe")

        # maxent: in-tree ts (its own baseline/maxent_01..05), all seeds.
        # CR vs this seed's own stories baseline.
        mx_ts = maxent_ts_for(seed_dir)
        if mx_ts is not None:
            rows += collect_stats(mx_ts / "stats.jsonl", seed, num_params,
                                  base, only_method="maxent")

        # FT-Full (coreftaux): its own ts dir; CR vs this seed's stories baseline.
        ct_ts = coreftaux_ts_for(seed_dir)
        if ct_ts is not None:
            rows += collect_stats(ct_ts / "stats.jsonl", seed, num_params,
                                  base, only_method="coreftaux")

    rows.sort(key=lambda r: (r["method"], r["seed"], r["retained"],
                             r["data_label"], r["label_class"]))
    fieldnames = ["method", "model_size", "num_params", "seed", "retained",
                  "data_label", "label_class", "loss", "compute_ratio",
                  "source"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

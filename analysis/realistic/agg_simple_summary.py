#!/usr/bin/env python3
"""
Summary CSV for the realistic_agg_simple figure.

Reproduces the exact aggregation the bars/error-bars use (analysis.common.plot
.aggregate_by_seed: mean within each seed first, then mean + t-CI across seeds
at ci_level=0.9), for the "alt" variant of realistic/plot.py -- methods
filtering / grmoe / lora / maxent, classes Core / Retain / Forget / Elicit.

Emits realistic_agg_simple_summary.csv: one row per (method, label_class)
aggregated value, with the mean compute ratio and the 90% t-CI half-width in a
dedicated `ci_90` column (plus explicit ci_low / ci_high bounds).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.common.plot import aggregate_by_seed

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "realistic.csv"
OUT = ROOT / "realistic_agg_simple_summary.csv"

CI_LEVEL = 0.9

# Match the "alt" variant of realistic/plot.py exactly (left-to-right order).
METHOD_ORDER = ["filtering", "grmoe", "lora", "maxent"]
METHOD_DISPLAY = {"filtering": "Filtering", "grmoe": "GRAM (ours)",
                  "lora": "Filter + LoRA", "maxent": "MaxEnt"}

CLASS_ORDER = ["core", "retain", "forget", "elicited_forget"]
CLASS_DISPLAY = {"core": "Core", "retain": "Retain",
                 "forget": "Forget", "elicited_forget": "Elicit"}
CLASS_DIRECTION = {"core": "up", "retain": "up",
                   "forget": "down", "elicited_forget": "down"}


def main() -> None:
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    df = df[df["method"].isin(METHOD_ORDER)]
    df = df[df["label_class"].isin(CLASS_ORDER)]

    agg = aggregate_by_seed(
        df, group_cols=["method", "label_class"], y_col="compute_ratio",
        seed_col="seed", ci_level=CI_LEVEL)

    # Order rows to match the figure (method groups L->R, classes within group).
    agg["method"] = pd.Categorical(agg["method"], METHOD_ORDER, ordered=True)
    agg["label_class"] = pd.Categorical(
        agg["label_class"], CLASS_ORDER, ordered=True)
    agg = agg.sort_values(["method", "label_class"]).reset_index(drop=True)

    out = pd.DataFrame({
        "method": agg["method"].astype(str).map(METHOD_DISPLAY),
        "metric": agg["label_class"].astype(str).map(CLASS_DISPLAY),
        "direction": agg["label_class"].astype(str).map(CLASS_DIRECTION),
        "compute_ratio_mean": agg["mean"].round(4),
        "ci_90": agg["ci"].round(4),
        "ci_low": (agg["mean"] - agg["ci"]).round(4),
        "ci_high": (agg["mean"] + agg["ci"]).round(4),
        "std": agg["std"].round(4),
        "sem": agg["sem"].round(4),
        "n_seeds": agg["count"].astype(int),
    })
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Summary CSV for the scaling.png figure.

Reproduces the exact aggregation the curves/CI-bands use (scaling/plot.py:
retained == "core+papers-biology" filter, clamp_elicited, then
aggregate_across_seeds -- mean within each seed first, then mean + t-CI across
seeds at ci_level=0.9), for all four panels and all three methods, at every
model size.

Emits scaling_agg_summary.csv: one row per (model size, method, panel/metric)
aggregated value, with the mean compute ratio and the 90% t-CI half-width in a
dedicated `ci_90` column (plus explicit ci_low / ci_high bounds).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.scaling.plot import (
    PANELS, METHOD_DISPLAY, aggregate_across_seeds, clamp_elicited,
)

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "scaling.csv"
OUT = ROOT / "scaling_agg_summary.csv"

CI_LEVEL = 0.9

# Panel (label_class) -> display exactly as the figure titles them.
PANEL_DISPLAY = dict(PANELS)  # core / retain / forget / elicited_forget
PANEL_DIRECTION = {"core": "up", "retain": "up",
                   "forget": "down", "elicited_forget": "down"}


def main() -> None:
    df = pd.read_csv(CSV)
    # Same filtering the figure applies.
    df = df[df["retained"] == "core+papers-biology"].copy()
    df["num_params"] = pd.to_numeric(df["num_params"], errors="coerce")
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["num_params", "compute_ratio"])
    df = clamp_elicited(df)

    # model_size label per num_params (for a human-readable column).
    size_lookup = (df.dropna(subset=["model_size"])
                     .groupby("num_params")["model_size"].first())

    rows: list[dict] = []
    for label_class, panel_disp in PANELS:
        df_class = df[df["label_class"] == label_class]
        for method, method_disp in METHOD_DISPLAY.items():
            sub = df_class[df_class["method"] == method]
            if sub.empty:
                continue
            agg = aggregate_across_seeds(sub, CI_LEVEL)
            for _, r in agg.iterrows():
                np_ = r["num_params"]
                rows.append({
                    "model_size": size_lookup.get(np_, ""),
                    "num_params": int(np_),
                    "method": method_disp,
                    "metric": panel_disp,
                    "direction": PANEL_DIRECTION[label_class],
                    "compute_ratio_mean": round(r["mean"], 4),
                    "ci_90": round(r["ci"], 4),
                    "ci_low": round(r["mean"] - r["ci"], 4),
                    "ci_high": round(r["mean"] + r["ci"], 4),
                    "std": round(r["std"], 4),
                    "n_seeds": int(r["count"]),
                })

    out = pd.DataFrame(rows)
    # Order: metric (panel) L->R, method, then ascending size -- matches reading
    # the figure panel-by-panel, line-by-line, left-to-right along x.
    panel_rank = {disp: i for i, (_, disp) in enumerate(PANELS)}
    method_rank = {disp: i for i, disp in enumerate(METHOD_DISPLAY.values())}
    out["_p"] = out["metric"].map(panel_rank)
    out["_m"] = out["method"].map(method_rank)
    out = out.sort_values(["_p", "_m", "num_params"]).drop(columns=["_p", "_m"])
    out = out.reset_index(drop=True)

    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Write the stories_modules numbers to CSV: one row per bar
(method, retained, data_label) with mean compute ratio and 90% t-CI half-width,
matching stories_modules.png / .tex. Non-elicited (Retain/Forget) only, the two
panes (filtering, grmoe), over the 5 retain configs. Two-stage aggregation:
mean within seed, then mean + t-interval across seeds.

Run: python -m analysis.stories.modules_summary -> analysis/stories/stories_modules_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "stories.csv"
OUT = ROOT / "stories_modules_summary.csv"
CI_LEVEL = 0.9

METHODS = [("filtering", "Filtering"), ("grmoe", "GRAM")]
RETAIN_ORDER = ["core", "a-deadline-or-time-limit+core", "alien-encounters+core",
                "bygone-eras+core", "core+cultural-traditions"]
ALL_LABELS = ["core", "a-deadline-or-time-limit", "alien-encounters",
              "bygone-eras", "cultural-traditions"]


def main():
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    # Retain/Forget figure: drop elicited rows.
    df = df[df["label_class"] != "elicited_forget"]
    df = df[df["method"].isin([m for m, _ in METHODS])]
    df = df[df["retained"].isin(RETAIN_ORDER)]

    sm = (df.groupby(["method", "retained", "data_label", "seed"])["compute_ratio"]
            .mean().reset_index())
    agg = (sm.groupby(["method", "retained", "data_label"])["compute_ratio"]
             .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    tcrit = np.where(agg["count"] > 1,
                     st.t.ppf(1 - (1 - CI_LEVEL) / 2, agg["count"] - 1), 0.0)
    agg["ci90"] = tcrit * sem

    disp = dict(METHODS)
    rmap = {r: i for i, r in enumerate(RETAIN_ORDER)}
    lmap = {l: i for i, l in enumerate(ALL_LABELS)}
    agg = agg[agg["data_label"].isin(ALL_LABELS)].copy()
    agg["_m"] = agg["method"].map({m: i for i, (m, _) in enumerate(METHODS)})
    agg["_r"] = agg["retained"].map(rmap)
    agg["_l"] = agg["data_label"].map(lmap)
    agg = agg.sort_values(["_m", "_r", "_l"])

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "method": disp[r["method"]],
            "retained": r["retained"],
            "data_label": r["data_label"],
            "compute_ratio": round(float(r["mean"]), 4),
            "ci90": round(float(r["ci90"]), 4),
            "ci_low": round(float(r["mean"] - r["ci90"]), 4),
            "ci_high": round(float(r["mean"] + r["ci90"]), 4),
            "n_seeds": int(r["count"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(out)} rows)\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

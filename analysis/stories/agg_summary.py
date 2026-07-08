#!/usr/bin/env python3
"""
Write the stories_agg numbers to CSV: per (method, label_class), the mean
compute ratio and the 90% t-CI half-width (CR +- ci90), matching stories_agg.png
/ .tex. Two-stage: mean within each seed, then mean + t-interval across seeds.

Run: python -m analysis.stories.agg_summary  ->  analysis/stories/stories_agg_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "stories.csv"
OUT = ROOT / "stories_agg_summary.csv"
CI_LEVEL = 0.9

METHODS = [("filtering", "Filtering"), ("grmoe", "GRAM"), ("lora", "FT-LoRA"), ("coreftaux", "FT-Full"),
           ("demix", "Demix"), ("maxent", "MaxEnt")]
CLASS_ORDER = ["core", "retain", "forget", "elicited_forget"]


def main():
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])

    seed_means = (df.groupby(["method", "label_class", "seed"])["compute_ratio"]
                    .mean().reset_index())
    agg = (seed_means.groupby(["method", "label_class"])["compute_ratio"]
                     .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    tcrit = np.where(agg["count"] > 1,
                     st.t.ppf(1 - (1 - CI_LEVEL) / 2, agg["count"] - 1), 0.0)
    agg["ci90"] = tcrit * sem

    disp = dict(METHODS)
    rows = []
    for m, mdisp in METHODS:
        for lc in CLASS_ORDER:
            sel = agg[(agg["method"] == m) & (agg["label_class"] == lc)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            rows.append({
                "method": mdisp,
                "label_class": lc,
                "compute_ratio": round(float(r["mean"]), 4),
                "ci90": round(float(r["ci90"]), 4),
                "ci_low": round(float(r["mean"] - r["ci90"]), 4),
                "ci_high": round(float(r["mean"] + r["ci90"]), 4),
                "n_seeds": int(r["count"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

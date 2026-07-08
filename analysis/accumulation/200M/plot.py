#!/usr/bin/env python3
"""
3x2 grid of grouped bar charts for the 200M accumulation compute ratios.

Rows  = method   (baseline / GRAM / FT-LoRA)
Cols  = acc_mode (uniform / heterogeneous)
Bars  = the five data_labels (Core / Bio / Lisp / Cyber / Nuclear), each
        colored by LABEL_COLOR. A reference line at CR = 1.0 is drawn in every
        panel; CR is normalized against the heterogeneous baseline, so the
        top-right panel (baseline, heterogeneous) is 1.0 by construction.

Reads analysis/accumulation/200M/accumulation_200M.csv (single seed_5).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "accumulation.csv"
OUT = ROOT / "accumulation.png"

LABELS = ["core", "papers-biology", "code-lisp", "papers-cyber", "papers-nuclear"]
LABEL_DISPLAY = {"core": "Core", "papers-biology": "Bio", "code-lisp": "Lisp",
                 "papers-cyber": "Cyber", "papers-nuclear": "Nuclear"}
LABEL_COLOR = {"core": "#1f77b4", "papers-biology": "#2ca02c",
               "code-lisp": "#9467bd", "papers-cyber": "#d62728",
               "papers-nuclear": "#ff7f0e"}

METHOD_ORDER = ["baseline", "grmoe", "lora"]
METHOD_DISPLAY = {"baseline": "Baseline", "grmoe": "GRAM", "lora": "FT-LoRA"}
MODE_ORDER = ["uniform", "heterogeneous"]
MODE_DISPLAY = {"uniform": "Uniform", "heterogeneous": "Heterogeneous"}

# Per panel/bar we show the compute ratio for a domain *while it is retained*:
# core from the core-only config, each auxiliary from its own retain config.
# (The CSV also contains the forgotten-domain evals, which this figure omits.)
REQUIRED_RETAIN = {
    "core":           "core",
    "code-lisp":      "code-lisp+core",
    "papers-biology": "core+papers-biology",
    "papers-cyber":   "core+papers-cyber",
    "papers-nuclear": "core+papers-nuclear",
}
# The baseline trains on everything, so it has a single all-retained config.
BASELINE_RETAIN = "code-lisp+core+papers-biology+papers-cyber+papers-nuclear"


def main() -> None:
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])

    def want(method, label):
        return BASELINE_RETAIN if method == "baseline" else REQUIRED_RETAIN[label]

    lut = {(r.method, r.acc_mode, r.data_label): r.compute_ratio
           for r in df.itertuples()
           if r.retained == want(r.method, r.data_label)}
    ymax = max(1.1, df["compute_ratio"].max() * 1.12)

    nrow, ncol = len(METHOD_ORDER), len(MODE_ORDER)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 6.0),
                             sharex=True, sharey=True)

    x = list(range(len(LABELS)))
    for ri, method in enumerate(METHOD_ORDER):
        for ci, mode in enumerate(MODE_ORDER):
            ax = axes[ri][ci]
            vals = [lut.get((method, mode, lab), float("nan")) for lab in LABELS]
            colors = [LABEL_COLOR[lab] for lab in LABELS]
            ax.bar(x, vals, color=colors, width=0.74, zorder=2)
            ax.axhline(1.0, color="#808080", alpha=0.7, linestyle="-",
                       linewidth=0.8, zorder=1)
            for xi, v in zip(x, vals):
                if v == v:  # not NaN
                    ax.text(xi, v + 0.02 * ymax, f"{v:.2f}", ha="center",
                            va="bottom", fontsize=7.5)
            ax.set_ylim(0, ymax)
            ax.set_xticks(x)
            ax.set_xticklabels([LABEL_DISPLAY[lab] for lab in LABELS], fontsize=8)
            ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
            ax.tick_params(axis="y", labelsize=8)
            # Column headers (top row) and row labels (left col).
            if ri == 0:
                ax.set_title(MODE_DISPLAY[mode], fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{METHOD_DISPLAY[method]}\nCompute Ratio",
                              fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUT, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()

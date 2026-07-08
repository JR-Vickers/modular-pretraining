#!/usr/bin/env python3
"""
SimpleStories aggregate CR bar plot (mirrors analysis/realistic/plot.py
default variant): one bar group per method (filtering / grmoe), four bars per
group (Core / Retain / Forget / Elicit). Aggregates within each seed, then
mean + t-CI across the three seeds, averaging over all retain sets.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common.plot import grouped_bar_chart

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "stories.csv"
OUT = ROOT / "stories_agg.png"

METHOD_ORDER = ["filtering", "grmoe", "lora", "coreftaux", "demix", "maxent"]
METHOD_DISPLAY = {"filtering": "Filtering", "grmoe": "GRAM",
                  "lora": "FT-LoRA", "coreftaux": "FT-Full",
                  "demix": "Demix", "maxent": "MaxEnt"}

CLASS_ORDER = ["core", "retain", "forget", "elicited_forget"]
CLASS_DISPLAY = {"core": "Core ↑", "retain": "Retain ↑",
                 "forget": "Forget ↓", "elicited_forget": "Elicit ↓"}
CLASS_COLOR = {"core": "#1f77b4", "retain": "#2ca02c",
               "forget": "#d62728", "elicited_forget": "#ff7f0e"}


def main() -> None:
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    df = df[df["method"].isin(METHOD_ORDER)]

    ax, _ = grouped_bar_chart(
        df, x_col="method", y_col="compute_ratio", group_col="label_class",
        seed_col="seed", ci_level=0.9,
        x_order=METHOD_ORDER, group_order=CLASS_ORDER,
        x_labels=METHOD_DISPLAY, group_labels=CLASS_DISPLAY,
        colors=CLASS_COLOR,
        title=None,
        x_axis_label="", y_axis_label="Compute Ratio",
        figsize=(9.0, 2.1), fontsize=11, y_min=0, y_max=1.1,
        error_bars=True, show_values=True,
    )
    ax.set_yticks([0.0, 0.5, 1.0])
    # Match realistic.tex shades: 0.5 = darkgray176 @ 0.5, 1.0 = gray @ 0.6.
    ax.axhline(0.5, color="#b0b0b0", alpha=0.5, linestyle="-", linewidth=0.6, zorder=0)
    ax.axhline(1.0, color="#808080", alpha=0.6, linestyle="-", linewidth=0.8, zorder=0.5)
    # Opaque white highlight behind bar-value labels.
    for txt in ax.texts:
        txt.set_bbox(dict(facecolor="white", edgecolor="none",
                          alpha=1.0, pad=1.0))
    # Single horizontal legend above the chart.
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncols=len(labels),
              frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()

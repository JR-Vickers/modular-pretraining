#!/usr/bin/env python3
"""
arch/100M grouped bar chart: 4 method configs × 4 label classes.

Reads arch.csv (produced by compile.py); aggregates mean-within-seed,
t-CI across the 3 seeds. Bars: Core / Retain / Forget / Elicit.

CSV stores raw config names (GR-MoE / GR-LoRA / FT-MoE / FT-LoRA); GR-MoE
is displayed as GRAM to match the project-wide rename.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common.plot import grouped_bar_chart

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "arch.csv"
OUT = ROOT / "arch.png"

METHOD_ORDER = ["GR-MoE", "GR-LoRA", "FT-MoE", "FT-LoRA"]
METHOD_DISPLAY = {"GR-MoE": "GRAM (MLP)", "GR-LoRA": "GRAM (LoRA)",
                  "FT-MoE": "FT-MLP", "FT-LoRA": "FT-LoRA"}
CLASS_ORDER = ["core", "retain", "forget", "elicited_forget"]
CLASS_DISPLAY = {"core": "Core ↑", "retain": "Retain ↑",
                 "forget": "Forget ↓", "elicited_forget": "Elicit ↓"}
CLASS_COLOR = {"core": "#1f77b4", "retain": "#2ca02c",
               "forget": "#d62728", "elicited_forget": "#ff7f0e"}


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df.method.isin(METHOD_ORDER) & df.compute_ratio.notna()].copy()

    print("=== mean CR by (method, label_class) ===")
    print(df.pivot_table(index="method", columns="label_class",
                         values="compute_ratio", aggfunc="mean")
            .reindex(METHOD_ORDER)[CLASS_ORDER].round(3).to_string())

    ax, _ = grouped_bar_chart(
        df, x_col="method", y_col="compute_ratio", group_col="label_class",
        seed_col="seed", ci_level=0.9,
        x_order=METHOD_ORDER, group_order=CLASS_ORDER,
        x_labels=METHOD_DISPLAY, group_labels=CLASS_DISPLAY,
        colors=CLASS_COLOR,
        title=None, x_axis_label="", y_axis_label="Compute Ratio",
        figsize=(8.5, 3.2), fontsize=11, y_min=0, y_max=1.25,
        error_bars=True, show_values=True,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    # White highlight behind the numeric bar labels for legibility.
    for txt in ax.texts:
        txt.set_bbox(dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.0))
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncols=len(labels), frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()

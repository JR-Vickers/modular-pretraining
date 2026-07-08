#!/usr/bin/env python3
"""
Grouped bar plot of partial-labeling CR at 400M, across 3 seeds.

Four method-config groups:
  - Core-Only Filter (Perfect):   filtering, label_prc = 1.0
  - Core-Only Filter (Partial):   filtering, label_prc = 0.5
  - GRAM:                       routed/moe with aux_route_prc = 0.0
  - FT-LoRA:                      routed/lora

Each group has four label-class bars (Core / Retain / Forget / Elicited).
Aggregates first within each seed, then takes mean + t-CI across seeds.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common.plot import grouped_bar_chart

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "partial.csv"
OUT = ROOT / "partial_simple.png"

MODEL_SIZE = "400M"

GROUP_ORDER = ["grmoe", "filter_partial", "lora"]
GROUP_DISPLAY = {
    "grmoe": "GRAM",
    "filter_partial": "Filtering",
    "lora": "Filter + LoRA",
}

CLASS_ORDER = ["core", "retain", "forget", "elicited_forget"]
CLASS_DISPLAY = {"core": "Core ↑", "retain": "Retain ↑",
                 "forget": "Forget ↓", "elicited_forget": "Elicit ↓"}
CLASS_COLOR = {"core": "#1f77b4", "retain": "#2ca02c",
               "forget": "#d62728", "elicited_forget": "#ff7f0e"}


def assign_group(row: pd.Series) -> str | None:
    if row["method"] == "filtering":
        if row["label_prc"] == 1.0:
            return "filter_perfect"
        if row["label_prc"] == 0.5:
            return "filter_partial"
        return None
    if row["method"] == "routed" and row["arch"] == "moe":
        if row["aux_route_prc"] == 0.0:
            return "grmoe"
        return None
    if row["method"] == "routed" and row["arch"] == "lora":
        return "lora"
    return None


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df["model_size"] == MODEL_SIZE].copy()
    df = df[df["retained"] == "core"]
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df["label_prc"] = pd.to_numeric(df["label_prc"], errors="coerce")
    df["aux_route_prc"] = pd.to_numeric(df["aux_route_prc"], errors="coerce")
    df["group"] = df.apply(assign_group, axis=1)
    df = df[df["group"].isin(GROUP_ORDER)]
    df = df.dropna(subset=["compute_ratio"])

    print(f"=== rows by (group, label_class) at {MODEL_SIZE} ===")
    print(df.groupby(["group", "label_class"]).size().to_string())

    ax, _ = grouped_bar_chart(
        df, x_col="group", y_col="compute_ratio", group_col="label_class",
        seed_col="seed", ci_level=0.9,
        x_order=GROUP_ORDER, group_order=CLASS_ORDER,
        x_labels=GROUP_DISPLAY, group_labels=CLASS_DISPLAY,
        colors=CLASS_COLOR,
        title=None,
        x_axis_label="", y_axis_label="Compute Ratio",
        figsize=(5.589, 2.5875), fontsize=11, y_min=0.0, y_max=1.15,
        error_bars=True, show_values=True,
    )
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    for txt in ax.texts:
        txt.set_bbox(dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.0))
    # Move legend to a single horizontal row above the chart.
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncols=len(labels),
              frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()

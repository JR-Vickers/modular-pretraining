#!/usr/bin/env python3
"""
5B CR bar plot for retained=core+papers-biology, one PNG per accumulation
mode (het / uni). Both compare against the same baseline (base/5B/seed_1/run_1),
fit in compile.py.

3 method groups (GRMoE / LoRA / Filtering), 5 bars per group — one per
data_label: core, papers-biology, code-lisp, papers-cyber, papers-nuclear.
The 3 forget data_labels are alpha-faded via fade_map, using
``analysis.common.plot.grouped_bar_chart``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common.plot import grouped_bar_chart

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "accumulation.csv"
RETAIN_FILTER = "core+papers-biology"

METHOD_ORDER = ["grmoe", "lora", "filtering"]
METHOD_DISPLAY = {"grmoe": "GR-MoE (Aux 3×)", "lora": "FT-LoRA",
                  "filtering": "Filtering"}

DATA_LABEL_ORDER = ["core", "papers-biology",
                    "code-lisp", "papers-cyber", "papers-nuclear"]
DATA_LABEL_DISPLAY = {
    "core": "Core", "papers-biology": "Biology", "code-lisp": "Lisp",
    "papers-cyber": "Cyber", "papers-nuclear": "Nuclear",
}
FORGET_DATA_LABELS = ["code-lisp", "papers-cyber", "papers-nuclear"]

# Per-accumulation panel config: (output PNG, title, y_max).
PANELS = {
    "het": ("scaling_cr_5B_corebio_hetacc.png",
            "5B - Heterogenous Acc Methods, Heterogeneous Acc Base", 1.6),
    "uni": ("scaling_cr_5B_corebio_uniacc.png",
            "5B - Uniform Acc Methods, Heterogenous Acc Baseline", 1.1),
}


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep core/core, papers-biology/retain, and forget rows for forget labels."""
    keep = (
        ((df["data_label"] == "core") & (df["label_class"] == "core"))
        | ((df["data_label"] == "papers-biology") & (df["label_class"] == "retain"))
        | (df["data_label"].isin(FORGET_DATA_LABELS) & (df["label_class"] == "forget"))
    )
    return df[keep]


def render_panel(df: pd.DataFrame, accumulation: str) -> None:
    out_name, title, y_max = PANELS[accumulation]
    sub = df[df["accumulation"] == accumulation]
    sub = filter_rows(sub)

    pivot = sub.pivot_table(
        index="method", columns="data_label",
        values="compute_ratio", aggfunc="mean",
    ).reindex(METHOD_ORDER)[DATA_LABEL_ORDER]
    print(f"=== accumulation = {accumulation} ===")
    print(pivot.round(3).to_string())

    ax, _ = grouped_bar_chart(
        sub, x_col="method", y_col="compute_ratio", group_col="data_label",
        x_order=METHOD_ORDER, group_order=DATA_LABEL_ORDER,
        x_labels=METHOD_DISPLAY, group_labels=DATA_LABEL_DISPLAY,
        fade_map={m: FORGET_DATA_LABELS for m in METHOD_ORDER},
        title=title, x_axis_label="", y_axis_label="Compute Ratio",
        figsize=(10, 5.5), fontsize=11, y_min=0, y_max=y_max,
        error_bars=False, show_values=True,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    out_path = ROOT / out_name
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}\n")


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df["retained"] == RETAIN_FILTER]
    for accumulation in PANELS:
        render_panel(df, accumulation)


if __name__ == "__main__":
    main()

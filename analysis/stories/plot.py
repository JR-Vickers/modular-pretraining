#!/usr/bin/env python3
"""
SimpleStories modules figure (mirrors analysis/realistic/plot.py --variant
modules): two panes, top = Data Filtering (many models), bottom = GRAM (one
model). x = retain/module config, 5 data-label bars each; non-retained labels
are faded + slash-hatched (forgotten). Aggregates mean + t-CI across seeds.
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
OUT = ROOT / "stories_modules.png"

ALL_LABELS = ["core", "a-deadline-or-time-limit", "alien-encounters",
              "bygone-eras", "cultural-traditions"]
DATA_LABEL_DISPLAY = {
    "core": "Core", "a-deadline-or-time-limit": "Deadlines",
    "alien-encounters": "Aliens", "bygone-eras": "Eras",
    "cultural-traditions": "Cultures",
}
DATA_LABEL_COLOR = {
    "core": "#1f77b4", "a-deadline-or-time-limit": "#ff7f0e",
    "alien-encounters": "#2ca02c", "bygone-eras": "#d62728",
    "cultural-traditions": "#9467bd",
}
# Retain sets in incremental order; CSV stores `retained` as sorted "+"-join.
RETAIN_ORDER = ["core", "a-deadline-or-time-limit+core", "alien-encounters+core",
                "bygone-eras+core", "core+cultural-traditions"]
RETAIN_ADD_LABEL = {  # data_label added at each step
    "core": None, "a-deadline-or-time-limit+core": "a-deadline-or-time-limit",
    "alien-encounters+core": "alien-encounters",
    "bygone-eras+core": "bygone-eras",
    "core+cultural-traditions": "cultural-traditions",
}


def _xlabels(noun: str) -> dict[str, str]:
    out = {}
    for key, add in RETAIN_ADD_LABEL.items():
        out[key] = (f"Core {noun} Only" if add is None
                    else f"+ {DATA_LABEL_DISPLAY[add]} {noun}")
    return out


def _draw_pane(df: pd.DataFrame, ax: plt.Axes, method: str,
               title: str, noun: str) -> None:
    pane = df[df["method"] == method]
    fade_map = {key: [lab for lab in ALL_LABELS if lab not in key.split("+")]
                for key in RETAIN_ORDER}
    grouped_bar_chart(
        pane, x_col="retained", y_col="compute_ratio", group_col="data_label",
        seed_col="seed", ci_level=0.9,
        x_order=RETAIN_ORDER, group_order=ALL_LABELS,
        x_labels=_xlabels(noun), group_labels=DATA_LABEL_DISPLAY,
        fade_map=fade_map, colors=DATA_LABEL_COLOR,
        title=title, x_axis_label="", y_axis_label="Compute\nRatio",
        fontsize=11, y_min=0.5, y_max=1.1,
        error_bars=True, show_values=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.5,
                  color="#dddddd")
    ax.set_axisbelow(True)
    # Darker reference gridline at 1.0.
    ax.axhline(1.0, color="#808080", linestyle="-", linewidth=0.7, zorder=0.5)
    for patch in ax.patches:
        a = patch.get_alpha()
        if a is not None and a < 1.0:
            fc = patch.get_facecolor()
            patch.set_facecolor(tuple(c * a + (1 - a) for c in fc[:3]) + (1.0,))
            patch.set_alpha(1.0)
            patch.set_hatch("//")
            patch.set_edgecolor("#888888")
            patch.set_linewidth(0.0)
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def _add_legend(fig: plt.Figure, cx: float) -> None:
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=DATA_LABEL_COLOR[d], edgecolor="none",
                     label=DATA_LABEL_DISPLAY[d]) for d in ALL_LABELS]
    handles += [
        Patch(facecolor="#666666", edgecolor="none", label="Retain ↑"),
        Patch(facecolor="#dddddd", edgecolor="#888888", hatch="//",
              linewidth=0, label="Forget ↓"),
    ]
    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", bbox_to_anchor=(cx, 1.0),
               ncols=len(handles), frameon=True, fontsize=11,
               columnspacing=1.6, handletextpad=0.5)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, default=CSV)
    ap.add_argument("-o", "--output", type=Path, default=OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    # Retain/Forget figure: drop the elicited rows.
    df = df[df["label_class"] != "elicited_forget"]
    df = df[df["retained"].isin(RETAIN_ORDER)]

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 2.76))
    _draw_pane(df, axes[0], "filtering", "Data Filtering (5 Models)", "Data")
    _draw_pane(df, axes[1], "grmoe", "GRAM (Ours, 1 Model)", "Module")
    plt.tight_layout()
    fig.canvas.draw()
    bb = axes[0].get_position()
    cx = (bb.x0 + bb.x1) / 2
    _add_legend(fig, cx)
    plt.savefig(args.output, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

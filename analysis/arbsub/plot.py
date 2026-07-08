#!/usr/bin/env python3
"""
arbsub CR bar plot -> arbsub.png.

Two stacked panels, one per retain config:
  - "Virology Module Active (Seen in Training)"   (retain = core + virology)
  - "All Modules Active (Not Seen in Training)"    (retain = all modules)
Each panel: x = method (GRAM / FT-LoRA), 5 data_label bars each, with 90%
CI error bars. data_labels NOT in the retain set are pale + slash-hatched to
mark them as being forgotten.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common.plot import grouped_bar_chart

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "arbsub.csv"

ALL_LABELS = ("core", "code-lisp", "papers-biology",
              "papers-cyber", "papers-nuclear")

DATA_LABEL_ORDER = list(ALL_LABELS)
DATA_LABEL_DISPLAY = {
    "core": "Core", "code-lisp": "Code",
    "papers-biology": "Virology", "papers-cyber": "Cyber",
    "papers-nuclear": "Nuclear",
}
# tab10 colors in DATA_LABEL_ORDER so legend swatches match the bars.
DATA_LABEL_LEGEND_COLOR = {
    "core": "#1f77b4", "code-lisp": "#ff7f0e",
    "papers-biology": "#2ca02c", "papers-cyber": "#d62728",
    "papers-nuclear": "#9467bd",
}
METHOD_XLABEL = {"grmoe": "GRAM", "lora": "Filter + LoRA"}

# (retained CSV value, panel title).
PANES = [
    ("core+papers-biology",
     "Virology Module Active (Seen in Training)"),
    ("code-lisp+core+papers-biology+papers-cyber+papers-nuclear",
     "All Modules Active (Not Seen in Training)"),
]


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    return df.dropna(subset=["compute_ratio"])


def _draw_pane(df: pd.DataFrame, ax: plt.Axes, retained: str, title: str) -> None:
    """One panel: x = method (GRAM / FT-LoRA), 5 data_label bars each,
    non-retained data_labels faded + hatched."""
    forget_labels = [lab for lab in ALL_LABELS if lab not in retained.split("+")]
    fade_map = {"grmoe": forget_labels, "lora": forget_labels}

    grouped_bar_chart(
        df, x_col="method", y_col="compute_ratio", group_col="data_label",
        seed_col="seed", ci_level=0.9,
        x_order=["grmoe", "lora"], group_order=DATA_LABEL_ORDER,
        x_labels=METHOD_XLABEL, group_labels=DATA_LABEL_DISPLAY,
        fade_map=fade_map,
        title=title, x_axis_label="", y_axis_label="Compute Ratio",
        fontsize=11, y_min=0, y_max=1.15,
        error_bars=True, show_values=False, ax=ax,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.7,
                  color="#b0b0b0", alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color="#808080", linestyle="-", linewidth=0.7, alpha=0.6)
    for patch in ax.patches:
        a = patch.get_alpha()
        if a is not None and a < 1.0:
            # Pale fill at full opacity + crisp dark-grey hatch (instead of a
            # faded white hatch on a translucent bar).
            fc = patch.get_facecolor()
            patch.set_facecolor(tuple(c * a + (1 - a) for c in fc[:3]) + (1.0,))
            patch.set_alpha(1.0)
            patch.set_hatch("//")
            patch.set_edgecolor("#666666")
            patch.set_linewidth(0.0)
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def _add_top_color_legend(fig: plt.Figure, cx: float) -> None:
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=DATA_LABEL_LEGEND_COLOR[d], edgecolor="none",
              label=DATA_LABEL_DISPLAY[d])
        for d in DATA_LABEL_ORDER
    ]
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center", bbox_to_anchor=(cx, 1.02),
        ncols=len(handles), frameon=True, fontsize=11,
        columnspacing=2.6, handletextpad=0.5,
    )


def main() -> None:
    df = _load_df()
    fig, axes = plt.subplots(2, 1, figsize=(6.34, 4.1), sharex=True)
    for ax, (retained, title) in zip(axes, PANES):
        _draw_pane(df[df["retained"] == retained], ax, retained, title)
        ax.set_box_aspect(1 / 3.3)  # ~3.3x wider than tall
        # sharex hides the top pane's x labels; repeat GRAM/FT-LoRA there too.
        ax.tick_params(axis="x", labelbottom=True)
    plt.tight_layout()
    fig.canvas.draw()
    bb = axes[0].get_position()
    _add_top_color_legend(fig, (bb.x0 + bb.x1) / 2)

    out_path = ROOT / "arbsub.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Modules-style CR plot for the realistic 800M MaxEnt BO study -> maxent_realistic.png.

Analogous to analysis/realistic/realistic_modules.png: two stacked panels,
x = retain set (which module/data is being retained), 5 data-label bars each,
non-retained data_labels pale + slash-hatched (being forgotten).

  - Top panel:    Data Filtering   (from analysis/realistic/realistic.csv)
  - Bottom panel: MaxEnt           (from this dir's maxent_realistic.csv)

Both methods' CRs are normalized against the same 800M seed_1 baseline
(20260422084509038162), so the panels are directly comparable.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parents[2]          # .../analysis
EXPERIMENT_ROOT = SCRIPT_DIR.parents[3]       # .../arXiv-Codebase
sys.path.insert(0, str(EXPERIMENT_ROOT))

from analysis.common.plot import grouped_bar_chart   # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

MAXENT_CSV = SCRIPT_DIR / "maxent_realistic.csv"
FILTERING_CSV = ANALYSIS_DIR / "realistic" / "realistic.csv"
OUT = SCRIPT_DIR / "maxent_realistic.png"

ALL_LABELS = ["core", "code-lisp", "papers-biology",
              "papers-cyber", "papers-nuclear"]
DATA_LABEL_DISPLAY = {
    "core": "Core", "code-lisp": "Code",
    "papers-biology": "Virology", "papers-cyber": "Cyber",
    "papers-nuclear": "Nuclear",
}
DATA_LABEL_COLOR = {
    "core": "#1f77b4", "code-lisp": "#ff7f0e",
    "papers-biology": "#2ca02c", "papers-cyber": "#d62728",
    "papers-nuclear": "#9467bd",
}
# Retain sets (CSV stores `retained` as sorted "+"-join), incremental order.
RETAIN_ORDER = ["core", "code-lisp+core", "core+papers-biology",
                "core+papers-cyber", "core+papers-nuclear"]
RETAIN_ADD_LABEL = {  # data_label added at each step
    "core": None, "code-lisp+core": "code-lisp",
    "core+papers-biology": "papers-biology",
    "core+papers-cyber": "papers-cyber",
    "core+papers-nuclear": "papers-nuclear",
}


def _xlabels(noun: str) -> dict[str, str]:
    out = {}
    for key, add in RETAIN_ADD_LABEL.items():
        out[key] = f"Core {noun} Only" if add is None \
            else f"+ {DATA_LABEL_DISPLAY[add]} {noun}"
    return out


def _load(csv_path: Path, method: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if method is not None:
        df = df[df["method"] == method]
    df = df.copy()
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    return df[df["retained"].isin(RETAIN_ORDER)]


def _draw_pane(df: pd.DataFrame, ax: plt.Axes, title: str, noun: str) -> None:
    """One pane: x = retain set, 5 data-label bars; non-retained labels are
    faded + slash-hatched (forgotten)."""
    fade_map = {
        key: [lab for lab in ALL_LABELS if lab not in key.split("+")]
        for key in RETAIN_ORDER
    }
    grouped_bar_chart(
        df, x_col="retained", y_col="compute_ratio", group_col="data_label",
        seed_col="seed", ci_level=0.9,
        x_order=RETAIN_ORDER, group_order=ALL_LABELS,
        x_labels=_xlabels(noun), group_labels=DATA_LABEL_DISPLAY,
        fade_map=fade_map, colors=DATA_LABEL_COLOR,
        title=title, x_axis_label="", y_axis_label="Compute\nRatio",
        fontsize=11, y_min=0.0, y_max=1.1,
        error_bars=True, show_values=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.5,
                  color="#dddddd")
    ax.set_axisbelow(True)
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
    handles = [
        Patch(facecolor=DATA_LABEL_COLOR[d], edgecolor="none",
              label=DATA_LABEL_DISPLAY[d])
        for d in ALL_LABELS
    ]
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
    filt = _load(FILTERING_CSV, method="filtering")
    mx = _load(MAXENT_CSV, method=None)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 3.5))
    _draw_pane(filt, axes[0], "Data Filtering (Many Models)", noun="Data")
    _draw_pane(mx, axes[1], "MaxEnt", noun="Retained")
    plt.tight_layout()
    fig.canvas.draw()
    bb = axes[0].get_position()
    _add_legend(fig, (bb.x0 + bb.x1) / 2)
    plt.savefig(OUT, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()

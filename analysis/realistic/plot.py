#!/usr/bin/env python3
"""
800M realistic-setting CR bar plot.

One bar group per method (base / coreftaux / filtering / grmoe / lora),
four bars per group (Core / Retain / Forget / Elicit). Aggregates first
within each seed, then takes mean + t-CI across the three seeds.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common.plot import grouped_bar_chart

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "realistic.csv"

VARIANTS = {
    "default": {
        "out": ROOT / "realistic_agg_full.png",
        "methods": ["filtering", "grmoe", "lora", "coreftaux", "maxent"],
        "figsize": (8.8, 2.2),
        "display_overrides": {"grmoe": "GRAM (ours)"},
    },
    "alt": {
        "out": ROOT / "realistic_agg_simple.png",
        "methods": ["filtering", "grmoe", "lora", "maxent"],
        "figsize": (6.9696, 2.2),
        "display_overrides": {"lora": "Filter + LoRA", "grmoe": "GRAM (ours)"},
        "legend_fontsize": 9,
    },
    "adversarial": {
        "out": ROOT / "realistic_adversarial.png",
        "methods": ["grmoe", "maxent"],
        "classes": ["core", "forget", "elicited_forget"],
        "figsize": (5.313, 2.52),
        "yticks": [0.0, 0.5, 1.0],
    },
}

METHOD_ORDER = ["filtering", "grmoe", "lora", "coreftaux", "maxent"]
METHOD_DISPLAY = {
    "filtering": "Filtering", "grmoe": "GRAM",
    "lora": "FT-LoRA", "coreftaux": "FT-Full",
    "maxent": "MaxEnt",
}

CLASS_ORDER = ["core", "retain", "forget", "elicited_forget"]
CLASS_DISPLAY = {"core": "Core ↑", "retain": "Retain ↑",
                 "forget": "Forget ↓", "elicited_forget": "Elicit ↓"}
CLASS_COLOR = {"core": "#1f77b4", "retain": "#2ca02c",
               "forget": "#d62728", "elicited_forget": "#ff7f0e"}

# --- modules layout (arbsub-style: retain-set groups, data-label bars) -------
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
# Retain sets in incremental order; CSV stores `retained` as sorted "+"-join.
RETAIN_ORDER = ["core", "core+papers-biology", "core+papers-nuclear"]
RETAIN_ADD_LABEL = {  # the data_label being added at each step
    "core": None, "code-lisp+core": "code-lisp",
    "core+papers-biology": "papers-biology",
    "core+papers-cyber": "papers-cyber",
    "core+papers-nuclear": "papers-nuclear",
}


def _modules_xlabels(noun: str) -> dict[str, str]:
    """Per-pane x labels, e.g. 'Core Data Only' / '+ Biology Data'."""
    out = {}
    for key, add in RETAIN_ADD_LABEL.items():
        if add is None:
            out[key] = f"Core {noun} Only"
        else:
            out[key] = f"+ {DATA_LABEL_DISPLAY[add]} {noun}"
    return out


def _draw_modules_pane(df: pd.DataFrame, ax: plt.Axes, method: str,
                       title: str, noun: str) -> None:
    """One pane: x = retain set, 5 data-label bars; non-retained labels are
    faded + slash-hatched (forgotten)."""
    pane = df[df["method"] == method]
    fade_map = {
        key: [lab for lab in ALL_LABELS if lab not in key.split("+")]
        for key in RETAIN_ORDER
    }
    grouped_bar_chart(
        pane, x_col="retained", y_col="compute_ratio", group_col="data_label",
        seed_col="seed", ci_level=0.9,
        x_order=RETAIN_ORDER, group_order=ALL_LABELS,
        x_labels=_modules_xlabels(noun), group_labels=DATA_LABEL_DISPLAY,
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


def _add_modules_legend(fig: plt.Figure, cx: float) -> None:
    """One row: data-label colors + Retain/Forget convention swatches."""
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


def plot_modules() -> None:
    out_path = ROOT / "realistic_modules.png"
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    df = df[df["retained"].isin(RETAIN_ORDER)]

    fig, axes = plt.subplots(2, 1, figsize=(8.25, 3.312))
    _draw_modules_pane(df, axes[0], "filtering",
                       "Data Filtering (Many Models)", noun="Data")
    _draw_modules_pane(df, axes[1], "grmoe",
                       "GRAM (Ours, 1 Model)", noun="Module")
    plt.tight_layout()
    fig.canvas.draw()
    bb = axes[0].get_position()
    cx = (bb.x0 + bb.x1) / 2
    _add_modules_legend(fig, cx)
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",
                        choices=(*VARIANTS, "modules"), default="default")
    args = parser.parse_args()
    if args.variant == "modules":
        plot_modules()
        return
    cfg = VARIANTS[args.variant]
    methods = cfg["methods"]
    out_path = cfg["out"]
    method_display = {**METHOD_DISPLAY, **cfg.get("display_overrides", {})}

    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    df = df[df["method"].isin(methods)]
    class_order = cfg.get("classes", CLASS_ORDER)
    df = df[df["label_class"].isin(class_order)]

    print(f"=== variant: {args.variant} -> {out_path.name} ===")
    print("rows by (method, label_class):")
    print(df.groupby(["method", "label_class"]).size().to_string())

    ax, _ = grouped_bar_chart(
        df, x_col="method", y_col="compute_ratio", group_col="label_class",
        seed_col="seed", ci_level=0.9,
        x_order=methods, group_order=class_order,
        x_labels=method_display, group_labels=CLASS_DISPLAY,
        colors=CLASS_COLOR,
        title=None,
        x_axis_label="", y_axis_label="Compute Ratio",
        figsize=cfg["figsize"], fontsize=11, y_min=0, y_max=1.25,
        error_bars=True, show_values=True,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    if cfg.get("yticks") is not None:
        ax.set_yticks(cfg["yticks"])
    # White highlight behind bar-value labels for legibility (opaque).
    for txt in ax.texts:
        txt.set_bbox(dict(facecolor="white", edgecolor="none",
                          alpha=1.0, pad=1.0))
    # Move legend to a single horizontal row above the chart.
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncols=len(labels),
              frameon=True, fontsize=cfg.get("legend_fontsize", 11))
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()

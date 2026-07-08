#!/usr/bin/env python3
"""
Grouped bar plot of partial-labeling CR at 400M, across 3 seeds.

Emits two figures:
  - partial.tex / .png       3 groups (GRAM, Filtering, FT-LoRA, all 50%
                             labeled) -- the main-body figure.
  - partial_full.tex / .png  4 groups, additionally including perfectly
                             labeled Filtering (100%) as a reference -- the
                             expanded appendix figure.

Each group has three label-class bars (Core / Forget / Elicit). Aggregates
first within each seed, then takes mean + t-CI across seeds.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common.plot import grouped_bar_chart

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "partial.csv"

MODEL_SIZE = "400M"

# 3-group main-body figure (all methods at 50% labeling).
GROUPS_3 = ["grmoe", "filter_partial", "lora"]
DISPLAY_3 = {"grmoe": "GRAM", "filter_partial": "Filtering", "lora": "FT-LoRA"}

# 4-group appendix figure, adding perfectly labeled Filtering as a reference.
GROUPS_4 = ["filter_perfect", "grmoe", "filter_partial", "lora"]
DISPLAY_4 = {"filter_perfect": "Filtering (100%)", "grmoe": "GRAM (50%)",
             "filter_partial": "Filtering (50%)", "lora": "FT-LoRA (50%)"}

CLASS_ORDER = ["core", "forget", "elicited_forget"]
CLASS_DISPLAY = {"core": "Core ↑", "forget": "Forget ↓",
                 "elicited_forget": "Elicit ↓"}
CLASS_COLOR = {"core": "#1f77b4", "forget": "#d62728",
               "elicited_forget": "#ff7f0e"}


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


def style_tikz(code: str) -> str:
    """Match the styling of the paper's other bar figure (arbsub.tex)."""
    code = code.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
    # matplot2tikz emits a spurious \addlegendentry after every bar \addplot
    # (all mislabeled); keep only the ones immediately after \addlegendimage.
    kept: list[str] = []
    for ln in code.split("\n"):
        if ln.lstrip().startswith(r"\addlegendentry"):
            prev = next((x for x in reversed(kept) if x.strip()), "")
            if not prev.lstrip().startswith(r"\addlegendimage"):
                continue
        kept.append(ln)
    code = "\n".join(kept)
    # Proper rectangle legend swatches instead of the default ybar double-bars.
    code = code.replace(
        r"\begin{tikzpicture}",
        "\\begin{tikzpicture}\n\n\\pgfplotsset{legend image code/.code={"
        "\\draw[#1] (0cm,-0.06cm) rectangle (0.25cm,0.06cm);}}", 1)
    code = code.replace("ybar,ybar legend,draw=none", "draw=none")
    # Consistent fonts + scale only axis (declared height is the plot box).
    code = code.replace(
        "\\begin{axis}[\n",
        "\\begin{axis}[\nscale only axis=true,\n"
        "tick label style={font=\\footnotesize},\n"
        "label style={font=\\footnotesize},\n", 1)
    code = code.replace(
        "legend style={\n",
        "legend style={\n  font=\\scriptsize,\n"
        "  /tikz/every even column/.append style={column sep=8pt},\n", 1)
    code = code.replace("at={(0.5,1.02)}", "at={(0.5,1.16)}")
    # Bar value labels: matplot2tikz shrinks them to scale=0.44; use \scriptsize.
    code = code.replace("scale=0.44,", r"font=\scriptsize,")
    # Force explicit y-ticks (matplot2tikz drops them).
    code = code.replace(
        "ytick style={color=black}",
        "ytick={0.4,0.6,0.8,1},\nyticklabels={0.4,0.6,0.8,1.0},\n"
        "ytick style={color=black}")
    # Escape any '%' inside the x-tick labels (e.g. "Filtering (50%)") so LaTeX
    # does not treat it as a comment; leave matplot2tikz's own "table {%" alone.
    m = re.search(r"xticklabels=\{[^}]*\}", code)
    if m:
        code = code.replace(m.group(0), re.sub(r"(?<!\\)%", r"\\%", m.group(0)))
    return code


def render(df: pd.DataFrame, group_order: list[str], group_display: dict,
           out_png: Path, out_tex: Path, axis_width: str, axis_height: str) -> None:
    d = df[df["group"].isin(group_order)]
    ax, _ = grouped_bar_chart(
        d, x_col="group", y_col="compute_ratio", group_col="label_class",
        seed_col="seed", ci_level=0.9,
        x_order=group_order, group_order=CLASS_ORDER,
        x_labels=group_display, group_labels=CLASS_DISPLAY,
        colors=CLASS_COLOR, title=None,
        x_axis_label="", y_axis_label="Compute Ratio",
        figsize=(1.5 * len(group_order) + 1.0, 2.4),
        fontsize=11, y_min=0.4, y_max=1.15,
        error_bars=True, show_values=True,
    )
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    for txt in ax.texts:
        txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncols=len(labels), frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=260, bbox_inches="tight")

    import matplot2tikz
    code = matplot2tikz.get_tikz_code(
        figure=ax.figure, axis_width=axis_width, axis_height=axis_height)
    out_tex.write_text(style_tikz(code))
    plt.close()
    print(f"saved: {out_png}\nsaved: {out_tex}")


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df["model_size"] == MODEL_SIZE].copy()
    df = df[df["retained"] == "core"]
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df["label_prc"] = pd.to_numeric(df["label_prc"], errors="coerce")
    df["aux_route_prc"] = pd.to_numeric(df["aux_route_prc"], errors="coerce")
    df["group"] = df.apply(assign_group, axis=1)
    df = df[df["label_class"].isin(CLASS_ORDER)]
    df = df.dropna(subset=["compute_ratio"])

    # Main-body 3-group figure (matches one arbsub.tex pane).
    render(df, GROUPS_3, DISPLAY_3, ROOT / "partial.png", ROOT / "partial.tex",
           axis_width=r"0.76\linewidth", axis_height=r"0.234\linewidth")
    # Expanded appendix 4-group figure (adds perfectly labeled Filtering).
    render(df, GROUPS_4, DISPLAY_4, ROOT / "partial_full.png",
           ROOT / "partial_full.tex",
           axis_width=r"0.62\linewidth", axis_height=r"0.20\linewidth")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple Stories hyperparameter sweep -> stories.png.

One figure, three panes side by side, sharing the y-axis (Compute Ratio).
Each pane is one sweep with that knob on the x-axis and four lines for the
core / retain / forget / elicit capability classes:

    pane 1  robust_prc      (GRAM core robustness)
    pane 2  aux_route_prc   (GRAM aux spread)
    pane 3  core_aux_ratio  (FT-LoRA core:aux split, log x)

Reads stories.csv (from compile.py). Aggregation: mean within each seed, then
mean + t-interval CI across seeds, per sweep point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "stories.csv"
DEFAULT_PNG = ROOT / "stories.png"

CLASS_ORDER = ["core", "retain", "forget", "elicited_forget"]
CLASS_DISPLAY = {"core": "Core ↑", "retain": "Retain ↑",
                 "forget": "Forget ↓", "elicited_forget": "Elicit ↓"}
CLASS_COLOR = {"core": "#1f77b4", "retain": "#2ca02c",
               "forget": "#d62728", "elicited_forget": "#ff7f0e"}

# GRAM panels split the aggregate Core into two lines: Core-Partial is the mean
# over the retain configs that keep core plus a single auxiliary module, and
# Core-Full is the core-only config (all auxiliary modules ablated).
CORE_PARTIAL_COLOR = "#bcbd22"   # tableau yellow (tab10)
CORE_PARTIAL_LABEL = "Core-Partial ↑"
CORE_FULL_COLOR = "#9467bd"      # tab10 purple
CORE_FULL_LABEL = "Core-Full ↑"

# One pane per sweep, in display order. x_scale in {None, "log"};
# x_categorical places sweep points at evenly-spaced positions (used for the
# robust panel so 0 / 0.1 / 0.2 / 0.4 / 0.6 / 1.0 are visually equidistant and
# the axis runs tight from 0 to 1.0 with no log-induced gap before 0).
PANES = [
    {"sweep": "robust_prc",
     "x_label": r"Core Robustness ($p_{cr}$)", "title": "GRAM",
     "x_categorical": True, "core_split": True, "drop_x": [0.1, 0.9]},
    {"sweep": "aux_route_prc",
     "x_label": r"Aux Spread ($p_{as}$)", "title": "GRAM", "x_scale": None},
    {"sweep": "core_aux_ratio",
     "x_label": r"Core : Aux ($p_{ca}$)", "title": "FT-LoRA",
     "x_categorical": True},
]


def aggregate(df: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    """Mean within each seed, then mean + t-CI across seeds, per (x, class)."""
    within = (df.groupby(["x", "label_class", "seed"])["compute_ratio"]
                .mean().reset_index())
    agg = (within.groupby(["x", "label_class"])["compute_ratio"]
                  .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    alpha = 1.0 - ci_level
    t_crit = np.where(agg["count"] > 1,
                      st.t.ppf(1 - alpha / 2, agg["count"] - 1), 0.0)
    agg["ci"] = t_crit * sem
    return agg.sort_values(["x", "label_class"])


def draw_panel(ax: plt.Axes, df: pd.DataFrame, cfg: dict,
               ci_level: float, show_ylabel: bool,
               box_aspect: float | None = 0.95,
               legend_labels: bool = True) -> None:
    drop_x = cfg.get("drop_x")
    if drop_x:
        df = df[~df["x"].apply(
            lambda v: any(abs(v - d) < 1e-9 for d in drop_x))]
    xs_all = sorted(df["x"].unique())
    categorical = cfg.get("x_categorical", False)
    pos = {v: i for i, v in enumerate(xs_all)}

    def X(vals):
        """Map sweep values to plot positions (evenly spaced if categorical)."""
        return (np.array([pos[v] for v in vals]) if categorical
                else np.asarray(vals, dtype=float))

    def _line(sub_df, color, label):
        """Aggregate one already-single-class subset and plot it as a line."""
        if sub_df.empty:
            return
        a = aggregate(sub_df, ci_level).sort_values("x")
        xs = X(a["x"].values)
        ys = a["mean"].values
        ci = np.where(a["count"].values > 1, a["ci"].values, 0.0)
        ax.fill_between(xs, ys - ci, ys + ci, color=color, alpha=0.18, linewidth=0)
        ax.plot(xs, ys, marker="o", markersize=4, color=color,
                label=label if legend_labels else "_nolegend_")

    # GRAM panels split the aggregate Core into Core-Partial (core + one aux) and
    # Core-Full (core only); other panels keep the aggregate Core line. Draw the
    # core line(s) first so the legend reads core-first (draw order == legend order
    # in the matplot2tikz groupplot export).
    core_split = cfg.get("core_split")
    if core_split:
        core = df[df["label_class"] == "core"]
        _line(core[core["retained"] != "core"], CORE_PARTIAL_COLOR, CORE_PARTIAL_LABEL)
        _line(core[core["retained"] == "core"], CORE_FULL_COLOR, CORE_FULL_LABEL)

    agg = aggregate(df, ci_level)
    for lc in CLASS_ORDER:
        if core_split and lc == "core":
            continue
        sub = agg[agg["label_class"] == lc].sort_values("x")
        if sub.empty:
            continue
        xs = X(sub["x"].values)
        ys = sub["mean"].values
        ci = np.where(sub["count"].values > 1, sub["ci"].values, 0.0)
        color = CLASS_COLOR[lc]
        ax.fill_between(xs, ys - ci, ys + ci, color=color, alpha=0.18,
                        linewidth=0)
        ax.plot(xs, ys, marker="o", markersize=4, color=color,
                label=CLASS_DISPLAY[lc] if legend_labels else "_nolegend_")

    ax.axhline(1.0, color="#4d4d4d", linestyle="-", linewidth=0.8, alpha=0.85,
               zorder=1)
    ax.set_xlabel(cfg["x_label"], fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Compute Ratio", fontsize=11)
    if cfg.get("title"):
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
    ax.grid(True, which="major", linestyle="-", alpha=0.4)
    if categorical:
        # Evenly-spaced positions; axis runs tight from first to last value.
        ax.set_xticks(range(len(xs_all)))
        ax.set_xticklabels([f"{x:g}" for x in xs_all], fontsize=9)
        ax.set_xlim(0, len(xs_all) - 1)
    else:
        if cfg.get("x_scale") == "log":
            ax.set_xscale("log")
        ax.set_xticks(xs_all)
        ax.set_xticklabels([f"{x:g}" for x in xs_all], fontsize=9)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="y", labelsize=9)
    if box_aspect:
        ax.set_box_aspect(box_aspect)


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df = df.dropna(subset=["compute_ratio", "x", "label_class"])
    return df[df["label_class"].isin(CLASS_ORDER)]


LEGEND_ORDER = ["Core ↑", CORE_PARTIAL_LABEL, CORE_FULL_LABEL,
                "Retain ↑", "Forget ↓", "Elicit ↓"]


def _ordered_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    lab2h = dict(zip(labels, handles))
    labels = ([l for l in LEGEND_ORDER if l in lab2h]
              + [l for l in labels if l not in LEGEND_ORDER])
    return [lab2h[l] for l in labels], labels


def save_tikz(fig, tex_path: Path, axis_width: str) -> None:
    """Export to a paper-ready tikzpicture; rewrite unicode arrows to math."""
    import matplot2tikz
    code = matplot2tikz.get_tikz_code(figure=fig, axis_width=axis_width,
                                      axis_height="3.57cm")
    # In groupplots matplot2tikz shares the legend across panels via internal
    # plot labels named after the legend text (e.g. "\label{Core ↑_plot}" and
    # "refstyle=Core ↑_plot"). Those names become csnames, so an arrow/math in
    # them breaks compilation -- strip the arrow from the "_plot" names only,
    # keeping it in the visible \addlegendentry text.
    code = code.replace(" ↑_plot", "_plot").replace(" ↓_plot", "_plot")
    code = code.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
    # matplot2tikz escapes underscores even in math, breaking subscripts like
    # p_{cr}; undo that for the "\_{...}" subscript pattern.
    code = code.replace(r"\_{", "_{")

    # --- match the paper's other line figures (sweep.tex / auxnum.tex) ---
    # Consistent fonts (matplot2tikz emits none -> pgfplots defaults to a too-big
    # \normalsize). Group options apply to every subplot; single axis otherwise.
    fonts = ("label style={font=\\footnotesize}, "
             "tick label style={font=\\scriptsize}, "
             "title style={font=\\footnotesize}, ")
    if r"\begin{groupplot}[" in code:
        code = code.replace(r"\begin{groupplot}[", r"\begin{groupplot}[" + fonts, 1)
    else:
        code = code.replace(
            "\\begin{axis}[\n",
            "\\begin{axis}[\nlabel style={font=\\footnotesize},\n"
            "tick label style={font=\\scriptsize},\n"
            "title style={font=\\footnotesize},\n", 1)
    # Faint grid so the dark 1.0 reference line stands out (matches sweep.tex).
    code = code.replace("x grid style={darkgray176}",
                        "x grid style={darkgray176, opacity=0.3}")
    code = code.replace("y grid style={darkgray176}",
                        "y grid style={darkgray176, opacity=0.3}")
    # Legend: smaller font, more whitespace above, and horizontally centered.
    # matplot2tikz emits it left-anchored on the first axis; recenter it (over
    # both panels for the groupplot, using ~figure-center in panel-1 coords).
    code = code.replace("legend style={\n", "legend style={\n  font=\\scriptsize,\n", 1)
    center_x = "1.03" if r"\begin{groupplot}[" in code else "0.5"
    code = code.replace("at={(0,1.02)}", f"at={{({center_x},1.16)}}", 1)
    code = code.replace("anchor=south west", "anchor=south", 1)
    tex_path.write_text(code)


def make_figure(df: pd.DataFrame, panes: list[dict], png_path: Path,
                tex_path: Path | None, ci: float, axis_width: str,
                box_aspect: float | None = 0.95) -> None:
    n = len(panes)
    figsize = (2.9 * n + 0.5, 3.4) if box_aspect else (3.0 * n, 3.2)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=200,
                             sharey=True, squeeze=False)
    axes = axes[0]
    for i, (ax, cfg) in enumerate(zip(axes, panes)):
        draw_panel(ax, df[df["sweep"] == cfg["sweep"]], cfg, ci,
                   show_ylabel=(i == 0), box_aspect=box_aspect,
                   legend_labels=(i == 0))
    handles, labels = _ordered_legend(axes[0])
    plt.tight_layout()
    if tex_path is None:
        # PNG-only overview: a single centered figure legend above the panels
        # (fig.legend looks best but isn't captured by matplot2tikz).
        fig.canvas.draw()
        # Use the tops of the panel titles (not just the axes) so the legend
        # clears them instead of overlapping.
        bbs = [ax.get_position() for ax in axes]
        cx = (bbs[0].x0 + bbs[-1].x1) / 2
        tops = [ax.title.get_window_extent().transformed(
                    fig.transFigure.inverted()).y1 for ax in axes]
        top = max(tops)
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(cx, top + 0.03), ncols=len(labels),
                   frameon=True, fontsize=10, columnspacing=1.4)
    else:
        # matplot2tikz captures an axes legend, not a figure legend.
        axes[0].legend(handles, labels, loc="lower left",
                       bbox_to_anchor=(0.0, 1.02), ncols=len(labels),
                       frameon=True, fontsize=9, columnspacing=1.2,
                       handletextpad=0.4)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, bbox_inches="tight")
    if tex_path is not None:
        try:
            save_tikz(fig, tex_path, axis_width)
            print(f"saved: {tex_path}")
        except Exception as e:
            print(f"tikz export skipped ({tex_path.name}): {e}")
    plt.close()
    print(f"saved: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--ci", type=float, default=0.9)
    parser.add_argument("--split", action="store_true",
                        help="Also write the three paper figures: sweep_robust.* "
                             "(GRAM p_cr, Core split), sweep_auxroute.* (GRAM "
                             "p_as), and sweep_lora.* (FT-LoRA core:aux).")
    args = parser.parse_args()

    df = load(args.input)

    # Default combined 3-pane overview (PNG only).
    make_figure(df, PANES, args.output, None, args.ci, axis_width=r"0.3\linewidth")

    if args.split:
        # Three separate single-pane paper figures. Only the robust panel splits
        # Core into Core-Partial / Core-Full; the other two show aggregate Core.
        robust, auxroute, lora = ([{**p, "title": None}] for p in PANES)
        for panes, stem in [(robust, "sweep_robust"),
                            (auxroute, "sweep_auxroute"),
                            (lora, "sweep_lora")]:
            make_figure(df, panes, ROOT / f"{stem}.png", ROOT / f"{stem}.tex",
                        args.ci, axis_width=r"0.5\linewidth", box_aspect=None)


if __name__ == "__main__":
    main()

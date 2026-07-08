#!/usr/bin/env python3
"""
Plot scaling compute-ratio curves -> scaling.png.

Four panels (Core / Retain (Virology) / Forget (Non-Virology) / Elicited
Forget), one line per method (Filtering / GRAM / FT-LoRA). Reads scaling.csv
(from compile.py); y is aggregated by seed mean, then mean + t-interval CI
across seeds, per model size.
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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "scaling.csv"
DEFAULT_PNG = SCRIPT_DIR / "scaling.png"
DEFAULT_TWEET_PNG = SCRIPT_DIR / "scaling_tweet.png"
DEFAULT_BLOG_PNG = SCRIPT_DIR / "scaling_blog.png"

# One panel per label_class.
PANELS = [
    ("core", "Core ↑"),
    ("retain", "Retain (Virology) ↑"),
    ("forget", "Forget (Non-Virology) ↓"),
    ("elicited_forget", "Elicited Forget ↓"),
]
# Tweet variant: drop the Retain panel, simpler titles.
PANELS_TWEET = [
    ("core", "Core ↑"),
    ("forget", "Forget ↓"),
    ("elicited_forget", "Elicited Forget ↓"),
]
YLIM = (0.3, 1.3)

# One line per method.
METHOD_COLOR = {"filtering": "#e6c200", "grmoe": "#4aab89", "lora": "#bb4771"}
METHOD_DISPLAY = {"filtering": "Filtering", "grmoe": "GRAM", "lora": "FT-LoRA"}
METHOD_MARKER = {"filtering": "o", "grmoe": "s", "lora": "^"}

PARAM_TICKS = [50e6, 100e6, 200e6, 400e6, 800e6, 2e9, 5e9]
PARAM_TICK_LABELS = ["50", "100", "200", "400", "800", "2000", "5000"]


def aggregate_across_seeds(df: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    """Mean within each seed, then t-CI across seeds, per num_params."""
    seed_means = (df.groupby(["num_params", "seed"])["compute_ratio"]
                    .mean().reset_index())
    agg = (seed_means.groupby("num_params")["compute_ratio"]
                     .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    alpha = 1.0 - ci_level
    t_crit = np.where(agg["count"] > 1,
                      st.t.ppf(1 - alpha / 2, agg["count"] - 1), 0.0)
    agg["ci"] = t_crit * sem
    return agg.sort_values("num_params")


def clamp_elicited(df: pd.DataFrame) -> pd.DataFrame:
    """CR semantics: higher = better. Elicited shouldn't worsen vs no FT."""
    key = ["method", "num_params", "seed", "data_label"]
    forget_lookup = (df[df["label_class"] == "forget"]
                     .set_index(key)["compute_ratio"])
    mask = df["label_class"] == "elicited_forget"
    if mask.any():
        keys = df.loc[mask, key].apply(tuple, axis=1)
        df.loc[mask, "compute_ratio"] = np.maximum(
            df.loc[mask, "compute_ratio"],
            keys.map(forget_lookup).fillna(-np.inf),
        )
    return df


def plot_panel(ax: plt.Axes, df_class: pd.DataFrame, title: str,
               show_ylabel: bool, ci_level: float,
               project_grmoe_2b: bool = False) -> None:
    """One panel = one label_class; one line per method."""
    grmoe_2b: tuple[float, float] | None = None
    for method, display in METHOD_DISPLAY.items():
        sub = df_class[df_class["method"] == method]
        if sub.empty:
            continue
        agg = aggregate_across_seeds(sub, ci_level)
        xs = agg["num_params"].values
        ys = agg["mean"].values
        yerr = np.where(agg["count"] > 1, agg["ci"].values, np.nan)
        color = METHOD_COLOR[method]
        marker = METHOD_MARKER.get(method, "o")
        # Draw GRAM on top of the other methods.
        z = 5 if method == "grmoe" else 3
        ax.plot(xs, ys, marker=marker, markersize=5, label=display,
                color=color, zorder=z)
        valid = ~np.isnan(yerr)
        ax.fill_between(xs[valid], (ys - yerr)[valid], (ys + yerr)[valid],
                        alpha=0.2, color=color, zorder=z - 1)
        if method == "grmoe":
            idx = int(np.argmin(np.abs(xs - 2e9)))
            if abs(xs[idx] - 2e9) / 2e9 < 0.1:
                grmoe_2b = (float(xs[idx]), float(ys[idx]))

    if project_grmoe_2b and grmoe_2b is not None:
        c = METHOD_COLOR["grmoe"]
        x0, y0 = grmoe_2b
        ax.plot([x0, 5e9], [y0, y0],
                linestyle=(0, (2, 1.5)), color=c, linewidth=1.5)
        # Very small solid square at the end of the dashed projection.
        ax.scatter([5e9], [y0], marker="s", s=4, color=c, zorder=6)

    ax.set_xscale("log")
    ax.set_title(title, fontsize=12, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Compute Ratio", fontsize=11)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.set_xticks(PARAM_TICKS)
    ax.set_xticklabels(PARAM_TICK_LABELS, fontsize=8)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(*YLIM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--ci", type=float, default=0.9)
    parser.add_argument("--tweet", action="store_true",
                        help="3-panel variant (Core / Forget / Elicited "
                             "Forget, no Retain) -> scaling_tweet.png.")
    parser.add_argument("--blog", action="store_true",
                        help="Same content as the default figure, but the "
                             "legend reads Filtering, GRAM, Filter + LoRA "
                             "(in that order) -> scaling_blog.png.")
    args = parser.parse_args()

    panels = PANELS_TWEET if args.tweet else PANELS
    out_path = args.output or (DEFAULT_TWEET_PNG if args.tweet
                               else DEFAULT_BLOG_PNG if args.blog
                               else DEFAULT_PNG)

    df = pd.read_csv(args.input)
    # scaling.csv now carries every retain config; this figure shows the
    # core+papers-biology config only (the filter the compile used to bake in).
    df = df[df["retained"] == "core+papers-biology"]
    df["num_params"] = pd.to_numeric(df["num_params"], errors="coerce")
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["num_params", "compute_ratio"])
    df = clamp_elicited(df)

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(1.95 * len(panels) * 1.1 ** 3 * 0.9,
                 3.375 * 0.75 * 0.9 * 1.1 ** 2),
        dpi=300, sharey=True, layout="constrained",
    )
    # A touch of vertical padding so the top legend isn't flush to panels.
    fig.get_layout_engine().set(h_pad=0.07)
    for idx, (lc, title) in enumerate(panels):
        plot_panel(
            axes[idx], df[df["label_class"] == lc], title,
            show_ylabel=(idx == 0), ci_level=args.ci,
            project_grmoe_2b=(lc == "retain"),
        )
    fig.supxlabel("Model Parameters (M)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    if args.blog:
        # Same plotted content; only reorder + relabel the legend entries.
        lab2h = dict(zip(labels, handles))
        spec = [("Filtering", "Filtering"), ("GRAM", "GRAM"),
                ("FT-LoRA", "Filter + LoRA")]
        handles = [lab2h[src] for src, _ in spec if src in lab2h]
        labels = [disp for src, disp in spec if src in lab2h]
    elif args.tweet:
        # Same content/order; only relabel FT-LoRA -> Filter + LoRA.
        labels = ["Filter + LoRA" if l == "FT-LoRA" else l for l in labels]
    fig.legend(handles, labels, loc="outside upper center",
               ncols=len(labels), fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot the auxnum compute-ratio curves -> auxnum.png (+ auxnum.tex).

One axis, four lines -- Core / Retain / Forget / Elicit -- with x = number of
auxiliary categories. This is the regenerated source for
``paper/figures/aux_scaling.tex``.

Reads auxnum.csv (from compile.py). For each label_class and num_aux, CR is
aggregated by seed mean first, then mean + t-interval CI across seeds (so the
band reflects seed variation, not the many retain configs within a seed).
A matplot2tikz export writes auxnum.tex alongside the PNG.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "auxnum.csv"
DEFAULT_PNG = SCRIPT_DIR / "auxnum.png"
DEFAULT_TEX = SCRIPT_DIR / "auxnum.tex"

# One line per label_class. Colours match paper/figures/aux_scaling.tex (tab10).
LINES = [
    ("core", "Core ↑", "#1f77b4"),
    ("retain", "Retain ↑", "#2ca02c"),
    ("forget", "Forget ↓", "#d62728"),
    ("elicited_forget", "Elicit ↓", "#ff7f0e"),
]
YLIM = (0.6, 1.0)
YTICKS = [0.6, 0.8, 1.0]


def aggregate_across_seeds(df: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    """Mean within each seed, then t-CI across seeds, per num_aux."""
    seed_means = (df.groupby(["num_aux", "seed"])["compute_ratio"]
                    .mean().reset_index())
    agg = (seed_means.groupby("num_aux")["compute_ratio"]
                     .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    alpha = 1.0 - ci_level
    t_crit = np.where(agg["count"] > 1,
                      st.t.ppf(1 - alpha / 2, agg["count"] - 1), 0.0)
    agg["ci"] = t_crit * sem
    return agg.sort_values("num_aux")


def clamp_elicited(df: pd.DataFrame) -> pd.DataFrame:
    """CR semantics: higher = better. Elicitation FT can only recover a
    forgotten capability, never worsen it, so floor each elicited_forget row at
    its matching (num_aux, seed, retained, data_label) forget row."""
    key = ["num_aux", "seed", "retained", "data_label"]
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


def make_figure(df: pd.DataFrame, ci_level: float):
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=300, layout="constrained")
    xmax = int(df["num_aux"].max())
    for lc, label, color in LINES:
        agg = aggregate_across_seeds(df[df["label_class"] == lc], ci_level)
        if agg.empty:
            continue
        xs = agg["num_aux"].values
        ys = agg["mean"].values
        yerr = np.where(agg["count"] > 1, agg["ci"].values, np.nan)
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.4,
                label=label, color=color, zorder=3)
        valid = ~np.isnan(yerr)
        ax.fill_between(xs[valid], (ys - yerr)[valid], (ys + yerr)[valid],
                        alpha=0.18, color=color, zorder=2, linewidth=0)
    # Reference line at parity with the baseline (matches arbsub/scaling).
    ax.axhline(1.0, color="#808080", linestyle="-", linewidth=0.5, alpha=0.6,
               zorder=1)
    ax.set_xlabel("Number of Auxiliary Categories", fontsize=11)
    ax.set_ylabel("Compute Ratio", fontsize=11)
    ax.set_xticks(sorted(df["num_aux"].unique()))
    ax.set_xlim(min(df["num_aux"]) - 0.8, xmax + 0.8)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.14),
              ncols=len(LINES), fontsize=10, frameon=True, columnspacing=1.6)
    return fig


def save_tikz(fig, tex_path: Path) -> None:
    """Export the figure to a paper-ready tikzpicture via matplot2tikz, then
    rewrite unicode arrows as LaTeX math so the legend compiles."""
    import matplot2tikz

    code = matplot2tikz.get_tikz_code(
        figure=fig, axis_width=r"0.82\linewidth", axis_height="4.2cm",
    )
    code = code.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
    # matplot2tikz emits no font styles -> pgfplots defaults to \normalsize,
    # which is too big and makes the legend overflow the column. Match the
    # paper's other line figures (\footnotesize labels, \scriptsize ticks/legend).
    code = code.replace(
        "\\begin{axis}[\n",
        "\\begin{axis}[\nlabel style={font=\\footnotesize},\n"
        "tick label style={font=\\scriptsize},\n", 1)
    code = code.replace(
        "legend style={\n", "legend style={\n  font=\\scriptsize,\n", 1)
    # matplot2tikz drops the explicit y-ticks; force them so the .tex matches
    # the PNG (0.6 / 0.8 / 1.0 only).
    ticks = ",".join(f"{t:g}" for t in YTICKS)
    labels = ",".join(f"{t:.1f}" for t in YTICKS)
    code = code.replace(
        "ytick style={color=black}",
        f"ytick={{{ticks}}},\nyticklabels={{{labels}}},\nytick style={{color=black}}")
    # Same for the x-ticks: show the actual num_aux values (4, 8, 12, 16, 20)
    # rather than pgfplots' default 5/10/15/20.
    ax0 = fig.axes[0]
    lo, hi = ax0.get_xlim()
    xt = [t for t in ax0.get_xticks() if lo <= t <= hi]
    xticks = ",".join(f"{t:g}" for t in xt)
    code = code.replace(
        "xtick style={color=black}",
        f"xtick={{{xticks}}},\nxticklabels={{{xticks}}},\nxtick style={{color=black}}")
    tex_path.write_text(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alt", action="store_true",
        help="Plot the 90%% core / 10%% aux composition variant: read "
             "auxnum_alt.csv and write auxnum_alt.{png,tex} by default.")
    parser.add_argument("-i", "--input", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--tex", type=Path, default=None)
    parser.add_argument("--ci", type=float, default=0.9)
    args = parser.parse_args()

    stem = "auxnum_alt" if args.alt else "auxnum"
    args.input = args.input or SCRIPT_DIR / f"{stem}.csv"
    args.output = args.output or SCRIPT_DIR / f"{stem}.png"
    args.tex = args.tex or SCRIPT_DIR / f"{stem}.tex"

    df = pd.read_csv(args.input)
    df["num_aux"] = pd.to_numeric(df["num_aux"], errors="coerce")
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["num_aux", "compute_ratio"])
    df["num_aux"] = df["num_aux"].astype(int)
    df = clamp_elicited(df)

    fig = make_figure(df, args.ci)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")
    try:
        save_tikz(fig, args.tex)
        print(f"Saved {args.tex}")
    except Exception as e:  # tikz export is best-effort; PNG is the main output
        print(f"tikz export skipped: {e}")
    plt.close(fig)


if __name__ == "__main__":
    main()

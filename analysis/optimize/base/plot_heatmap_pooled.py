#!/usr/bin/env python3
"""Pooled-across-seeds version of plot_heatmap.

Instead of one column per seed, pools all trials for each model size into a
single quadratic-surface fit and shows one column.

Usage:
    python -m analysis.optimize.plot_heatmap_pooled results/optimize/base/realistic
    python -m analysis.optimize.plot_heatmap_pooled results/optimize/base/realistic -o pooled.png --title BASE
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.gridspec import GridSpec

from analysis.optimize.common import load_trials
from analysis.optimize.plot_heatmap import (
    MODEL_ORDER,
    FIXED_LR_BOUNDS,
    FIXED_BS_BOUNDS,
    LR_TICKS,
    BS_TICKS,
    MIN_POINTS_FOR_FIT,
    build_surface,
    fmt_lr,
)


def plot_pooled(
    all_data: dict[str, dict[int, list[tuple[float, int, float]]]],
    output_path: Path,
    title: str,
    bs_bounds: tuple[float, float] = FIXED_BS_BOUNDS,
    lr_bounds: tuple[float, float] = FIXED_LR_BOUNDS,
    score_label: str = "Score (CE)",
):
    discovered = sorted(all_data.keys())
    models = [m for m in MODEL_ORDER if m in all_data]
    models += [m for m in discovered if m not in MODEL_ORDER]
    n_rows = len(models)

    fig = plt.figure(figsize=(5.5, 3.3 * n_rows), dpi=250)
    gs = GridSpec(
        n_rows, 2,
        figure=fig,
        width_ratios=[1, 0.04],
        hspace=0.45,
    )
    fig.suptitle(title, fontsize=14, y=0.995)

    for row, model_label in enumerate(models):
        seed_data = all_data[model_label]
        pooled: list[tuple[float, int, float]] = []
        for seed_trials in seed_data.values():
            pooled.extend(seed_trials)

        if not pooled:
            continue

        surface = None
        if len(pooled) >= MIN_POINTS_FOR_FIT:
            surface = build_surface(pooled, bs_bounds=bs_bounds, lr_bounds=lr_bounds)

        scores = np.array([t[2] for t in pooled], dtype=float)
        if surface is not None:
            vmin, vmax = float(np.nanmin(surface.zz)), float(np.nanmax(surface.zz))
        else:
            vmin, vmax = float(np.nanmin(scores)), float(np.nanmax(scores))
        levels = np.linspace(vmin, vmax, 20)

        ax = fig.add_subplot(gs[row, 0])

        if surface is not None:
            contour = ax.contourf(
                np.exp(surface.log_bs_range), np.exp(surface.log_lr_range), surface.zz,
                levels=levels, cmap="viridis_r",
            )
            lrs, bss, scores_arr = surface.lrs, surface.bss, surface.scores
        else:
            contour = None
            lrs = np.array([t[0] for t in pooled], dtype=float)
            bss = np.array([t[1] for t in pooled], dtype=float)
            scores_arr = scores

        ax.scatter(
            bss, lrs, c=scores_arr, cmap="viridis_r", vmin=vmin, vmax=vmax,
            edgecolors="white", linewidths=0.8, s=50, zorder=5,
        )
        if surface is not None:
            ax.scatter(
                surface.bs_opt, surface.lr_opt, marker="*", c="red", s=200,
                edgecolors="white", linewidths=0.8, zorder=10,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*bs_bounds)
        ax.set_ylim(*lr_bounds)
        x_ticks = [t for t in BS_TICKS if bs_bounds[0] <= t <= bs_bounds[1]] or BS_TICKS
        y_ticks = [t for t in LR_TICKS if lr_bounds[0] <= t <= lr_bounds[1]] or LR_TICKS
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in x_ticks]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([fmt_lr(v) for v in y_ticks]))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.tick_params(axis="both", labelsize=9)

        ax.set_ylabel(f"{model_label}\nLearning Rate", fontsize=11)
        if row == n_rows - 1:
            ax.set_xlabel("Batch Size", fontsize=11)
        ax.set_title(f"{len(pooled)} trials (pooled)", fontsize=10)

        if contour is not None:
            cax = fig.add_subplot(gs[row, 1])
            cbar = fig.colorbar(contour, cax=cax)
            cbar.set_label(score_label, fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--title", default="BASE")
    parser.add_argument("--bs-min", type=float, default=FIXED_BS_BOUNDS[0])
    parser.add_argument("--bs-max", type=float, default=FIXED_BS_BOUNDS[1])
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--ablated-core-only", action="store_true")
    args = parser.parse_args()

    output = args.output or Path(__file__).resolve().parent / "heatmaps_pooled.png"

    print(f"Scanning {args.results_dir} …")
    data = load_trials(
        args.results_dir,
        core_only=args.core_only,
        ablated_core_only=args.ablated_core_only,
    )
    for model in MODEL_ORDER:
        if model not in data:
            continue
        total = sum(len(t) for t in data[model].values())
        print(f"  {model}: {total} trials (across {len(data[model])} seeds)")

    title = f"{args.title} — Pooled Response Surfaces\nRed ★ = fitted optimum"
    plot_pooled(data, output, title, bs_bounds=(args.bs_min, args.bs_max))


if __name__ == "__main__":
    main()

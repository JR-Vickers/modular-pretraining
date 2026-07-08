#!/usr/bin/env python3
"""Plot per-seed response-surface heatmaps from raw trial results.

Usage:
    python plot_heatmap.py /path/to/results/optimize/base/realistic
    python plot_heatmap.py /path/to/results/optimize/base/realistic -o my_heatmaps.png
    python plot_heatmap.py /path/to/results/optimize/base/realistic --title "GRMOE"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, FormatStrFormatter

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
from matplotlib.gridspec import GridSpec

from analysis.optimize.common import (
    load_trials,
    compute_all_vsf,
)


# ---------------------------------------------------------------------------
# Predetermined (lr, bs) grids per model size
# ---------------------------------------------------------------------------

GRIDS: dict[str, dict] = {
    "50M":  {"lr": [4e-4, 8e-4, 1.6e-3], "bs": [32, 64, 128, 256]},
    "100M": {"lr": [4e-4, 8e-4, 1.6e-3], "bs": [32, 64, 128, 256]},
    "200M": {"lr": [2e-4, 4e-4, 8e-4, 1.6e-3], "bs": [64, 128, 256, 512]},
    "400M": {"lr": [2e-4, 4e-4, 8e-4], "bs": [256, 512, 1024]},
}

MODEL_ORDER = ["50M", "100M", "200M", "400M"]
FIXED_LR_BOUNDS = (1e-4, 6.4e-3)
FIXED_BS_BOUNDS = (16, 2048)
LR_TICKS = [2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3]
BS_TICKS = [32, 64, 128, 256, 512, 1024]
SURFACE_GRID_SIZE = 100
MIN_POINTS_FOR_FIT = 6


@dataclass
class SurfaceData:
    zz: np.ndarray
    log_lr_range: np.ndarray
    log_bs_range: np.ndarray
    lrs: np.ndarray
    bss: np.ndarray
    scores: np.ndarray
    lr_opt: float
    bs_opt: float


# ---------------------------------------------------------------------------
# Surface fitting
# ---------------------------------------------------------------------------

def fit_surface(trials: list[tuple[float, int, float]]):
    """Fit a quadratic in log-space: log(score) ~ 1 + logLR + logBS + logLR² + logBS² + logLR·logBS."""
    lrs = np.array([t[0] for t in trials])
    bss = np.array([t[1] for t in trials], dtype=float)
    scores = np.array([t[2] for t in trials])
    log_lr = np.log(lrs)
    log_bs = np.log(bss)
    log_s = np.log(scores)
    X = np.column_stack(
        [np.ones(len(trials)), log_lr, log_bs, log_lr**2, log_bs**2, log_lr * log_bs]
    )
    beta = np.linalg.lstsq(X, log_s, rcond=None)[0]
    return beta, lrs, bss, scores


def find_optimum(beta, lrs, bss):
    """Minimise the fitted quadratic (analytic if convex, else grid search)."""
    H = np.array([[2 * beta[3], beta[5]], [beta[5], 2 * beta[4]]])
    g = np.array([beta[1], beta[2]])
    ll_min, ll_max = np.log(lrs.min()), np.log(lrs.max())
    lb_min, lb_max = np.log(bss.min()), np.log(bss.max())
    try:
        opt = -np.linalg.solve(H, g)
        if (
            np.all(np.linalg.eigvalsh(H) > 0)
            and ll_min <= opt[0] <= ll_max
            and lb_min <= opt[1] <= lb_max
        ):
            return np.exp(opt[0]), np.exp(opt[1])
    except np.linalg.LinAlgError:
        pass
    best_s = np.inf
    best_pt = (lrs[0], bss[0])
    for ll in np.linspace(ll_min, ll_max, 200):
        for lb in np.linspace(lb_min, lb_max, 200):
            s = (
                beta[0]
                + beta[1] * ll
                + beta[2] * lb
                + beta[3] * ll**2
                + beta[4] * lb**2
                + beta[5] * ll * lb
            )
            if s < best_s:
                best_s = s
                best_pt = (np.exp(ll), np.exp(lb))
    return best_pt


def eval_surface(beta, log_lr, log_bs):
    a, b, c, d, e, f = beta
    return np.exp(
        a + b * log_lr + c * log_bs + d * log_lr**2 + e * log_bs**2 + f * log_lr * log_bs
    )


def build_surface(
    trials: list[tuple[float, int, float]],
    bs_bounds: tuple[float, float] = FIXED_BS_BOUNDS,
    lr_bounds: tuple[float, float] = FIXED_LR_BOUNDS,
) -> SurfaceData:
    beta, lrs, bss, scores = fit_surface(trials)
    lr_opt, bs_opt = find_optimum(beta, lrs, bss)
    log_lr_range = np.linspace(np.log(lr_bounds[0]), np.log(lr_bounds[1]), SURFACE_GRID_SIZE)
    log_bs_range = np.linspace(np.log(bs_bounds[0]), np.log(bs_bounds[1]), SURFACE_GRID_SIZE)
    LL, BB = np.meshgrid(log_lr_range, log_bs_range)
    zz = eval_surface(beta, LL.T, BB.T)
    return SurfaceData(
        zz=zz,
        log_lr_range=log_lr_range,
        log_bs_range=log_bs_range,
        lrs=lrs,
        bss=bss,
        scores=scores,
        lr_opt=lr_opt,
        bs_opt=bs_opt,
    )


# ---------------------------------------------------------------------------
# Tick formatting
# ---------------------------------------------------------------------------

def fmt_lr(v: float) -> str:
    exp = int(np.floor(np.log10(v)))
    mantissa = v / 10**exp
    if abs(mantissa - round(mantissa)) < 0.01:
        return f"{int(round(mantissa))}e{exp}"
    elif abs(mantissa * 10 - round(mantissa * 10)) < 0.1:
        return f"{mantissa:.1f}e{exp}"
    return f"{mantissa:.2f}e{exp}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    all_data: dict[str, dict[int, list[tuple[float, int, float]]]],
    output_path: Path,
    title: str,
    bs_bounds: tuple[float, float] = FIXED_BS_BOUNDS,
    lr_bounds: tuple[float, float] = FIXED_LR_BOUNDS,
    score_label: str = "Score (CE)",
    vsf_map: dict[tuple[str, int, float, int], float] | None = None,
):
    discovered = sorted(all_data.keys())
    models = [m for m in MODEL_ORDER if m in all_data]
    models += [m for m in discovered if m not in MODEL_ORDER]
    seeds = sorted({s for d in all_data.values() for s in d})
    n_rows = len(models)
    n_cols = len(seeds)

    fig = plt.figure(figsize=(2.7 * n_cols + 0.6, 1.98 * n_rows), dpi=250)
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[1] * n_cols + [0.04],
        wspace=0.35,
        hspace=0.45,
    )
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    for row, model_label in enumerate(models):
        seed_data = all_data[model_label]

        row_trials: list[list[tuple[float, int, float]]] = []
        row_surfaces: list[SurfaceData | None] = []
        for seed in seeds:
            trials = seed_data.get(seed, [])
            row_trials.append(trials)
            if len(trials) < MIN_POINTS_FOR_FIT:
                row_surfaces.append(None)
                continue
            row_surfaces.append(build_surface(trials, bs_bounds=bs_bounds, lr_bounds=lr_bounds))

        row_scores = [t[2] for trials in row_trials for t in trials]
        if not row_scores:
            continue
        valid_surfaces = [s for s in row_surfaces if s is not None]
        if valid_surfaces:
            all_zz = np.concatenate([s.zz.ravel() for s in valid_surfaces])
            vmin, vmax = float(np.nanmin(all_zz)), float(np.nanmax(all_zz))
        else:
            # Fallback for rows with points but no fitted surfaces.
            vmin, vmax = float(np.nanmin(row_scores)), float(np.nanmax(row_scores))
        levels = np.linspace(vmin, vmax, 20)

        last_contour = None
        for col, seed in enumerate(seeds):
            ax = fig.add_subplot(gs[row, col])
            trials = row_trials[col]

            if not trials:
                ax.text(
                    0.5, 0.5,
                    f"Seed {seed}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10,
                )
            else:
                surface = row_surfaces[col]
                if surface is not None:
                    contour = ax.contourf(
                        np.exp(surface.log_bs_range), np.exp(surface.log_lr_range), surface.zz,
                        levels=levels, cmap="viridis_r",
                    )
                    last_contour = contour
                    lrs, bss, scores = surface.lrs, surface.bss, surface.scores
                else:
                    lrs = np.array([t[0] for t in trials], dtype=float)
                    bss = np.array([t[1] for t in trials], dtype=float)
                    scores = np.array([t[2] for t in trials], dtype=float)

                ax.scatter(
                    bss, lrs, c=scores, cmap="viridis_r", vmin=vmin, vmax=vmax,
                    edgecolors="white", linewidths=0.8, s=50, zorder=5,
                )
                if vsf_map is not None:
                    for i in range(len(lrs)):
                        key = (model_label, seed, float(lrs[i]), int(bss[i]))
                        vsf_val = vsf_map.get(key, float("nan"))
                        if np.isfinite(vsf_val):
                            ax.annotate(
                                f"{vsf_val:.2f}",
                                (bss[i], lrs[i]),
                                textcoords="offset points",
                                xytext=(0, 7),
                                ha="center", va="bottom",
                                fontsize=6, fontweight="bold",
                                color="red" if vsf_val >= 0.1 else "green",
                                zorder=15,
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
            x_ticks = [t for t in BS_TICKS if bs_bounds[0] <= t <= bs_bounds[1]]
            if not x_ticks:
                x_ticks = BS_TICKS
            y_ticks = [t for t in LR_TICKS if lr_bounds[0] <= t <= lr_bounds[1]]
            if not y_ticks:
                y_ticks = LR_TICKS
            ax.xaxis.set_major_locator(FixedLocator(x_ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in x_ticks]))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.yaxis.set_major_locator(FixedLocator(y_ticks))
            ax.yaxis.set_major_formatter(FixedFormatter([fmt_lr(v) for v in y_ticks]))
            ax.yaxis.set_minor_locator(FixedLocator([]))
            ax.tick_params(axis="both", labelsize=9)

            if row == 0:
                ax.set_title(f"Seed {seed}", fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{model_label}\nLearning Rate", fontsize=11)
            if row == n_rows - 1:
                ax.set_xlabel("Batch Size", fontsize=11)

        if last_contour is not None:
            cax = fig.add_subplot(gs[row, n_cols])
            # Pull the colorbar in toward the rightmost heatmap column.
            _p = cax.get_position()
            cax.set_position([_p.x0 - 0.03, _p.y0, _p.width, _p.height])
            cbar = fig.colorbar(last_contour, cax=cax)
            cbar.set_label(score_label, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-seed response-surface heatmaps from raw trial results."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Parent directory containing {50M,100M,...}/seed_*/…/trial_*/ sub-trees",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <results_dir>/../heatmaps.png relative to script)",
    )
    parser.add_argument(
        "--title",
        default="BASE",
        help="Method label used in figure title (default: BASE)",
    )
    parser.add_argument(
        "--bs-min",
        type=float,
        default=FIXED_BS_BOUNDS[0],
        help=f"Lower batch-size bound for x-axis/surface (default: {FIXED_BS_BOUNDS[0]}).",
    )
    parser.add_argument(
        "--bs-max",
        type=float,
        default=FIXED_BS_BOUNDS[1],
        help=f"Upper batch-size bound for x-axis/surface (default: {FIXED_BS_BOUNDS[1]}).",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="If set, score from core rows only (data_label == 'core').",
    )
    parser.add_argument(
        "--ablated-core-only",
        action="store_true",
        help="Score = core loss from fully-ablated eval (retained=['core'], data_label='core').",
    )
    parser.add_argument(
        "--vsf",
        action="store_true",
        help="Annotate each point with its vs-filtering score.",
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        default=None,
        help="Root of filtered results (for --vsf).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Root of baseline scaling results (for --vsf).",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path(__file__).resolve().parent / "heatmaps.png"

    print(f"Scanning {args.results_dir} …")
    data = load_trials(
        args.results_dir,
        core_only=args.core_only,
        ablated_core_only=args.ablated_core_only,
    )

    for model in MODEL_ORDER:
        if model not in data:
            continue
        for seed in sorted(data[model]):
            n = len(data[model][seed])
            print(f"  {model} seed {seed}: {n} trials")

    vsf = None
    if args.vsf:
        results_dir = args.results_dir.resolve()
        filtered_root = (
            args.filtered.resolve() if args.filtered
            else results_dir.parents[2] / "filtering" / "realistic"
        )
        baseline_root = (
            args.baseline.resolve() if args.baseline
            else results_dir.parents[3] / "scaling" / "realistic" / "base"
        )
        print(f"\nComputing vsf scores …")
        print(f"  Filtered: {filtered_root}")
        print(f"  Baseline: {baseline_root}")
        vsf = compute_all_vsf(results_dir, baseline_root, filtered_root)
        n_scored = sum(1 for v in vsf.values() if np.isfinite(v))
        print(f"  Scored {n_scored}/{len(vsf)} trials")

    title = (f"{args.title} \u2014 Per-Seed Response Surfaces\nRed \u2605 = fitted optimum"
             if args.title else "")
    plot(data, output, title, bs_bounds=(args.bs_min, args.bs_max), vsf_map=vsf)


if __name__ == "__main__":
    main()

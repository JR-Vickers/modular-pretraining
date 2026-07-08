#!/usr/bin/env python3
"""Fit and plot optimal-hyperparameter power laws from raw trial results.

The script:
  1. pools all trials across seeds per model size,
  2. fits a single quadratic surface per model size to find LR*/BS*,
  3. fits a log-log power law across model sizes,
  4. plots the power law with per-size optima and +0.1% CEL error bars
     from each model size's quadratic surface.

Usage:
    python -m analysis.optimize.plot_power_law results/optimize/grmoe/realistic/new/opt_lr_bs
    python -m analysis.optimize.plot_power_law results/optimize/base/realistic -o power_laws.png --title BASE
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

from analysis.optimize.common import load_trials


# ---------------------------------------------------------------------------
# Model-size mapping
# ---------------------------------------------------------------------------

MODEL_ORDER = ["50M", "100M", "200M", "400M"]

N_VALUES: dict[str, float] = {
    "50M": 50e6, "100M": 100e6, "200M": 200e6, "400M": 400e6,
}


# ---------------------------------------------------------------------------
# Surface fitting helpers
# ---------------------------------------------------------------------------

def _fit_and_find_optimum(log_lr, log_bs, log_s, lr_min, lr_max, bs_min, bs_max):
    """Fit quadratic in log-space and return (log_lr_opt, log_bs_opt)."""
    n = len(log_lr)
    X = np.column_stack(
        [np.ones(n), log_lr, log_bs, log_lr**2, log_bs**2, log_lr * log_bs]
    )
    beta = np.linalg.lstsq(X, log_s, rcond=None)[0]

    H = np.array([[2 * beta[3], beta[5]], [beta[5], 2 * beta[4]]])
    g = np.array([beta[1], beta[2]])
    ll_min, ll_max = np.log(lr_min), np.log(lr_max)
    lb_min, lb_max = np.log(bs_min), np.log(bs_max)
    try:
        opt = -np.linalg.solve(H, g)
        if (
            np.all(np.linalg.eigvalsh(H) > 0)
            and ll_min <= opt[0] <= ll_max
            and lb_min <= opt[1] <= lb_max
        ):
            return opt[0], opt[1]
    except np.linalg.LinAlgError:
        pass
    best_s = np.inf
    best_ll, best_lb = (ll_min + ll_max) / 2, (lb_min + lb_max) / 2
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
                best_ll, best_lb = ll, lb
    return best_ll, best_lb


def _fit_pl(logN, logy):
    """Least-squares fit: logy = a + b*logN. Returns (a, b, R²)."""
    X = np.column_stack([np.ones(len(logN)), logN])
    beta = np.linalg.lstsq(X, logy, rcond=None)[0]
    res = logy - X @ beta
    ss_res = np.sum(res**2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return beta[0], beta[1], r2


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _find_optimum_for_trials(trials, global_lr_min, global_lr_max, global_bs_min, global_bs_max):
    """Fit quadratic surface and return (lr_opt, bs_opt, lr_range, bs_range).

    lr_range and bs_range are (lo, hi) tuples for the +0.1% CEL region.
    """
    log_lrs = np.array([np.log(lr) for lr, _, _ in trials])
    log_bss = np.array([np.log(bs) for _, bs, _ in trials])
    log_scores = np.array([np.log(s) for _, _, s in trials])

    log_lr_span = log_lrs.max() - log_lrs.min()
    log_bs_span = log_bss.max() - log_bss.min()
    ext_lr_min = np.exp(log_lrs.min() - 0.5 * max(log_lr_span, 0.1))
    ext_lr_max = np.exp(log_lrs.max() + 0.5 * max(log_lr_span, 0.1))
    ext_bs_min = np.exp(log_bss.min() - 0.5 * max(log_bs_span, 0.1))
    ext_bs_max = np.exp(log_bss.max() + 0.5 * max(log_bs_span, 0.1))

    opt_ll, opt_lb = _fit_and_find_optimum(
        log_lrs, log_bss, log_scores, ext_lr_min, ext_lr_max, ext_bs_min, ext_bs_max,
    )

    n = len(log_lrs)
    X = np.column_stack(
        [np.ones(n), log_lrs, log_bss, log_lrs**2, log_bss**2, log_lrs * log_bss]
    )
    beta = np.linalg.lstsq(X, log_scores, rcond=None)[0]
    min_val = (beta[0] + beta[1]*opt_ll + beta[2]*opt_lb
               + beta[3]*opt_ll**2 + beta[4]*opt_lb**2 + beta[5]*opt_ll*opt_lb)
    threshold = min_val + np.log(1.001)

    ll_scan = np.linspace(np.log(ext_lr_min), np.log(ext_lr_max), 2000)
    lr_surf = (beta[0] + beta[1]*ll_scan + beta[2]*opt_lb
               + beta[3]*ll_scan**2 + beta[4]*opt_lb**2 + beta[5]*ll_scan*opt_lb)
    within = ll_scan[lr_surf <= threshold]
    lr_range = (np.exp(within.min()), np.exp(within.max())) if len(within) > 0 else (np.exp(opt_ll), np.exp(opt_ll))

    lb_scan = np.linspace(np.log(ext_bs_min), np.log(ext_bs_max), 2000)
    bs_surf = (beta[0] + beta[1]*opt_ll + beta[2]*lb_scan
               + beta[3]*opt_ll**2 + beta[4]*lb_scan**2 + beta[5]*opt_ll*lb_scan)
    within = lb_scan[bs_surf <= threshold]
    bs_range = (np.exp(within.min()), np.exp(within.max())) if len(within) > 0 else (np.exp(opt_lb), np.exp(opt_lb))

    return np.exp(opt_ll), np.exp(opt_lb), lr_range, bs_range


def compute_power_laws(
    data: dict[str, dict[int, list[tuple[float, int, float]]]],
):
    """Pool trials across seeds, fit one surface per model size, then power law.

    Steps:
      1. Pool all trials across seeds for each model size.
      2. Fit a single quadratic surface per model size → (LR*, BS*).
      3. Derive +0.1% CEL error bars from that surface.
      4. Fit a log-log power law across model sizes.
    """
    model_optima: dict[str, tuple[float, float]] = {}
    model_ranges_lr: dict[str, tuple[float, float]] = {}
    model_ranges_bs: dict[str, tuple[float, float]] = {}

    print("\nPer-model-size surface optima (pooled across seeds):")
    models_with_data: list[str] = []
    for model_label in MODEL_ORDER:
        if model_label not in data:
            continue
        pooled: list[tuple[float, int, float]] = []
        for seed_trials in data[model_label].values():
            pooled.extend(seed_trials)
        if len(pooled) < 6:
            print(f"  {model_label}: skipped ({len(pooled)} < 6 trials)")
            continue

        lr_opt, bs_opt, lr_range, bs_range = _find_optimum_for_trials(
            pooled,
            min(t[0] for t in pooled), max(t[0] for t in pooled),
            min(t[1] for t in pooled), max(t[1] for t in pooled),
        )
        models_with_data.append(model_label)
        model_optima[model_label] = (lr_opt, bs_opt)
        model_ranges_lr[model_label] = lr_range
        model_ranges_bs[model_label] = bs_range
        print(f"  {model_label} ({len(pooled)} trials): LR*={lr_opt:.4e}, BS*={bs_opt:.0f}")

    if len(models_with_data) < 2:
        raise ValueError("Need at least 2 model sizes to fit power laws")

    fit_logN = np.array([np.log(N_VALUES[m]) for m in models_with_data])
    fit_loglr = np.array([np.log(model_optima[m][0]) for m in models_with_data])
    fit_logbs = np.array([np.log(model_optima[m][1]) for m in models_with_data])
    fit_a_lr, fit_b_lr, fit_r2_lr = _fit_pl(fit_logN, fit_loglr)
    fit_a_bs, fit_b_bs, fit_r2_bs = _fit_pl(fit_logN, fit_logbs)

    print(f"\nPower law fit ({len(models_with_data)} points):")
    print(f"  LR*(N) = {np.exp(fit_a_lr):.4f} × N^({fit_b_lr:.4f}), R²={fit_r2_lr:.3f}")
    print(f"  BS*(N) = {np.exp(fit_a_bs):.4f} × N^({fit_b_bs:.4f}), R²={fit_r2_bs:.3f}")

    return (models_with_data, model_optima,
            fit_a_lr, fit_b_lr, fit_r2_lr, fit_a_bs, fit_b_bs, fit_r2_bs,
            model_ranges_lr, model_ranges_bs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    models_with_data,
    model_optima,
    a_lr, b_lr, r2_lr,
    a_bs, b_bs, r2_bs,
    output_path: Path,
    title: str,
    ranges_lr=None,
    ranges_bs=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.75, 3.75))
    N_plot = np.linspace(30e6, 800e6, 200)

    Ns = np.array([N_VALUES[m] for m in models_with_data])
    lr_vals = np.array([model_optima[m][0] for m in models_with_data])
    bs_vals = np.array([model_optima[m][1] for m in models_with_data])

    # --- LR panel ---
    if ranges_lr is not None:
        lr_lo = lr_vals - np.array([ranges_lr[m][0] for m in models_with_data])
        lr_hi = np.array([ranges_lr[m][1] for m in models_with_data]) - lr_vals
        ax1.errorbar(
            Ns, lr_vals, yerr=[lr_lo, lr_hi],
            fmt="kD", markersize=8, capsize=5, lw=2, zorder=6,
            label="Optimum (+0.1% CEL)",
        )
    else:
        ax1.plot(Ns, lr_vals, "kD", markersize=8, zorder=6, label="Optimum (+0.1% CEL)")
    ax1.plot(
        N_plot, np.exp(a_lr) * N_plot**b_lr, "k-", linewidth=2,
        label="Prediction",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    all_lr_hi = max(
        [ranges_lr[m][1] for m in models_with_data] if ranges_lr else [1.6e-3],
        default=1.6e-3,
    )
    lr_y_max = max(3.2e-3, all_lr_hi * 1.2)
    ax1.set_ylim(2e-4, lr_y_max)
    lr_ticks = [2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3]
    ax1.yaxis.set_major_locator(FixedLocator(lr_ticks))
    ax1.yaxis.set_minor_locator(FixedLocator([]))
    def _fmt_lr(v, _):
        exp = int(np.floor(np.log10(v)))
        coeff = v / 10**exp
        return f"{coeff:.0f}E{exp}" if coeff == int(coeff) else f"{coeff:.1f}E{exp}"
    ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_lr))
    ax1.set_xlabel("Model Size N (parameters)", fontsize=12)
    ax1.set_ylabel("Optimal Learning Rate", fontsize=12)
    ax1.set_title(
        f"LR*(N) = {np.exp(a_lr):.3f} x N^({b_lr:.3f}),  "
        f"R\u00b2 = {r2_lr:.2f}",
        fontsize=12, pad=12,
    )
    ax1.legend(fontsize=9, loc="best")
    ax1.set_xticks([5e7, 1e8, 2e8, 4e8, 8e8])
    ax1.set_xticklabels(["50M", "100M", "200M", "400M", "800M"])

    # --- BS panel ---
    if ranges_bs is not None:
        bs_lo = bs_vals - np.array([ranges_bs[m][0] for m in models_with_data])
        bs_hi = np.array([ranges_bs[m][1] for m in models_with_data]) - bs_vals
        ax2.errorbar(
            Ns, bs_vals, yerr=[bs_lo, bs_hi],
            fmt="kD", markersize=8, capsize=5, lw=2, zorder=6,
            label="Optimum (+0.1% CEL)",
        )
    else:
        ax2.plot(Ns, bs_vals, "kD", markersize=8, zorder=6, label="Optimum (+0.1% CEL)")
    ax2.plot(
        N_plot, np.exp(a_bs) * N_plot**b_bs, "k-", linewidth=2,
        label="Prediction",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    bs_ticks = [32, 64, 128, 256, 512]
    ax2.yaxis.set_major_locator(FixedLocator(bs_ticks))
    ax2.yaxis.set_minor_locator(FixedLocator([]))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax2.set_xlabel("Model Size N (parameters)", fontsize=12)
    ax2.set_ylabel("Optimal Batch Size", fontsize=12)
    ax2.set_title(
        f"BS*(N) = {np.exp(a_bs):.2e} x N^({b_bs:.3f}),  "
        f"R\u00b2 = {r2_bs:.2f}",
        fontsize=12, pad=12,
    )
    ax2.legend(fontsize=9, loc="best")
    ax2.set_xticks([5e7, 1e8, 2e8, 4e8, 8e8])
    ax2.set_xticklabels(["50M", "100M", "200M", "400M", "800M"])

    if title:
        fig.suptitle(
            f"{title} \u2014 Optimal Hyperparameter Power Laws",
            fontsize=14, fontweight="bold", y=1.02,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit and plot optimal-hyperparameter power laws from raw trial results."
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
        help="Output PNG path (default: power_laws.png next to this script)",
    )
    parser.add_argument(
        "--title",
        default="BASE",
        help="Method label used in figure title (default: BASE)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="If provided, only include these seeds (e.g. --seeds 1 2)",
    )
    parser.add_argument(
        "--ablated-core-only",
        action="store_true",
        help="Score = core loss from fully-ablated eval (retained=['core'], data_label='core').",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path(__file__).resolve().parent / "power_laws.png"

    print(f"Scanning {args.results_dir} …")
    data = load_trials(args.results_dir, ablated_core_only=args.ablated_core_only)

    if args.seeds is not None:
        keep = set(args.seeds)
        for model in list(data):
            data[model] = {s: v for s, v in data[model].items() if s in keep}
            if not data[model]:
                del data[model]
        print(f"  Filtering to seeds: {sorted(keep)}")

    for model in MODEL_ORDER:
        if model not in data:
            continue
        for seed in sorted(data[model]):
            n = len(data[model][seed])
            print(f"  {model} seed {seed}: {n} trials")

    (
        models_with_data, model_optima,
        a_lr, b_lr, r2_lr,
        a_bs, b_bs, r2_bs,
        ranges_lr, ranges_bs,
    ) = compute_power_laws(data)

    plot(
        models_with_data, model_optima,
        a_lr, b_lr, r2_lr,
        a_bs, b_bs, r2_bs,
        output, args.title,
        ranges_lr, ranges_bs,
    )

    json_path = output.with_name("power_laws.json")
    params = {
        "lr": {"coef": float(np.exp(a_lr)), "exp": float(b_lr), "r2": float(r2_lr)},
        "bs": {"coef": float(np.exp(a_bs)), "exp": float(b_bs), "r2": float(r2_bs)},
    }
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

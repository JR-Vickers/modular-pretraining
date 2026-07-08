"""
Shared plotting utilities for the analysis experiment dirs.

Every experiment's ``plot.py`` imports :func:`grouped_bar_chart` (and the
shared display-name / colour maps) from here so the figures share one styling
and one seed-aggregation convention.  ``grouped_bar_chart`` aggregates two
ways: it averages within each seed first, then takes a t-interval across
seeds (see :func:`aggregate_by_seed`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


# ============================================================
# Display Name Mappings
# ============================================================

STAGE_NAMES = {
    "baseline": "Baseline",
    "filtering": "Filtering",
    "coreftaux": "Finedtuned Core",
    "routed": "Routed",
    "rmu": "RMU",
    "maxent": "MaxEnt",
    "gradient_ascent": "Grad Ascent",
}

FIELD_NAMES = {
    "name": "Stage",
    "loss": "Loss",
    "compute_ratio": "Compute Ratio",
    "step_equiv": "Step Equivalent",
    "ppl": "Perplexity",
    "ppl_ratio": "Perplexity Ratio",
    "loss_ratio": "Loss Ratio",
    "model_size": "Model Size",
}

LABEL_CLASS_COLORS = {
    "Core": "#1f77b4",
    "Retain": "#2ca02c",
    "Forget": "#d62728",
    "Elicited Forget": "#ff7f0e",
}


# ============================================================
# Aggregation
# ============================================================

def aggregate_by_seed(
    df,
    group_cols: list[str],
    y_col: str,
    seed_col: str = "seed",
    ci_level: float = 0.9,
):
    """Two-stage aggregation: average within seed first, then across seeds.

    Recommended for aggregate metrics where seed is the unit of replication.
    """
    # Stage 1: mean within each (group_cols..., seed)
    seed_group_cols = group_cols + [seed_col]
    seed_means = (
        df.groupby(seed_group_cols)[y_col]
        .mean()
        .reset_index()
        .rename(columns={y_col: "seed_mean"})
    )

    # Stage 2: t-interval across seeds
    agg = (
        seed_means.groupby(group_cols)["seed_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

    alpha = 1.0 - float(ci_level)
    t_crit = np.where(agg["count"] > 1, st.t.ppf(1 - alpha / 2, agg["count"] - 1), 0.0)
    agg["ci"] = t_crit * agg["sem"]

    return agg


# ============================================================
# Visualization Functions
# ============================================================

def grouped_bar_chart(
    df,
    x_col: str,
    y_col: str,
    group_col: str | None = None,
    seed_col: str = "seed",
    ci_level: float = 0.9,
    x_order: list[str] | None = None,
    group_order: list[str] | None = None,
    x_labels: dict[str, str] | None = None,
    group_labels: dict[str, str] | None = None,
    title: str | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    y_min: float | None = None,
    y_max: float | None = None,
    error_bars: bool = True,
    show_values: bool = True,
    fade_map: dict[str, list[str]] | None = None,
    colors: dict[str, str] | None = None,
    min_bar_height: float | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Axes, "object"]:
    """Create a grouped bar chart.

    Args:
        df: DataFrame with data to plot.
        x_col: Column for x-axis categories.
        y_col: Column for y-axis values.
        group_col: Column for bars within each x-category. If None, one bar per x.
        seed_col: Column containing seed identifiers.
        ci_level: Confidence interval level (0-1).
        x_order: Order of x-axis categories.
        group_order: Order of groups within each x-category.
        x_labels: Dict mapping x values to display labels.
        group_labels: Dict mapping group values to legend display labels.
        title: Plot title.
        x_axis_label, y_axis_label: Axis labels.
        figsize: Figure size.
        fontsize: Base font size.
        y_min, y_max: Y-axis limits.
        error_bars: If True, show confidence interval error bars.
        show_values: If True, show numeric values above bars.
        fade_map: Dict mapping x values to list of groups to fade.
        colors: Dict mapping group values to colors.
        min_bar_height: Minimum display height for bars.

    Returns:
        Tuple of (matplotlib Axes, aggregated DataFrame).
    """
    # Filter data
    data = df.copy()
    if x_order is not None:
        data = data[data[x_col].isin(x_order)]
    if group_col is not None and group_order is not None:
        data = data[data[group_col].isin(group_order)]

    # Dispatch to aggregation method
    group_cols = [x_col] if group_col is None else [x_col, group_col]
    agg = aggregate_by_seed(data, group_cols, y_col, seed_col, ci_level)

    # Determine order
    if x_order is not None:
        x_vals = [v for v in x_order if v in agg[x_col].values]
    else:
        x_vals = list(agg[x_col].unique())

    if group_col is not None:
        if group_order is not None:
            groups = [g for g in group_order if g in agg[group_col].values]
        else:
            groups = list(agg[group_col].unique())
    else:
        groups = [None]

    # Pivot for plotting
    if group_col is not None:
        pivot_mean = agg.pivot(index=x_col, columns=group_col, values="mean")
        pivot_ci = agg.pivot(index=x_col, columns=group_col, values="ci").fillna(0.0)
        pivot_count = agg.pivot(index=x_col, columns=group_col, values="count")
        pivot_mean = pivot_mean.reindex(index=x_vals, columns=groups)
        pivot_ci = pivot_ci.reindex(index=x_vals, columns=groups).fillna(0.0)
        pivot_count = pivot_count.reindex(index=x_vals, columns=groups)
    else:
        pivot_mean = agg.set_index(x_col)["mean"].reindex(x_vals)
        pivot_ci = agg.set_index(x_col)["ci"].reindex(x_vals).fillna(0.0)
        pivot_count = agg.set_index(x_col)["count"].reindex(x_vals)

    # Create figure
    if ax is None:
        if figsize is None:
            figsize = (max(6, len(x_vals) * 1.2), 5)
        fig, ax = plt.subplots(figsize=figsize)

    n_x = len(x_vals)
    n_groups = len(groups) if group_col else 1
    bar_spacing = 0.02
    total_width = 0.85 - (n_groups - 1) * bar_spacing
    bar_width = total_width / n_groups
    x = np.arange(n_x)

    color_map = {**(colors or {}), **LABEL_CLASS_COLORS}

    for i, g in enumerate(groups):
        offset = -total_width / 2 + bar_width / 2 + i * (bar_width + bar_spacing)

        if group_col is not None:
            means = pivot_mean[g].values
            errors = pivot_ci[g].values if error_bars else None
            counts = pivot_count[g].values if error_bars else None
        else:
            means = pivot_mean.values
            errors = pivot_ci.values if error_bars else None
            counts = pivot_count.values if error_bars else None

        if error_bars and errors is not None and counts is not None:
            errors = np.where(counts > 1, errors, np.nan)

        display_means = means.copy()
        if min_bar_height is not None:
            display_means = np.where(
                ~np.isnan(means), np.maximum(means, min_bar_height), means
            )

        bar_kwargs = {"width": bar_width, "edgecolor": "none"}
        if g is not None:
            label_map_g = group_labels or {}
            bar_kwargs["label"] = label_map_g.get(g, g)
            if g in color_map:
                bar_kwargs["color"] = color_map[g]
        if error_bars:
            bar_kwargs["yerr"] = errors
            bar_kwargs["capsize"] = 3
            bar_kwargs["ecolor"] = "black"

        bars = ax.bar(x + offset, display_means, **bar_kwargs)

        if fade_map is not None:
            for bar, x_val in zip(bars, x_vals):
                if g in fade_map.get(x_val, []):
                    bar.set_alpha(0.3)

        if show_values:
            for j, (bar, mean_val) in enumerate(zip(bars, means)):
                if not np.isnan(mean_val):
                    err = errors[j] if error_bars and not np.isnan(errors[j]) else 0
                    bar_top = display_means[j] if not np.isnan(display_means[j]) else mean_val
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar_top + err + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0] or 1),
                        f"{mean_val:.2f}",
                        ha="center", va="bottom", fontsize=fontsize * 0.8,
                    )

    ax.set_xticks(x)
    x_label_map = {**STAGE_NAMES, **(x_labels or {})}
    ax.set_xticklabels([x_label_map.get(v, v) for v in x_vals], fontsize=fontsize)

    xlabel = x_axis_label if x_axis_label is not None else FIELD_NAMES.get(x_col, x_col)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(y_axis_label or FIELD_NAMES.get(y_col, y_col), fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=fontsize)

    if y_min is not None or y_max is not None:
        ylim = ax.get_ylim()
        ax.set_ylim(y_min if y_min is not None else ylim[0],
                    y_max if y_max is not None else ylim[1])

    if group_col is not None:
        legend = ax.legend(fontsize=fontsize, bbox_to_anchor=(1.01, 1), loc="upper left")
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)

    if ax.figure is not None and len(ax.figure.axes) == 1:
        plt.tight_layout()

    return ax, agg

"""Render the three-panel Phase 3 headline figure from a compiled report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


MODEL_COLORS = {
    "gram": "#2F6B9A",
    "baseline": "#666666",
    "filtered": "#D17A22",
}


def _plot_line(ax: Axes, x: range, values: list[float], *, label: str,
               color: str, linestyle: str = "-") -> None:
    ax.plot(x, values, marker="o", linewidth=2, markersize=4.5, label=label,
            color=color, linestyle=linestyle)


def render_headline_figure(report: dict[str, Any], output: Path) -> None:
    """Render a compiled report's headline series to a 400-DPI PNG."""
    series = report["headline_series"]
    precisions = series["precision_order"]
    if precisions != ["FP32", "int8", "int6", "int4"]:
        raise ValueError(f"Unexpected precision order: {precisions}")
    x = range(len(precisions))
    raw = series["raw_deadline_loss"]
    signed = series["signed_percent"]
    retained = series["mean_retained_topic_relative_degradation_percent"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.25))

    _plot_line(axes[0], x, raw["gram_all_on"], label="GRAM all-on",
               color=MODEL_COLORS["gram"])
    _plot_line(axes[0], x, raw["gram_deadline_off"], label="GRAM deadline-off",
               color=MODEL_COLORS["gram"], linestyle="--")
    _plot_line(axes[0], x, raw["dense_baseline"], label="Dense baseline",
               color=MODEL_COLORS["baseline"])
    _plot_line(axes[0], x, raw["deadline_filtered"], label="Deadline-filtered",
               color=MODEL_COLORS["filtered"])
    axes[0].set_title("Deadline-topic loss")
    axes[0].set_ylabel("Cross-entropy loss (lower is better)")
    axes[0].legend(fontsize=8)

    _plot_line(axes[1], x, signed["capability_recovery"],
               label="Capability recovery", color=MODEL_COLORS["gram"])
    _plot_line(axes[1], x, signed["isolation_erosion"],
               label="Isolation erosion", color=MODEL_COLORS["gram"], linestyle="--")
    axes[1].axhline(0, color="#222222", linewidth=1, alpha=0.7)
    axes[1].axhline(20, color="#A33A3A", linewidth=1, linestyle=":", label="20% verdict threshold")
    axes[1].set_title("Signed isolation outcomes")
    axes[1].set_ylabel("Signed change (%) — unclipped")
    axes[1].legend(fontsize=8)

    _plot_line(axes[2], x, retained["gram_all_on"], label="GRAM all-on",
               color=MODEL_COLORS["gram"])
    _plot_line(axes[2], x, retained["dense_baseline"], label="Dense baseline",
               color=MODEL_COLORS["baseline"])
    _plot_line(axes[2], x, retained["deadline_filtered"], label="Deadline-filtered",
               color=MODEL_COLORS["filtered"])
    axes[2].axhline(0, color="#222222", linewidth=1, alpha=0.7)
    axes[2].axhline(10, color="#A33A3A", linewidth=1, linestyle=":", label="10% utility guard")
    axes[2].set_title("Retained-topic utility")
    axes[2].set_ylabel("Mean loss change from FP32 (%)\n(lower is better)")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(list(x), precisions)
        ax.set_xlabel("Weight precision")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    if report.get("manifest", {}).get("smoke"):
        fig.suptitle("SMOKE ONLY — pipeline validation, not scientific evidence",
                     color="#A33A3A", fontsize=11, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Directory containing phase3_report.json")
    parser.add_argument("--output", type=Path,
                        help="PNG path (default: RESULT_DIR/phase3_headline.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output = args.output.resolve() if args.output else result_dir / "phase3_headline.png"
    report = json.loads((result_dir / "phase3_report.json").read_text())
    render_headline_figure(report, output)
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot capability-titration curves for GRAM and FT-LoRA -> titration.png.

Reads titration.csv (from compile.py) and draws a 2-row x 4-column grid:

  columns: one per titrated aux capability (Virology / Cyber / Nuclear / Lisp)
  row 0:   compute ratio on the titrated aux  vs titration t  (capability dialed in)
  row 1:   compute ratio on core              vs titration t  (collateral effect)
  lines:   GRAM and FT-LoRA, mean over seeds with a 90% t-CI band

As t goes 0 -> 1 the aux module is scaled from off to on, so the top row should
rise (capability returning) while the bottom row stays flat (core untouched).

  python -m analysis.titration.plot
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "titration.csv"
OUT = ROOT / "titration.png"

# (csv method name, legend label, color, marker) -- colors/markers match scaling.png.
METHODS = [
    ("grmoe", "GRAM", "#4aab89", "s"),
    ("lora", "FT-LoRA", "#bb4771", "^"),
]
AUX_ORDER = ["papers-biology", "papers-cyber", "papers-nuclear", "code-lisp"]
AUX_DISPLAY = {
    "papers-biology": "Virology", "papers-cyber": "Cybersecurity",
    "papers-nuclear": "Nuclear", "code-lisp": "Lisp Code",
}


def curve(df: pd.DataFrame, method: str, aux: str, data_label: str,
          ci_level: float = 0.9):
    """Return (t, mean_CR, ci_halfwidth) over seeds for one (method, aux, label)."""
    sub = df[(df["method"] == method) & (df["aux"] == aux)
             & (df["data_label"] == data_label)]
    agg = (sub.groupby("titration")["compute_ratio"]
              .agg(["mean", "std", "count"]).reset_index().sort_values("titration"))
    if agg.empty:
        return np.array([]), np.array([]), np.array([])
    std = agg["std"].fillna(0.0).to_numpy()
    cnt = agg["count"].to_numpy()
    sem = std / np.sqrt(np.clip(cnt, 1, None))
    t_crit = np.where(cnt > 1, st.t.ppf(1 - (1 - ci_level) / 2, np.clip(cnt - 1, 1, None)), 0.0)
    return agg["titration"].to_numpy(), agg["mean"].to_numpy(), t_crit * sem


def main():
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df["titration"] = pd.to_numeric(df["titration"], errors="coerce")

    auxes = [a for a in AUX_ORDER if a in set(df["aux"])]
    n = len(auxes)

    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.0),
                             squeeze=False, sharex=True, sharey="row")
    for j, aux in enumerate(auxes):
        panels = [(0, aux), (1, "core")]
        for row, data_label in panels:
            ax = axes[row][j]
            for method, label, color, marker in METHODS:
                t, mean, ci = curve(df, method, aux, data_label)
                if t.size == 0:
                    continue
                ax.plot(t, mean, marker=marker, color=color, label=label,
                        markersize=4, linewidth=1.5)
                ax.fill_between(t, mean - ci, mean + ci, color=color, alpha=0.18)
            ax.axhline(1.0, color="gray", ls="-", lw=0.8, alpha=0.7)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(0, 1.2)
            if row == 0:
                ax.set_title(AUX_DISPLAY.get(aux, aux), fontsize=11, fontweight="bold")
            else:
                ax.set_xlabel("Titration $t$", fontsize=10)
            if j == 0:
                ax.set_ylabel("Compute Ratio\n" + ("(target aux)" if row == 0 else "(core)"),
                              fontsize=10)
            if row == 0 and j == 0:
                ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Emit a pgfplots/TikZ version of stories_agg.png: one bar group per method
(Filtering / GRAM / FT-LoRA / Demix / MaxEnt), four bars per group
(Core / Retain / Forget / Elicit), 90% t-CI error bars across seeds, value
labels, and reference lines matching realistic.tex (0.5 = darkgray176 @ 0.5,
1.0 = gray @ 0.6).

Run: python -m analysis.stories.make_agg_tex  ->  analysis/stories/stories_agg.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "stories.csv"
OUT = ROOT / "stories_agg.tex"
CI_LEVEL = 0.9

METHODS = [("filtering", "Filtering"), ("grmoe", "GRAM"), ("lora", "FT-LoRA"), ("coreftaux", "FT-Full"),
           ("demix", "Demix"), ("maxent", "MaxEnt")]
CLASSES = [("core", "Core $\\uparrow$", "steelblue31119180"),
           ("retain", "Retain $\\uparrow$", "forestgreen4416044"),
           ("forget", "Forget $\\downarrow$", "crimson2143940"),
           ("elicited_forget", "Elicit $\\downarrow$", "darkorange25512714")]
# bar offsets within a group + half width (matches matplotlib 4-bar layout).
OFFSETS = [-0.29625, -0.07875, 0.13875, 0.35625]
HW = 0.09875
CAP = 0.045


def aggregate(df):
    """(method, label_class) -> (mean, lo, hi)."""
    sm = (df.groupby(["method", "label_class", "seed"])["compute_ratio"]
            .mean().reset_index())
    agg = (sm.groupby(["method", "label_class"])["compute_ratio"]
             .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    tcrit = np.where(agg["count"] > 1, st.t.ppf(1 - (1 - CI_LEVEL) / 2,
                                                agg["count"] - 1), 0.0)
    agg["ci"] = tcrit * sem
    out = {}
    for _, r in agg.iterrows():
        out[(r["method"], r["label_class"])] = (r["mean"], r["mean"] - r["ci"],
                                                r["mean"] + r["ci"])
    return out


def main():
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    df = df[df["method"].isin([m for m, _ in METHODS])]
    vals = aggregate(df)

    L = []
    L.append(r"% stories_agg.tex — pgfplots port of stories_agg.png.")
    L.append(r"\begin{tikzpicture}")
    L.append(r"  \pgfplotsset{legend image code/.code={\draw[#1] (0cm,-0.08cm) rectangle (0.22cm,0.08cm);}}")
    L.append("")
    L.append(r"  \definecolor{steelblue31119180}{RGB}{31,119,180}")
    L.append(r"  \definecolor{forestgreen4416044}{RGB}{44,160,44}")
    L.append(r"  \definecolor{crimson2143940}{RGB}{214,39,40}")
    L.append(r"  \definecolor{darkorange25512714}{RGB}{255,127,14}")
    L.append(r"  \definecolor{darkgray176}{RGB}{176,176,176}")
    L.append(r"  \definecolor{gray}{RGB}{128,128,128}")
    L.append(r"  \definecolor{lightgray204}{RGB}{204,204,204}")
    L.append("")
    L.append(r"  \begin{axis}[")
    L.append(r"  width=0.92\linewidth, height=0.13\linewidth, scale only axis=true,")
    L.append(r"  tick align=outside, tick pos=left,")
    L.append(r"  ymin=0, ymax=1.3,")
    L.append(r"  ytick={0,0.5,1.0},")
    L.append(r"  yticklabel style={font=\footnotesize, /pgf/number format/.cd, fixed, fixed zerofill, precision=1},")
    L.append(r"  ylabel={Compute Ratio}, ylabel style={font=\footnotesize},")
    nm = len(METHODS)
    xmax = nm - 1 + 0.6
    L.append(f"  xmin=-0.6, xmax={xmax:.1f},")
    L.append(r"  xtick={" + ",".join(str(i) for i in range(nm)) + "},")
    L.append(r"  xticklabels={" + ",".join("{%s}" % d for _, d in METHODS) + "},")
    L.append(r"  xticklabel style={font=\footnotesize},")
    L.append(r"  xtick style={color=black}, ytick style={color=black},")
    L.append(r"  legend columns=-1,")
    L.append(r"  legend style={font=\footnotesize, draw=lightgray204, at={(0.5,1.04)}, "
             r"anchor=south, /tikz/every even column/.append style={column sep=0.7em}},")
    L.append(r"  ]")
    # legend
    for _, disp, color in CLASSES:
        L.append(f"  \\addlegendimage{{area legend,draw=none,fill={color}}}")
        L.append(f"  \\addlegendentry{{{disp}}}")
    # reference lines (drawn first, behind bars). Newline-separated rows so the
    # paper's pgfplots reads them as inline data, not as a filename.
    L.append(r"  \addplot [darkgray176, opacity=0.5, line width=0.5pt, forget plot]"
             f"\n  table {{%\n  -0.6 0.5\n  {xmax:.1f} 0.5\n  }};")
    L.append(r"  \addplot [gray, opacity=0.6, line width=0.7pt, forget plot]"
             f"\n  table {{%\n  -0.6 1\n  {xmax:.1f} 1\n  }};")
    # bars + error bars + value labels
    for gi, (m, _) in enumerate(METHODS):
        for off, (lc, _, color) in zip(OFFSETS, CLASSES):
            key = (m, lc)
            if key not in vals:
                continue
            mean, lo, hi = vals[key]
            xc = gi + off
            x1, x2 = xc - HW, xc + HW
            L.append(f"  \\draw[draw=none,fill={color}] "
                     f"(axis cs:{x1:.4f},0) rectangle (axis cs:{x2:.4f},{mean:.5f});")
            if hi - lo > 1e-9:
                L.append(f"  \\draw[black,line width=0.5pt] "
                         f"(axis cs:{xc:.4f},{lo:.5f})--(axis cs:{xc:.4f},{hi:.5f});")
                for yv in (lo, hi):
                    L.append(f"  \\draw[black,line width=0.5pt] "
                             f"(axis cs:{xc-CAP:.4f},{yv:.5f})--(axis cs:{xc+CAP:.4f},{yv:.5f});")
            L.append(f"  \\node[font=\\scriptsize, anchor=south, inner sep=1.2pt, "
                     f"fill=white, fill opacity=1, text opacity=1] at "
                     f"(axis cs:{xc:.4f},{max(mean, hi):.5f}) {{{mean:.2f}}};")
    L.append(r"  \end{axis}")
    L.append(r"\end{tikzpicture}")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

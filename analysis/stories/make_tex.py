#!/usr/bin/env python3
"""
Emit a pgfplots/TikZ version of stories_modules.png (analysis/stories/plot.py).

Two stacked panes (groupplot 1x2): top = Data Filtering (5 Models), bottom =
GRAM (Ours, 1 Model). x = retain/module config, 5 data-label bars each;
non-retained labels are faded + slash-hatched. 90% t-CI error bars across the
three seeds. Light gridlines + darker solid line at 1.0, matching the repo's
other .tex figures.

Run:  python3 -m analysis.stories.make_tex   ->  analysis/stories/stories_modules.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "stories.csv"
OUT = ROOT / "stories_modules.tex"

CI_LEVEL = 0.9
Y_MIN, Y_MAX = 0.5, 1.1
YTICKS = [0.6, 0.8, 1.0]

ALL_LABELS = ["core", "a-deadline-or-time-limit", "alien-encounters",
              "bygone-eras", "cultural-traditions"]
LABEL_DISPLAY = {"core": "Core", "a-deadline-or-time-limit": "Deadlines",
                 "alien-encounters": "Aliens", "bygone-eras": "Eras",
                 "cultural-traditions": "Cultures"}
LABEL_COLOR = {"core": "steelblue31119180",
               "a-deadline-or-time-limit": "darkorange25512714",
               "alien-encounters": "forestgreen4416044",
               "bygone-eras": "crimson2143940",
               "cultural-traditions": "mediumpurple148103189"}
RETAIN_ORDER = ["core", "a-deadline-or-time-limit+core", "alien-encounters+core",
                "bygone-eras+core", "core+cultural-traditions"]
RETAIN_ADD = {"core": None, "a-deadline-or-time-limit+core": "a-deadline-or-time-limit",
              "alien-encounters+core": "alien-encounters",
              "bygone-eras+core": "bygone-eras",
              "core+cultural-traditions": "cultural-traditions"}

# Bar geometry within a group centred on integer x (matches matplotlib layout).
OFFSETS = [-0.308, -0.134, 0.04, 0.214, 0.388]
HALF_W = 0.077
CAP = 0.04

PANES = [("filtering", "Data Filtering (5 Models)", "Data"),
         ("grmoe", "GRAM (Ours, 1 Model)", "Module")]


def aggregate(df: pd.DataFrame) -> dict:
    """(method, retained, data_label) -> (mean, lo, hi)."""
    seed_means = (df.groupby(["method", "retained", "data_label", "seed"])
                    ["compute_ratio"].mean().reset_index())
    agg = (seed_means.groupby(["method", "retained", "data_label"])["compute_ratio"]
           .agg(["mean", "std", "count"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    sem = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    alpha = 1.0 - CI_LEVEL
    tcrit = np.where(agg["count"] > 1, st.t.ppf(1 - alpha / 2, agg["count"] - 1), 0.0)
    agg["ci"] = tcrit * sem
    out = {}
    for _, r in agg.iterrows():
        out[(r["method"], r["retained"], r["data_label"])] = (
            r["mean"], r["mean"] - r["ci"], r["mean"] + r["ci"])
    return out


def xlabels(noun: str) -> list[str]:
    out = []
    for key in RETAIN_ORDER:
        add = RETAIN_ADD[key]
        out.append(f"Core {noun} Only" if add is None
                   else f"+ {LABEL_DISPLAY[add]} {noun}")
    return out


def pane_body(method: str, noun: str, vals: dict) -> str:
    L = []
    # bars + hatches
    for gi, ret in enumerate(RETAIN_ORDER):
        retained = set(ret.split("+"))
        for lab, off in zip(ALL_LABELS, OFFSETS):
            key = (method, ret, lab)
            if key not in vals:
                continue
            mean, lo, hi = vals[key]
            xc = gi + off
            x1, x2 = xc - HALF_W, xc + HALF_W
            color = LABEL_COLOR[lab]
            faded = lab not in retained
            if faded:
                L.append(f"  \\draw[draw=none,fill={color},fill opacity=0.2] "
                         f"(axis cs:{x1:.4f},0) rectangle (axis cs:{x2:.4f},{mean:.6f});")
                L.append(f"  \\draw[draw=none,pattern=diagonal lines wide,"
                         f"pattern color=gray,opacity=0.5] "
                         f"(axis cs:{x1:.4f},0) rectangle (axis cs:{x2:.4f},{mean:.6f});")
            else:
                L.append(f"  \\draw[draw=none,fill={color}] "
                         f"(axis cs:{x1:.4f},0) rectangle (axis cs:{x2:.4f},{mean:.6f});")
    # error bars (drawn on top)
    for gi, ret in enumerate(RETAIN_ORDER):
        for lab, off in zip(ALL_LABELS, OFFSETS):
            key = (method, ret, lab)
            if key not in vals:
                continue
            _, lo, hi = vals[key]
            xc = gi + off
            L.append(f"  \\draw[black,line width=0.5pt] "
                     f"(axis cs:{xc:.4f},{lo:.6f})--(axis cs:{xc:.4f},{hi:.6f});")
            L.append(f"  \\draw[black,line width=0.5pt] "
                     f"(axis cs:{xc-CAP:.4f},{lo:.6f})--(axis cs:{xc+CAP:.4f},{lo:.6f});")
            L.append(f"  \\draw[black,line width=0.5pt] "
                     f"(axis cs:{xc-CAP:.4f},{hi:.6f})--(axis cs:{xc+CAP:.4f},{hi:.6f});")
    # darker reference line at 1.0
    L.append(r"  \addplot [line width=0.28pt, gray, opacity=0.6, forget plot] table {%")
    L.append("  -0.6 1")
    L.append("  4.6 1")
    L.append("  };")
    return "\n".join(L)


def main() -> None:
    df = pd.read_csv(CSV)
    df["compute_ratio"] = pd.to_numeric(df["compute_ratio"], errors="coerce")
    df = df.dropna(subset=["compute_ratio"])
    df = df[df["label_class"] != "elicited_forget"]
    df = df[df["method"].isin([p[0] for p in PANES])]
    df = df[df["retained"].isin(RETAIN_ORDER)]
    vals = aggregate(df)

    legend_imgs = []
    for lab in ALL_LABELS:
        legend_imgs.append(f"  \\addlegendimage{{area legend,draw=none,fill={LABEL_COLOR[lab]}}}")
        legend_imgs.append(f"  \\addlegendentry{{{LABEL_DISPLAY[lab]}}}")
    legend_imgs.append(r"  \addlegendimage{area legend,draw=none,fill=gray}")
    legend_imgs.append(r"  \addlegendentry{Retain $\uparrow$}")
    legend_imgs.append(r"  \addlegendimage{area legend,draw=none,fill=lightgray204,"
                       r"pattern=diagonal lines wide,pattern color=gray}")
    legend_imgs.append(r"  \addlegendentry{Forget $\downarrow$}")
    legend_block = "\n".join(legend_imgs)

    def axis_opts(title, noun, with_legend):
        xt = ",".join(f"{{{t}}}" for t in xlabels(noun))
        opts = [
            "  tick align=outside, tick pos=left,",
            "  ymajorgrids,",
            "  y grid style={darkgray176, opacity=0.5, line width=0.28pt},",
            f"  ymin={Y_MIN}, ymax={Y_MAX},",
            f"  ytick={{{','.join(str(t) for t in YTICKS)}}},",
            r"  yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=1},",
            "  ylabel={Compute\\\\Ratio}, ylabel style={align=center, font=\\footnotesize},",
            "  xmin=-0.6, xmax=4.6,",
            "  xtick={0,1,2,3,4},",
            f"  xticklabels={{{xt}}},",
            "  xticklabel style={font=\\footnotesize},",
            "  ytick style={color=black}, xtick style={color=black},",
            f"  title={{\\textbf{{{title}}}}}, title style={{font=\\footnotesize, yshift=-0.6em}},",
        ]
        if with_legend:
            opts += [
                "  legend columns=-1,",
                "  legend style={font=\\scriptsize, draw=lightgray204, at={(0.5,1.55)}, "
                "anchor=south, /tikz/every even column/.append style={column sep=0.6em}},",
            ]
        return "\n".join(opts)

    parts = []
    parts.append(r"% stories_modules.tex — pgfplots port of stories_modules.png.")
    parts.append(r"\begin{tikzpicture}")
    parts.append(r"  \pgfplotsset{legend image code/.code={\draw[#1] (0cm,-0.07cm) rectangle (0.25cm,0.07cm);}}")
    parts.append("")
    parts.append(r"  \definecolor{steelblue31119180}{RGB}{31,119,180}")
    parts.append(r"  \definecolor{darkorange25512714}{RGB}{255,127,14}")
    parts.append(r"  \definecolor{forestgreen4416044}{RGB}{44,160,44}")
    parts.append(r"  \definecolor{crimson2143940}{RGB}{214,39,40}")
    parts.append(r"  \definecolor{mediumpurple148103189}{RGB}{148,103,189}")
    parts.append(r"  \definecolor{darkgray176}{RGB}{176,176,176}")
    parts.append(r"  \definecolor{gray}{RGB}{128,128,128}")
    parts.append(r"  \definecolor{lightgray204}{RGB}{204,204,204}")
    parts.append("")
    parts.append(r"  \begin{groupplot}[group style={group size=1 by 2, vertical sep=1.2cm}, "
                 r"width=0.86\linewidth, height=0.088\linewidth, scale only axis=true]")
    for i, (method, title, noun) in enumerate(PANES):
        parts.append(f"  \\nextgroupplot[")
        parts.append(axis_opts(title, noun, with_legend=(i == 0)))
        parts.append("  ]")
        if i == 0:
            parts.append(legend_block)
        parts.append(pane_body(method, noun, vals))
    parts.append(r"  \end{groupplot}")
    parts.append(r"\end{tikzpicture}")
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

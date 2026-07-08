#!/usr/bin/env python3
"""
Compile the 200M uniform-vs-heterogeneous accumulation results into a CSV with
compute ratios.

Single seed (seed_5), 200M, comparing uniform and heterogeneous gradient
accumulation for the baseline, GRAM, and FT-LoRA. All results are read from
``results/accumulation/{base,grmoe,lora}/200M/seed_5/acc_mode_*``. For GRAM and
FT-LoRA only the ``_factor_1.0`` runs are used.

Every ``do_eval`` row in the applicable ``stats.jsonl`` files is emitted: the
baseline has a single retain config (all five labels) and five eval rows; GRAM
and FT-LoRA each have five retain configs x five labels = 25 eval rows per
acc_mode (the retained-domain and forgotten-domain evals). Each row is tagged
with its ``label_class`` (core / retain / forget / elicited_forget).

Every compute ratio is normalized against the **heterogeneous baseline**: the
power-law learning curve is fit to
``base/200M/seed_5/acc_mode_heterogeneous/baseline/losses.pkl`` (per data_label),
and the denominator is ``step_equiv`` of that heterogeneous baseline's own final
loss. By construction the baseline-heterogeneous rows are therefore 1.0 for
every data_label, and forgotten-domain rows are measured against the same fully
trained reference (low compute ratio = strong removal).

  CR[row] = step_equiv(row.loss, curve[row.data_label])
            / step_equiv(het_baseline_final[row.data_label], curve[row.data_label])
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parents[2]
sys.path.insert(0, str(EXP_ROOT))
from analysis.common.compile import (  # noqa: E402
    fit_power, label_class, load_val_losses, step_equiv,
)

ACC_ROOT = EXP_ROOT / "results" / "accumulation"
SIZE, SEED = "200M", "seed_5"
OUT = HERE / "accumulation.csv"

LABELS = ["core", "papers-biology", "code-lisp", "papers-cyber", "papers-nuclear"]
MODES = ["uniform", "heterogeneous"]

# (method label, results subdir, acc_mode suffix)
METHODS = [
    ("baseline", "base",  ""),
    ("grmoe",    "grmoe", "_factor_1.0"),
    ("lora",     "lora",  "_factor_1.0"),
]


def eval_rows(stats_path: Path):
    """Yield every non-elicited (we keep elicited too) do_eval record."""
    for line in open(stats_path):
        r = json.loads(line)
        if r.get("function") != "do_eval":
            continue
        if r["data_label"] not in LABELS:
            continue
        yield r


def main() -> None:
    # Reference: het-baseline power-law curve + its own final-loss denominator.
    ref_dir = ACC_ROOT / "base" / SIZE / SEED / "acc_mode_heterogeneous"
    curves = {lab: fit_power(xs, ys)
              for lab, (xs, ys) in load_val_losses(ref_dir / "baseline" / "losses.pkl").items()
              if lab in LABELS}
    ref_finals = {r["data_label"]: r["loss"]
                  for r in eval_rows(ref_dir / "stats.jsonl")}
    ref_se = {lab: step_equiv(ref_finals[lab], *curves[lab])
              for lab in curves if lab in ref_finals}

    rows: list[dict] = []
    for method, subdir, suffix in METHODS:
        for mode in MODES:
            sp = ACC_ROOT / subdir / SIZE / SEED / f"acc_mode_{mode}{suffix}" / "stats.jsonl"
            if not sp.exists():
                print(f"  skip {method}/{mode}: missing {sp}")
                continue
            for r in eval_rows(sp):
                lab = r["data_label"]
                if lab not in ref_se:
                    continue
                retained = r.get("retained", []) or []
                elicited = bool(r.get("elicited", False))
                cr = step_equiv(r["loss"], *curves[lab]) / ref_se[lab]
                rows.append({
                    "method": method,
                    "model_size": SIZE,
                    "seed": SEED,
                    "acc_mode": mode,
                    "retained": "+".join(sorted(retained)),
                    "data_label": lab,
                    "label_class": label_class(lab, retained, elicited),
                    "elicited": elicited,
                    "loss": r["loss"],
                    "compute_ratio": cr,
                    "source": sp.relative_to(ACC_ROOT.parent).as_posix(),
                })

    rows.sort(key=lambda r: (METHODS.index(next(m for m in METHODS if m[0] == r["method"])),
                             r["acc_mode"], r["retained"], r["data_label"]))
    fieldnames = ["method", "model_size", "seed", "acc_mode", "retained",
                  "data_label", "label_class", "elicited", "loss", "compute_ratio",
                  "source"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()

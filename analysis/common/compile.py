#!/usr/bin/env python3
"""
Compute-ratio (CR) machinery shared by every experiment's ``compile.py``.

CR expresses a model's loss as the *fraction of baseline training compute* the
baseline itself would need to reach that same loss --- so 1.0 means "as good as
the fully-trained baseline", independent of dataset difficulty or model scale.

The pipeline, applied per dataset (``data_label``) and per seed:

  1. During baseline training we periodically record validation loss, giving a
     learning curve of (step, loss) points.                  [load_val_losses]
  2. Fit a power law  L(s) = A * (s + x0) ** -alpha  to that curve, in log
     space (the standard way to fit a power law).                 [fit_power]
  3. Invert it: a loss maps back to the baseline step that first reached it,
     L^{-1}(loss) --- the "step-equivalent". Past the end of training the curve
     is extended by its tangent, so a better-than-baseline loss still gets a
     finite step-equivalent (and CR > 1).                        [step_equiv]
  4. CR = L^{-1}(model_loss) / L^{-1}(baseline_final_loss). The denominator is
     ~the baseline's training length, so the baseline scores ~1.  [compute_cr]

Normalization is **per seed**: a run trained with seed n is divided by the
seed-n baseline. Each experiment's own ``compile.py`` decides which results
directory the baseline lives in and hands it to ``build_baseline``; this module
owns only the math.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def model_size_sort_key(s: str) -> float:
    """Sort key turning '50M' / '800M' / '2B' into a raw parameter count."""
    if s.endswith("M"):
        return float(s[:-1]) * 1e6
    if s.endswith("B"):
        return float(s[:-1]) * 1e9
    return float("inf")


def label_class(data_label: str, retained, elicited: bool) -> str:
    """Bucket a (data_label, retained, elicited) row into core/retain/forget."""
    if data_label == "core":
        return "core"
    if data_label in retained:               # an auxiliary capability we keep
        return "retain"
    return "elicited_forget" if elicited else "forget"


# ---------------------------------------------------------------------------
# The power-law learning curve: fit (step 2) and inverse (step 3)
# ---------------------------------------------------------------------------

def fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit the learning curve  y ~= A * (x + x0) ** -alpha  (asymptote 0).

    Returns ``(A, alpha, x0)``. The curve is strictly decreasing (alpha > 0),
    so it is always invertible. There is no additive floor ``c``: a floored
    form ``... + c`` would cap the smallest reachable loss at ``c`` and leave
    better losses undefined.

    We fit in **log space** --- the standard linearization for a power law ---
    by ordinary least squares on the residual
    ``log(A) - alpha*log(x + x0) - log(y)``. Working in log space weights
    *relative* error, so the converged low-loss tail (where CR is read off) is
    fit as faithfully as the high-loss early steps; an unweighted fit in loss
    space would be dominated by the large early residuals.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    eps = 1e-6
    # Bounds: A > 0; a sane exponent range; and x0 large enough that
    # (x + x0) > 0 everywhere (so log is defined) but not unboundedly large.
    lb = [1e-10, 1e-3, -float(x.min()) + eps]
    ub = [1e8, 5.0, float(x.max()) + 10 * dx]
    p0 = [max(float(y[0]), 1e-3), 0.3, min(100.0, ub[2] * 0.9)]
    log_y = np.log(y)

    def resid(p):
        A, alpha, x0 = p
        return (np.log(A) - alpha * np.log(x + x0)) - log_y

    res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=40000)
    return tuple(res.x)


def step_equiv(loss: float, A: float, alpha: float, x0: float) -> float:
    """Map a loss to the baseline step that first reached it: the exact inverse
    of ``fit_power``,  s = (A / loss) ** (1/alpha) - x0.

    The fitted power law is used as-is everywhere, including losses below the
    baseline's final loss --- a model that beats the baseline simply lands at a
    step beyond the end of training (step_equiv > s_max, i.e. CR > 1).
    """
    eps = 1e-12
    s = (A / max(loss, eps)) ** (1.0 / alpha) - x0
    return max(float(s), 1e-3)                          # floor for numerical safety


# ---------------------------------------------------------------------------
# Building a baseline reference (load curves + reference losses)
# ---------------------------------------------------------------------------

def load_val_losses(pkl_path: Path) -> dict:
    """Return ``{data_label: (steps, losses)}`` over the full baseline run.

    ``losses.pkl`` holds ``{"train": {...}, "val": {...}}``; we use the val
    split (falling back to the top-level dict if there is no ``"val"`` key).
    Each value is an (N, 2) array of (step, loss) rows.
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    d = d.get("val", d)
    out = {}
    for k, v in d.items():
        arr = np.asarray(v, float)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 1:
            out[k] = (arr[:, 0], arr[:, 1])
    return out


def build_baseline(ts_dir: Path) -> dict | None:
    """Turn a baseline run directory into a reusable CR reference.

    Reads the learning curves from ``ts_dir/baseline/losses.pkl`` and the
    baseline's own final eval losses from ``ts_dir/stats.jsonl`` (the
    non-elicited ``baseline`` stage rows). Returns a dict::

        curves     {data_label: (A, alpha, x0)}   fitted learning curves
        ref_se     {data_label: step_equiv(baseline_loss)}   CR denominator
        ref_loss   {data_label: baseline_loss}              ppl_ratio reference
        num_params int | None                               base model size
        ts         ts_dir.name

    Returns ``None`` if the pickle or stats file is missing.
    """
    pkl = ts_dir / "baseline" / "losses.pkl"
    stats = ts_dir / "stats.jsonl"
    if not (pkl.exists() and stats.exists()):
        return None

    # Step 2: fit one power-law curve per dataset.
    raw = load_val_losses(pkl)            # {label: (steps, losses)} -- kept for pooling
    curves = {lab: fit_power(xs, ys) for lab, (xs, ys) in raw.items()}

    # The CR denominator: invert the baseline's *own* final loss through its
    # curve (~ its total training length, so the baseline scores CR ~ 1).
    ref_se: dict[str, float] = {}
    ref_loss: dict[str, float] = {}
    with open(stats) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("elicited"):
                continue
            if (rec.get("stage") or {}).get("name") != "baseline":
                continue
            lab = rec["data_label"]
            if lab in curves:
                ref_loss[lab] = rec["loss"]
                ref_se[lab] = step_equiv(rec["loss"], *curves[lab])

    num_params = None
    cfg_path = ts_dir / "config.json"
    if cfg_path.exists():
        try:
            num_params = int(json.load(open(cfg_path))["run"]["num_base_params"])
        except (KeyError, json.JSONDecodeError):
            pass

    return {
        "curves": curves,
        "ref_se": ref_se,
        "ref_loss": ref_loss,
        "raw": raw,
        "num_params": num_params,
        "ts": ts_dir.name,
    }


def pool_baselines(bases: dict, group_fn=lambda k: None) -> dict:
    """Hybrid CR reference: pooled curve + common (mean) denominator.

    Two changes from the naive per-seed baseline, both aimed at the inflated CR
    confidence intervals:

    1. **Pooled curve.** Refit ONE shared power-law per data_label by pooling
       every seed's learning-curve points (within each group from ``group_fn``).
       Three separate single-run fits add fitting noise; one pooled fit is
       stable and robust to an outlier baseline run.
    2. **Common denominator.** Use a single CR denominator per data_label: the
       **mean across seeds of each baseline's step-equivalent** (its own final
       loss inverted through the pooled curve). The baseline's training compute
       is a fixed quantity; the per-seed step-equivalents are noisy (tail-
       amplified) estimates of it, so averaging them gives a stable reference
       and keeps the baseline at CR ~ 1 on average. Reported CR error bars then
       reflect variation of the *method* across its training seeds against this
       fixed reference (they do not include baseline-reference uncertainty).

    ``bases`` maps key -> baseline dict (from ``build_baseline``); every key in
    a group gets the same pooled ``curves`` and the same common ``ref_se``.
    ``group_fn(key)`` chooses which baselines share a curve/denominator
    (default: all); for multi-size experiments pass ``group_fn=lambda k: k[0]``
    to pool per model size.
    """
    groups: dict = {}
    for k, b in bases.items():
        if b is not None:
            groups.setdefault(group_fn(k), {})[k] = b

    out = dict(bases)
    for members in groups.values():
        # 1. pooled curve per data_label
        pooled_x: dict = {}
        pooled_y: dict = {}
        for b in members.values():
            for lab, (xs, ys) in b.get("raw", {}).items():
                pooled_x.setdefault(lab, []).append(np.asarray(xs, float))
                pooled_y.setdefault(lab, []).append(np.asarray(ys, float))
        curves = {lab: fit_power(np.concatenate(pooled_x[lab]),
                                 np.concatenate(pooled_y[lab]))
                  for lab in pooled_x}
        # 2. common denominator = mean over seeds of each baseline's step_equiv
        se_by_lab: dict = {}
        for b in members.values():
            for lab, loss in b.get("ref_loss", {}).items():
                if lab in curves:
                    se_by_lab.setdefault(lab, []).append(step_equiv(loss, *curves[lab]))
        common_ref_se = {lab: float(np.mean(ses)) for lab, ses in se_by_lab.items()}
        for k, b in members.items():
            out[k] = {**b, "curves": curves, "ref_se": common_ref_se}
    return out


def latest_baseline_ts(seed_dir: Path, *, exclude_test: bool = False) -> Path | None:
    """Newest timestamp dir under ``seed_dir`` that carries a usable baseline.

    "Usable" = has both ``stats.jsonl`` and ``baseline/losses.pkl``; "newest" =
    lexicographically-largest name (the timestamps sort chronologically).
    """
    cands = [
        d for d in seed_dir.iterdir()
        if d.is_dir()
        and (not exclude_test or "_test" not in d.name)
        and (d / "stats.jsonl").exists()
        and (d / "baseline" / "losses.pkl").exists()
    ]
    return max(cands, key=lambda d: d.name) if cands else None


# ---------------------------------------------------------------------------
# The ratios (step 4)
# ---------------------------------------------------------------------------

def compute_cr(loss: float, data_label: str, base: dict) -> float | None:
    """Compute ratio = step_equiv(model loss) / step_equiv(baseline loss).

    ``base`` is a dict from ``build_baseline``. Returns ``None`` if this dataset
    has no fitted curve / reference in ``base``.
    """
    if data_label not in base["curves"] or data_label not in base["ref_se"]:
        return None
    se = step_equiv(loss, *base["curves"][data_label])
    ref = base["ref_se"][data_label]
    return se / ref if ref else None


def compute_ppl_ratio(loss: float, data_label: str, base: dict) -> float | None:
    """Perplexity ratio: exp(model loss) / exp(baseline loss)."""
    ref_loss = base.get("ref_loss", {})
    if data_label not in ref_loss:
        return None
    return float(np.exp(loss) / np.exp(ref_loss[data_label]))

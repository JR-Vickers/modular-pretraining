#!/usr/bin/env python3
"""Shared analysis utilities for optimize plotting scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analysis.common.load_data import load_stats_jsonl, load_val_losses

LOG_SCALE_PARAMS = {"lr", "aux_lr_factor"}
BS_PARAM_NAMES = {"batch_size", "effective_batch_size", "eff_bs"}


# ---------------------------------------------------------------------------
# CEL scoring
# ---------------------------------------------------------------------------

def cel_score(
    df: pd.DataFrame,
    core_weight: float = 0.5,
    core_only: bool = False,
    ablated_core_only: bool = False,
) -> float:
    """Compute CEL score from eval rows.

    Default score is a weighted mean of:
      - core loss (data_label == "core")
      - retained non-core loss (data_label in retained and data_label != "core")

    Rows marked as elicited or finetuned are excluded.

    If ablated_core_only, return the core loss from the fully-ablated eval
    (retained == ['core'], data_label == 'core') — a single row.
    """
    if df.empty:
        return float("inf")

    is_eval = df["function"] == "do_eval"
    not_elicited = (
        ~df.get("elicited", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
    )
    no_finetune = df["finetune"].isna()
    evals = df[is_eval & not_elicited & no_finetune]

    if evals.empty:
        return float("inf")

    if ablated_core_only:
        mask = (evals["data_label"] == "core") & evals["retained"].apply(
            lambda r: isinstance(r, list) and r == ["core"]
        )
        losses = evals.loc[mask, "loss"].values
        if len(losses) == 0:
            return float("inf")
        return float(losses.mean())

    core_losses = evals.loc[evals["data_label"] == "core", "loss"].values
    if core_only:
        if len(core_losses) == 0:
            return float("inf")
        return float(core_losses.mean())

    retain_mask = evals.apply(
        lambda row: isinstance(row.get("retained"), list)
        and row["data_label"] in row["retained"]
        and row["data_label"] != "core",
        axis=1,
    )
    retain_losses = evals.loc[retain_mask, "loss"].values

    if len(core_losses) == 0 and len(retain_losses) == 0:
        return float("inf")
    if len(core_losses) == 0:
        return float(retain_losses.mean())
    if len(retain_losses) == 0:
        return float(core_losses.mean())

    return core_weight * float(core_losses.mean()) + (1 - core_weight) * float(
        retain_losses.mean()
    )


# ---------------------------------------------------------------------------
# Trial loading
# ---------------------------------------------------------------------------

def _extract_bs(run_cfg: dict) -> int:
    """Use batch size from config (prefer effective if available)."""
    if "effective_batch_size" in run_cfg:
        return int(run_cfg["effective_batch_size"])
    if "batch_size" in run_cfg:
        return int(run_cfg["batch_size"])
    raise KeyError("Missing batch size field")


def _infer_model_seed(cfg: dict) -> tuple[str, int]:
    """Infer (model_label, seed) from config metadata only."""
    run_cfg = cfg["run"]
    num_base_params = int(run_cfg["num_base_params"])
    model_label = f"{int(round(num_base_params / 10_000_000) * 10)}M"
    seed = int(run_cfg["seed"])
    return model_label, seed


def load_trials(
    results_root: Path,
    core_only: bool = False,
    ablated_core_only: bool = False,
) -> dict[str, dict[int, list[tuple[float, int, float]]]]:
    """Recursively scan for config/stats pairs and return per-model/seed trials.

    Returns ``{model_label: {seed: [(lr, bs, score), ...]}}``.
    """
    data: dict[str, dict[int, list]] = {}

    for config_fp in sorted(results_root.rglob("config.json")):
        if "_OLD" in str(config_fp):
            continue
        stats_fp = config_fp.parent / "stats.jsonl"
        if not stats_fp.exists():
            continue

        try:
            with open(config_fp) as f:
                cfg = json.load(f)
            model_label, seed = _infer_model_seed(cfg)
            lr = float(cfg["stages"][0]["lr"])
            bs = _extract_bs(cfg["run"])
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            continue

        try:
            df = pd.read_json(stats_fp, lines=True)
        except Exception:
            continue
        score = cel_score(df, core_only=core_only, ablated_core_only=ablated_core_only)
        if score == float("inf"):
            continue

        data.setdefault(model_label, {}).setdefault(seed, []).append(
            (lr, bs, score)
        )

    return data


# ---------------------------------------------------------------------------
# Power-law fitting & compute ratio
# ---------------------------------------------------------------------------

def fit_power_no_floor(x, y):
    """Fit loss(step) = A * (step + x0)^(-alpha) with floor fixed at zero."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y_mon = np.minimum.accumulate(y)
    A0 = max(float(y_mon[0]), 1e-3)
    alpha0 = 0.3
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    x_min = float(x.min())
    eps = 1e-6
    lb = [1e-10, 1e-3, -x_min + eps]
    ub = [1e8, 5.0, x.max() + 10 * dx]
    x0_0 = min(100.0, ub[2] * 0.9)
    p0 = [A0, alpha0, x0_0]
    def resid(p):
        A, alpha, x0 = p
        return A * np.power(x + x0, -alpha) - y
    res = least_squares(resid, p0, bounds=(lb, ub),
                        loss='soft_l1', f_scale=0.01, max_nfev=40000)
    A, alpha, x0 = res.x
    return (A, alpha, 0.0, x0)  # c=0 floor


def step_equiv_extrapolated(loss, A, alpha, c, x0, s_max):
    """Compute step-equivalent via inverse power law (c=0 floor).

    For losses above f(s_max), uses the standard inverse.
    For losses below f(s_max), extrapolates linearly using the tangent.
    """
    loss = np.asarray(loss, float)
    f_smax = c + A * np.power(s_max + x0, -alpha)

    eps = 1e-12
    denom = np.maximum(loss - c, eps)
    base = np.maximum(A / denom, eps)
    normal = np.power(base, 1.0 / alpha) - x0
    normal = np.maximum(normal, 1e-3)

    slope_mag = alpha * A * np.power(s_max + x0, -(alpha + 1))
    extrapolated = s_max + np.maximum(f_smax - loss, 0.0) / slope_mag

    result = np.where(loss >= f_smax, normal, extrapolated)
    return float(result) if result.ndim == 0 else result


def fast_add_metrics(
    df: pd.DataFrame,
    curves: dict,
    max_steps: dict,
) -> pd.DataFrame:
    """Add ppl, step_equiv, and compute_ratio columns using pre-computed curves."""
    df = df.copy()
    df["ppl"] = np.exp(df["loss"])

    def _step_eq(label, loss):
        if pd.isna(label) or pd.isna(loss) or label not in curves:
            return np.nan
        A, alpha, c, x0 = curves[label]
        return step_equiv_extrapolated(loss, A, alpha, c, x0, max_steps[label])

    df["step_equiv"] = df.apply(
        lambda r: _step_eq(r["data_label"], r["loss"]), axis=1,
    )

    not_elicited = (
        ~df.get("elicited", pd.Series(False, index=df.index))
        .fillna(False).astype(bool)
    )
    baseline_select = (df["name"] == "baseline") & not_elicited
    baselines = (
        df[baseline_select][["data_label", "seed", "step_equiv"]]
        .groupby(["seed", "data_label"], dropna=False)
        .mean()
        .to_dict("index")
    )

    def _cr(row):
        label, seed, val = row["data_label"], row["seed"], row["step_equiv"]
        if pd.isna(label) or pd.isna(val):
            return np.nan
        ref = baselines.get((seed, label), {}).get("step_equiv", np.nan)
        if pd.isna(ref) or ref == 0:
            return np.nan
        return val / ref

    df["compute_ratio"] = df.apply(_cr, axis=1)
    return df


# ---------------------------------------------------------------------------
# Baseline / filtering loading & VSF scoring
# ---------------------------------------------------------------------------

def fit_baseline_curves(losses_pkl: Path) -> tuple[dict, dict]:
    """Fit power-law curves (floor=0) once.  Returns ``(curves, max_steps)``."""
    losses = load_val_losses(losses_pkl)
    curves, max_steps = {}, {}
    for lab in losses:
        x_b, y_b = losses[lab]
        max_steps[lab] = float(x_b[-1])
        curves[lab] = fit_power_no_floor(x_b, y_b)
    return curves, max_steps


def find_latest_run(parent: Path) -> Path | None:
    """Return the latest timestamped run directory under *parent*."""
    candidates = sorted(
        (d for d in parent.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_baseline(scaling_root: Path, model_size: str) -> tuple[pd.DataFrame, Path]:
    """Return ``(baseline_df, losses_pkl_path)`` for *model_size*."""
    seed_dir = scaling_root / model_size / "seed_1"
    run_dir = find_latest_run(seed_dir)
    if run_dir is None:
        raise FileNotFoundError(f"No baseline run found under {seed_dir}")
    stats_path = run_dir / "stats.jsonl"
    losses_pkl = run_dir / "baseline" / "losses.pkl"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}")
    if not losses_pkl.exists():
        raise FileNotFoundError(f"Missing {losses_pkl}")
    return load_stats_jsonl(stats_path, verbose=False), losses_pkl


def load_filtering_stats(
    filtered_root: Path, model_size: str, seed: int = 1,
) -> pd.DataFrame:
    """Load and concatenate every filtering stats.jsonl for *model_size*/*seed*."""
    seed_dir = filtered_root / model_size / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Missing {seed_dir}")
    dfs = []
    for run_dir in sorted(seed_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        stats_path = run_dir / "stats.jsonl"
        if stats_path.exists():
            dfs.append(load_stats_jsonl(stats_path, verbose=False))
    if not dfs:
        raise FileNotFoundError(f"No filtering stats found under {seed_dir}")
    return pd.concat(dfs, ignore_index=True)


def build_filtering_reference(
    baseline_df: pd.DataFrame,
    filtering_df: pd.DataFrame,
    curves: dict,
    max_steps: dict,
) -> pd.DataFrame:
    """Pre-compute filtering reference results (called once per model size)."""
    bl = baseline_df.copy()
    fl = filtering_df.copy()
    for df in (bl, fl):
        df["seed"] = 0
    ref = fast_add_metrics(
        pd.concat([bl, fl], ignore_index=True),
        curves, max_steps,
    ).query("name != 'baseline'").copy()
    ref["_retained_key"] = ref["retained"].apply(str)
    ref["_finetune_key"] = ref["finetune"].fillna("__none__")
    return ref


def vsf_score(
    trial_df: pd.DataFrame,
    ref_results: pd.DataFrame,
    baseline_df: pd.DataFrame,
    curves: dict,
    max_steps: dict,
) -> tuple[float, pd.DataFrame]:
    """Score a trial against pre-computed filtering reference results.

    Definition: symmetric group MSE (core/retain/forget), evenly weighted (1/3 each).

    Returns ``(score, details_df)``.
    """
    trial_df = trial_df.copy()
    baseline_copy = baseline_df.copy()
    for df in (baseline_copy, trial_df):
        df["seed"] = 0

    trial_results = fast_add_metrics(
        pd.concat([baseline_copy, trial_df], ignore_index=True),
        curves, max_steps,
    ).query("name != 'baseline'").copy()

    trial_results["_retained_key"] = trial_results["retained"].apply(str)
    trial_results["_finetune_key"] = trial_results["finetune"].fillna("__none__")

    merge_keys = ["_retained_key", "data_label", "elicited", "_finetune_key"]
    merged = ref_results[merge_keys + ["retained", "compute_ratio"]].merge(
        trial_results[merge_keys + ["compute_ratio"]],
        on=merge_keys,
        suffixes=("_filter", "_trial"),
    )

    merged = merged[
        ~merged["elicited"].fillna(False).astype(bool)
    ].reset_index(drop=True)

    if merged.empty:
        return float("inf"), pd.DataFrame()

    is_core = merged["data_label"] == "core"
    is_retain = merged.apply(
        lambda r: r["data_label"] != "core" and r["data_label"] in r["retained"],
        axis=1,
    )
    is_forget = ~is_core & ~is_retain
    cr_filter = merged["compute_ratio_filter"].values
    cr_trial = merged["compute_ratio_trial"].values
    is_core_or_retain = (is_core | is_retain).values
    se = (cr_filter - cr_trial) ** 2
    merged["ae"] = se
    merged["group"] = np.where(is_core, "core", np.where(is_retain, "retain", "forget"))
    core_mse = float(se[is_core].mean()) if is_core.any() else 0.0
    retain_mse = float(se[is_retain].mean()) if is_retain.any() else 0.0
    forget_mse = float(se[is_forget].mean()) if is_forget.any() else 0.0
    score = (core_mse + retain_mse + forget_mse) / 3.0

    # CLR alternative (kept commented for reference):
    # agg = (
    #     merged.groupby(["_retained_key", "data_label"], as_index=False)[
    #         ["compute_ratio_filter", "compute_ratio_trial"]
    #     ]
    #     .mean()
    #     .rename(columns={"_retained_key": "retained_key"})
    # )
    # all_aes: list[float] = []
    # details_rows: list[dict] = []
    # eps = 1e-12
    # for retained_key, sub in agg.groupby("retained_key"):
    #     valid = sub[
    #         sub["compute_ratio_filter"].notna()
    #         & sub["compute_ratio_trial"].notna()
    #     ].copy()
    #     if valid.empty:
    #         continue
    #     labels = valid["data_label"].astype(str).tolist()
    #     cr_f = np.clip(valid["compute_ratio_filter"].to_numpy(dtype=float), eps, None)
    #     cr_t = np.clip(valid["compute_ratio_trial"].to_numpy(dtype=float), eps, None)
    #     clr_f = np.log(cr_f)
    #     clr_f = clr_f - np.mean(clr_f)
    #     clr_t = np.log(cr_t)
    #     clr_t = clr_t - np.mean(clr_t)
    #     aes = np.abs(clr_f - clr_t)
    #     all_aes.extend(aes.tolist())
    #     for label, cf, ct, ae in zip(labels, clr_f, clr_t, aes):
    #         details_rows.append(
    #             {
    #                 "retained_key": retained_key,
    #                 "data_label": label,
    #                 "clr_filter": float(cf),
    #                 "clr_trial": float(ct),
    #                 "ae": float(ae),
    #             }
    #         )
    # if not all_aes:
    #     return float("inf"), pd.DataFrame()
    # score = float(np.mean(all_aes))
    # details = pd.DataFrame(details_rows)
    # return score, details

    return score, merged


def compute_all_vsf(
    results_root: Path,
    scaling_root: Path,
    filtered_root: Path,
) -> dict[tuple[str, int, float, int], float]:
    """Compute vsf score for every trial under *results_root*.

    Returns ``{(model_label, seed, lr, bs): vsf_score}``.
    """
    trial_info: list[tuple[str, int, float, int, Path]] = []
    for config_fp in sorted(results_root.rglob("config.json")):
        if "_OLD" in str(config_fp):
            continue
        stats_fp = config_fp.parent / "stats.jsonl"
        if not stats_fp.exists():
            continue
        try:
            with open(config_fp) as f:
                cfg = json.load(f)
            model_label, seed = _infer_model_seed(cfg)
            lr = float(cfg["stages"][0]["lr"])
            bs = _extract_bs(cfg["run"])
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            continue
        trial_info.append((model_label, seed, lr, bs, stats_fp))

    models_needed = sorted({t[0] for t in trial_info})
    refs: dict[str, tuple] = {}
    for ml in models_needed:
        try:
            baseline_df, losses_pkl = load_baseline(scaling_root, ml)
            curves, max_steps = fit_baseline_curves(losses_pkl)
            filtering_df = load_filtering_stats(filtered_root, ml)
            ref_results = build_filtering_reference(baseline_df, filtering_df, curves, max_steps)
            refs[ml] = (baseline_df, curves, max_steps, ref_results)
        except FileNotFoundError as e:
            print(f"  [vsf] skipping {ml}: {e}")

    vsf_map: dict[tuple[str, int, float, int], float] = {}
    for ml, seed, lr, bs, stats_fp in trial_info:
        if ml not in refs:
            continue
        baseline_df, curves, max_steps, ref_results = refs[ml]
        try:
            trial_df = load_stats_jsonl(stats_fp, verbose=False)
            score, _ = vsf_score(trial_df, ref_results, baseline_df, curves, max_steps)
        except Exception:
            score = float("nan")
        vsf_map[(ml, seed, lr, bs)] = score
    return vsf_map



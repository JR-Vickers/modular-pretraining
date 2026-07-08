import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# set pd to show all columns
pd.set_option('display.max_columns', None)

def load_stats_jsonl(filepath: Path, verbose: bool = True) -> pd.DataFrame:
    """Load a single stats.jsonl file and return as a DataFrame.

    Extracts scalar fields from the nested 'stage' dict (skipping nested
    dicts like 'model') so they become top-level columns alongside the
    record's own fields (data_label, loss, retained, etc.).
    Also reads config.json from the same directory for aux_labels and seed.
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            stage_dict = record.pop('stage', {})
            for key, value in stage_dict.items():
                if not isinstance(value, (dict, list)):
                    record[key] = value

            records.append(record)

    df = pd.DataFrame(records)
    df['source_file'] = str(filepath)

    config_path = filepath.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        try:
            aux_labels = tuple(sorted(config['data']['aux']['labels']))
            if verbose:
                print(f"aux_labels: {aux_labels}")
            df['aux_labels'] = [aux_labels] * len(df.index)
        except KeyError:
            pass
        try:
            df['seed'] = config['run']['seed']
        except KeyError:
            pass

    for col in ("retained", "expert_labels", "aux_labels"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: tuple(sorted(x)) if isinstance(x, (list, tuple)) else tuple()
            )

    return df

def load_dir(directory: str, verbose: bool = False) -> pd.DataFrame:
    """Load all stats.jsonl files from a directory and concatenate into a single DataFrame.
    
    Args:
        directory: Path to the root directory to search (e.g., 
            '/workspace/gradient-routing/experiments/ICML-Codebase/src/results/stories/01/combined_2025-11-30_20-05-12')
    
    Returns:
        A concatenated DataFrame with all records from all stats.jsonl files.
        The 'stage' nested dict is flattened so its keys become columns with 'stage_' prefix.
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    stats_files = list(directory.rglob('stats.jsonl'))
    
    if not stats_files:
        raise ValueError(f"No stats.jsonl files found in {directory}")
    
    if verbose:
        print(f"Found {len(stats_files)} stats.jsonl file(s)")
    
    dfs = []
    for filepath in stats_files:
        df = load_stats_jsonl(filepath, verbose=verbose)
        dfs.append(df)
        if verbose:
            print(f"  Loaded {len(df)} records from {filepath}")
    
    df = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"Total: {len(df)} records")

    return df


# ============================================================
# Metrics computation
# ============================================================

def load_val_losses(pkl_path: Path, warmup_prc: float = 0.02) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load validation losses from pickle file, dropping warmup portion.

    Returns dict mapping label -> (steps, losses) arrays.
    The pickle contains {"train": {...}, "val": {...}} where each value
    is a dict mapping label -> (N, 2) array of [step, loss] pairs.
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    assert "val" in d and isinstance(d["val"], dict)
    d = d["val"]

    out = {}
    for k, v in d.items():
        arr = np.asarray(v, float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        steps, losses = arr[:, 0], arr[:, 1]
        warmup_n = int(losses.size * warmup_prc)
        out[k] = (steps[warmup_n:], losses[warmup_n:])

    return out


def _power_law(x, A, alpha, x0):
    """Power-law: A * (x + x0)^(-alpha)."""
    return A * np.power(x + x0, -alpha)

def _fit_power_law(x, y) -> tuple[float, float, float]:
    """Fit power-law curve to loss data. Returns (A, alpha, x0)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y_mon = np.minimum.accumulate(y)

    A0 = max(float(y_mon[0]), 1e-3)

    dx = x[1] - x[0] if len(x) > 1 else 1.0
    x_min = float(x.min())
    eps = 1e-6
    lb = [1e-10, 1e-3, -x_min + eps]
    ub = [1e8, 5.0, x.max() + 10 * dx]
    p0 = [A0, 0.3, min(100.0, ub[2] * 0.9)]

    def resid(p):
        A, alpha, x0 = p
        return A * np.power(x + x0, -alpha) - y

    res = least_squares(resid, p0, bounds=(lb, ub),
                        loss='soft_l1', f_scale=0.01, max_nfev=40000)
    return tuple(res.x)  # (A, alpha, x0)


def _g_inv_power_law(ell, A, alpha, x0) -> np.ndarray:
    """Inverse of power-law: given loss, return equivalent step."""
    ell = np.asarray(ell, float)
    eps = 1e-12
    base = np.maximum(A / np.maximum(ell, eps), eps)
    xin = np.power(base, 1.0 / alpha) - x0
    return np.maximum(xin, 1e-3)


def _step_equiv_extrapolated(loss, A, alpha, x0, s_max):
    """Compute step-equivalent with linear extrapolation beyond the training range.

    For losses within the fitted curve's range (loss >= f(s_max)), uses the
    standard inverse power-law.  For losses below f(s_max), extrapolates
    linearly using the tangent at s_max:

        step_equiv = s_max + (f(s_max) - loss) / |f'(s_max)|

    This keeps step_equiv finite and continuous while still allowing
    values > s_max (i.e. compute_ratio > 1) when a method genuinely
    beats the baseline.
    """
    loss = np.asarray(loss, float)
    f_smax = A * np.power(s_max + x0, -alpha)
    normal = _g_inv_power_law(loss, A, alpha, x0)

    slope_mag = alpha * A * np.power(s_max + x0, -(alpha + 1))
    extrapolated = s_max + np.maximum(f_smax - loss, 0.0) / slope_mag

    result = np.where(loss >= f_smax, normal, extrapolated)
    return float(result) if result.ndim == 0 else result

def plot_loss_curve(x, y, curve, label):
    plt.plot(x, y, label=f"{label} (raw)")
    plt.plot(x, _power_law(x, *curve), label=f"{label} (fitted)")
    plt.legend()
    plt.show()
    return

def add_metrics(
    df: pd.DataFrame,
    losses_pkl_path: str,
    split: str = 'test'
) -> pd.DataFrame:
    """Add computed metrics to the DataFrame using baseline losses from a pickle file.
    
    This function:
    1. Loads baseline validation losses from the pickle file
    2. Fits power-law curves to each label's loss trajectory
    3. Computes metrics for each row in the DataFrame:
       - ppl: perplexity (exp of loss)
       - step_equiv: equivalent training step for the observed loss
       - compute_ratio: ratio of step_equiv to baseline step_equiv
       - ppl_ratio: ratio of ppl to baseline ppl
       - loss_ratio: ratio of loss to baseline loss
    
    Args:
        df: DataFrame with columns including 'name', 'data_label', 'loss'
        losses_pkl_path: Path to the losses.pkl file containing baseline losses
    
    Returns:
        DataFrame with additional metric columns added.
    """
    df = df.copy()
    losses_pkl_path = Path(losses_pkl_path)
    
    # Load and fit curves
    losses = load_val_losses(losses_pkl_path)
    labels = list(losses.keys())
    
    curves = {}
    max_steps = {}
    for lab in labels:
        x_b, y_b = losses[lab]
        max_steps[lab] = float(x_b[-1])
        curves[lab] = _fit_power_law(x_b, y_b)

    # Compute perplexity
    df['ppl'] = np.exp(df['loss'])
    
    # Compute step equivalent with linear extrapolation beyond the
    # training range to prevent divergence near the power-law floor.
    def step_equiv(label, loss):
        if pd.isna(label) or pd.isna(loss) or label not in curves:
            return np.nan
        A, alpha, x0 = curves[label]
        return _step_equiv_extrapolated(loss, A, alpha, x0, max_steps[label])
    
    df["step_equiv"] = df.apply(lambda r: step_equiv(r["data_label"], r["loss"]), axis=1)

    # Per-seed baseline references: all rows (elicited or not) are compared
    # against the non-elicited baseline so that CRs are on the same scale.
    not_elicited = ~df.get('elicited', pd.Series(False, index=df.index)).fillna(False).astype(bool)
    baseline_select = (df['name'] == 'baseline') & not_elicited
    if 'split' in df.columns:
        split_matches = (df['split'] == split) | df['split'].isna()
        baseline_select = baseline_select & split_matches
    baselines = (
        df[baseline_select][["data_label", "seed", "loss", "step_equiv", "ppl"]]
        .groupby(["seed", "data_label"], dropna=False)
        .mean()
        .to_dict('index')
    )

    def calc_ratio(row, metric):
        label, seed, val = row["data_label"], row["seed"], row[metric]
        if pd.isna(label) or pd.isna(val):
            return np.nan
        key = (seed, label)
        ref = baselines.get(key, {}).get(metric, np.nan)
        if pd.isna(ref) or ref == 0:
            return np.nan
        return val / ref

    df["compute_ratio"] = df.apply(lambda r: calc_ratio(r, "step_equiv"), axis=1)
    df["ppl_ratio"] = df.apply(lambda r: calc_ratio(r, "ppl"), axis=1)
    df["loss_ratio"] = df.apply(lambda r: calc_ratio(r, "loss"), axis=1)
    df["log_compute_ratio"] = np.log(df["compute_ratio"])

    return df
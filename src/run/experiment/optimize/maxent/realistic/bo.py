"""
Bayesian optimisation of MaxEnt hyperparameters at 800M, across seeds 1-3.

For each (seed, retain_target) pair the script runs one independent BO study
optimising (lr, alpha_retain) against the matching filtering reference.
Studies converge to entirely different hyperparameters per pair. Within a
study, every trial restarts maxent from the seed's baseline checkpoint
(shared via hardlink) and is scored with ``vsf_score`` against that
(seed, retain) target's filtering ``stats.jsonl``.

Retain targets:
    - ["core"]
    - ["core", "papers-biology"]
    - ["core", "code-lisp"]
    - ["core", "papers-cyber"]
    - ["core", "papers-nuclear"]

Baseline + filtering experiment ids are auto-discovered per seed by
scanning results/scaling/realistic/{base,filtering}/800M/seed_N/.

Per-study outputs:
    results/optimize/maxent/800M/seed_N/<retain_key>/bo_trial_NNN/
    results/optimize/maxent/800M/seed_N/<retain_key>/bo_summary.json
    results/optimize/maxent/800M/bo_summary_all.json

Usage (single-node):
    export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.optimize.maxent.bo
"""
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import optuna

from src.run.util.config import ExperimentConfig
from src.run.util.distributed import is_main_process, barrier
from src.run.train.base import BaselineConfig
from src.run.train.maxent import MaxentConfig
from src.run.experiment.config import GetRealisticConfig
from src.run.experiment.common import (
    ROOT_DIR, make_param_str, get_bs, get_lr,
)
from src.run.main import run as run_single

from analysis.common.load_data import load_stats_jsonl
from analysis.optimize.common import (
    fit_baseline_curves,
    build_filtering_reference,
    vsf_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed experiment constants
# ---------------------------------------------------------------------------
MODEL_SIZE = 800e6
SEEDS = (1, 2, 3)
STEPS = 2000

ALL_RETAIN_TARGETS: tuple[tuple[str, ...], ...] = (
    ("core",),
    ("core", "papers-biology"),
    ("core", "code-lisp"),
    ("core", "papers-cyber"),
    ("core", "papers-nuclear"),
)

# Per-seed override of which retain targets to run. Seeds present here
# only run the listed targets; seeds absent run all of ALL_RETAIN_TARGETS.
# (Seed 1 had every target except papers-nuclear already swept, so we
# only redo nuclear there.)
SEED_RETAIN_TARGETS: dict[int, tuple[tuple[str, ...], ...]] = {
    2: (("core", "papers-nuclear"),),
}

# Param ranges
LR_RANGE = (5e-5, 2e-4)
ALPHA_RETAIN_RANGE = (50.0, 600.0)

# Path roots
PARAM_STR = make_param_str(MODEL_SIZE)
SCALING_BASE_ROOT = ROOT_DIR / "scaling" / "realistic" / "base" / PARAM_STR
SCALING_FILTERING_ROOT = ROOT_DIR / "scaling" / "realistic" / "filtering" / PARAM_STR
BO_ROOT = ROOT_DIR / "optimize" / "maxent" / PARAM_STR


def _retain_key(retain_target: list[str] | tuple[str, ...]) -> str:
    """Compact string id for a retain-target combo (used in dir / study names)."""
    return "_".join(retain_target)


# ---------------------------------------------------------------------------
# Auto-discovery of per-seed baseline + filtering experiment ids.
# ---------------------------------------------------------------------------

def _discover_baseline_id(seed: int) -> str:
    """Pick the latest ts_dir under base/<PARAM_STR>/seed_N/ with both
    stats.jsonl and baseline/losses.pkl. (checkpoint.pth is validated at
    trial-staging time so this works on lightweight local mirrors too.)"""
    seed_dir = SCALING_BASE_ROOT / f"seed_{seed}"
    candidates = [
        d for d in seed_dir.iterdir()
        if d.is_dir()
        and (d / "stats.jsonl").exists()
        and (d / "baseline" / "losses.pkl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No usable baseline ts_dir under {seed_dir} "
            "(need stats.jsonl + baseline/losses.pkl)"
        )
    return max(candidates, key=lambda d: d.name).name


def _discover_filtering_ids(seed: int) -> dict[tuple[str, ...], str]:
    """Build {retain_tuple: ts_id} for filtering runs under
    filtering/<PARAM_STR>/seed_N/. Each filtering ts_dir contains rows for
    exactly one retain set; later ts_dirs win if the same retain set was
    re-run."""
    seed_dir = SCALING_FILTERING_ROOT / f"seed_{seed}"
    out: dict[tuple[str, ...], str] = {}
    # Sort ASCENDING by name so later ts_dirs overwrite earlier ones.
    for ts_dir in sorted(seed_dir.iterdir()):
        if not ts_dir.is_dir():
            continue
        stats = ts_dir / "stats.jsonl"
        if not stats.exists():
            continue
        retained = None
        with open(stats) as f:
            for line in f:
                rec = json.loads(line)
                stage = rec.get("stage") or {}
                if stage.get("name") != "filtering":
                    continue
                retained = tuple(sorted(rec.get("retained", [])))
                break
        if retained is not None:
            out[retained] = ts_dir.name
    return out


# ---------------------------------------------------------------------------
# Per-trial scaffolding.
# ---------------------------------------------------------------------------

def _stage_trial_baseline(
    study_root: Path, trial_id: str, baseline_src_dir: Path,
) -> None:
    """Create <study_root>/<trial_id>/baseline/ with hardlinked checkpoint +
    copied metadata, so each trial has a valid experiment layout. Rank 0 only."""
    trial_baseline = study_root / trial_id / "baseline"
    if is_main_process():
        trial_baseline.mkdir(parents=True, exist_ok=True)
        ckpt_src = baseline_src_dir / "checkpoint.pth"
        ckpt_dst = trial_baseline / "checkpoint.pth"
        if ckpt_src.exists() and not ckpt_dst.exists():
            os.link(ckpt_src, ckpt_dst)
        for name in ("stage.json", "losses.pkl"):
            src = baseline_src_dir / name
            if src.exists() and not (trial_baseline / name).exists():
                shutil.copy(src, trial_baseline / name)
    barrier()


def _make_config(
    study_root: Path,
    s3_prefix: str,
    trial_id: str,
    seed: int,
    lr: float,
    alpha_retain: float,
    retain_target: list[str],
    do_elicit: bool,
) -> ExperimentConfig:
    """Build a single-trial experiment config for one retain target."""
    eff_bs = get_bs(MODEL_SIZE)
    align = 64  # 800M < 5e9
    config = GetRealisticConfig(MODEL_SIZE, align=align)

    baseline_stage = BaselineConfig(
        lr=get_lr(MODEL_SIZE),
        num_checkpoints=100,
        num_train_evals=100,
        do_elicit=False,
    )
    maxent_stage = MaxentConfig(
        lr=lr,
        steps=STEPS,
        alpha_retain=alpha_retain,
        num_checkpoints=-1,
        do_elicit=do_elicit,
        retain_targets=[retain_target],
    )
    config.stages = [baseline_stage, maxent_stage]

    config.run.target_effective_batch_size = eff_bs
    config.run.seed = seed
    config.run.cleanup_distributed = False
    config.run.res_root = study_root
    config.run.experiment_id = trial_id
    config.run.log_level = "DEBUG"
    config.run.compile = True
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = s3_prefix
    return config


def _load_trial_stats(study_root: Path, trial_id: str):
    stats_path = study_root / trial_id / "stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing trial stats at {stats_path}")
    return load_stats_jsonl(stats_path, verbose=False)


# ---------------------------------------------------------------------------
# Per-(seed, retain) BO study.
# ---------------------------------------------------------------------------

def run_bo_for_target(
    seed: int,
    retain_target: list[str],
    baseline_id: str,
    filtering_id: str,
    n_trials: int,
    n_startup: int,
    do_elicit: bool,
    baseline_df,
    curves: dict,
    max_steps: dict,
) -> optuna.Study:
    """Run one independent BO study for a single (seed, retain_target)."""
    key = _retain_key(retain_target)
    bo_root_seed = BO_ROOT / f"seed_{seed}"
    study_root = bo_root_seed / key
    s3_prefix = f"optimize/maxent/{PARAM_STR}/seed_{seed}/{key}"

    baseline_src_dir = SCALING_BASE_ROOT / f"seed_{seed}" / baseline_id / "baseline"
    filtering_stats_path = (
        SCALING_FILTERING_ROOT / f"seed_{seed}" / filtering_id / "stats.jsonl"
    )
    filtering_df = load_stats_jsonl(filtering_stats_path, verbose=False)
    ref_results = build_filtering_reference(
        baseline_df, filtering_df, curves, max_steps,
    )

    logger.info(
        f"[seed_{seed}/{key}] BO study setup: baseline={baseline_id}, "
        f"filtering={filtering_id}, retain={retain_target}, "
        f"n_trials={n_trials}, study_root={study_root}"
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        alpha_retain = trial.suggest_float("alpha_retain", *ALPHA_RETAIN_RANGE)

        trial_id = f"bo_trial_{trial.number:03d}"
        _stage_trial_baseline(study_root, trial_id, baseline_src_dir)

        config = _make_config(
            study_root, s3_prefix, trial_id, seed, lr, alpha_retain,
            retain_target, do_elicit=do_elicit,
        )

        logger.info(
            f"[seed_{seed}/{key}] Trial {trial.number}: lr={lr:.3e}, "
            f"alpha_retain={alpha_retain:.1f}, steps={STEPS}"
        )

        try:
            run_single(config)
            trial_df = _load_trial_stats(study_root, trial_id)
            score, _details = vsf_score(
                trial_df, ref_results, baseline_df, curves, max_steps,
            )
        except Exception as e:
            logger.error(
                f"[seed_{seed}/{key}] Trial {trial.number} failed: {e}",
                exc_info=True,
            )
            raise optuna.TrialPruned(f"Trial failed: {e}")

        logger.info(f"[seed_{seed}/{key}] Trial {trial.number}: score={score:.4f}")
        return score

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup, seed=seed,
    )
    study = optuna.create_study(
        study_name=f"maxent_bo_{int(MODEL_SIZE/1e6)}M_seed{seed}_{key}",
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    if is_main_process():
        summary = {
            "seed": seed,
            "retain_target": retain_target,
            "baseline_experiment_id": baseline_id,
            "filtering_experiment_id": filtering_id,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "all_trials": [
                {
                    "number": t.number,
                    "state": t.state.name,
                    "value": t.value,
                    "params": t.params,
                }
                for t in study.trials
            ],
        }
        summary_path = study_root / "bo_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"[seed_{seed}/{key}] Summary written to {summary_path}")
        logger.info(
            f"[seed_{seed}/{key}] Best: score={study.best_value:.4f}, "
            f"params={study.best_params}"
        )

    return study


# ---------------------------------------------------------------------------
# Orchestrator: loops over (seed, retain_target).
# ---------------------------------------------------------------------------

def _build_run_plan(
    seeds: tuple[int, ...],
    retain_keys: list[str] | None,
) -> list[tuple[int, list[str], str, str]]:
    """Return [(seed, retain_target, baseline_id, filtering_id), ...].

    Raises if a (seed, retain) target has no matching filtering ts_dir.
    """
    plan: list[tuple[int, list[str], str, str]] = []
    for seed in seeds:
        baseline_id = _discover_baseline_id(seed)
        filtering_ids = _discover_filtering_ids(seed)
        targets_for_seed = SEED_RETAIN_TARGETS.get(seed, ALL_RETAIN_TARGETS)
        logger.info(
            f"[seed_{seed}] baseline={baseline_id}; "
            f"retain targets: {[_retain_key(list(rt)) for rt in targets_for_seed]}; "
            f"filtering ids: {[(_retain_key(list(k)), v) for k, v in filtering_ids.items()]}"
        )
        for rt in targets_for_seed:
            rt_list = list(rt)
            if retain_keys is not None and _retain_key(rt_list) not in retain_keys:
                continue
            rt_sorted = tuple(sorted(rt))
            if rt_sorted not in filtering_ids:
                raise FileNotFoundError(
                    f"No filtering ts_dir for seed={seed} retain={rt}. "
                    f"Available: {list(filtering_ids.keys())}"
                )
            plan.append((seed, rt_list, baseline_id, filtering_ids[rt_sorted]))
    return plan


def run_bo(
    n_trials: int = 30,
    n_startup: int = 10,
    do_elicit: bool = False,
    seeds: tuple[int, ...] = SEEDS,
    retain_keys: list[str] | None = None,
) -> dict[tuple[int, str], optuna.Study]:
    """Run one independent BO study per (seed, retain_target).

    Args:
        n_trials: trials per study.
        n_startup: TPE warm-start trials per study.
        do_elicit: whether to run adversarial FT per trial.
        seeds: seeds to iterate over. Defaults to (1, 2, 3).
        retain_keys: optional subset of retain keys (e.g. ["core",
            "core_papers-biology"]) to run; defaults to all 5.

    Returns:
        dict mapping (seed, retain_key) -> Study.
    """
    plan = _build_run_plan(seeds, retain_keys)
    if not plan:
        raise ValueError(
            "Empty run plan. Available retain keys: "
            f"{[_retain_key(list(k)) for k in ALL_RETAIN_TARGETS]}"
        )

    studies: dict[tuple[int, str], optuna.Study] = {}
    # Cache baseline df / curves per seed to avoid redundant fits.
    seed_cache: dict[int, tuple] = {}

    for seed, retain_target, baseline_id, filtering_id in plan:
        if seed not in seed_cache:
            baseline_stats_path = (
                SCALING_BASE_ROOT / f"seed_{seed}" / baseline_id / "stats.jsonl"
            )
            baseline_losses_pkl = (
                SCALING_BASE_ROOT / f"seed_{seed}" / baseline_id
                / "baseline" / "losses.pkl"
            )
            baseline_df = load_stats_jsonl(baseline_stats_path, verbose=False)
            curves, max_steps = fit_baseline_curves(baseline_losses_pkl)
            seed_cache[seed] = (baseline_df, curves, max_steps)
        baseline_df, curves, max_steps = seed_cache[seed]

        key = _retain_key(retain_target)
        logger.info(
            f"=== Starting BO study: seed={seed} retain={retain_target} ==="
        )
        studies[(seed, key)] = run_bo_for_target(
            seed=seed,
            retain_target=retain_target,
            baseline_id=baseline_id,
            filtering_id=filtering_id,
            n_trials=n_trials,
            n_startup=n_startup,
            do_elicit=do_elicit,
            baseline_df=baseline_df,
            curves=curves,
            max_steps=max_steps,
        )

    if is_main_process():
        combined = {
            f"seed_{seed}/{key}": {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            }
            for (seed, key), study in studies.items()
        }
        combined_path = BO_ROOT / "bo_summary_all.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(combined, indent=2, default=str))
        logger.info(f"Combined summary written to {combined_path}")

    return studies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_startup", type=int, default=10)
    parser.add_argument(
        "--do_elicit", action="store_true",
        help="Run adversarial FT per trial (adds ~40min/trial; VSF score "
             "ignores elicited rows so this is off by default).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS),
        help="Seeds to run BO studies for (default: 1 2 3).",
    )
    parser.add_argument(
        "--retain_keys", nargs="+", default=None,
        help="Optional subset of retain keys to run (e.g. core "
             "core_papers-biology). Defaults to all 5.",
    )
    args = parser.parse_args()
    run_bo(
        n_trials=args.n_trials,
        n_startup=args.n_startup,
        do_elicit=args.do_elicit,
        seeds=tuple(args.seeds),
        retain_keys=args.retain_keys,
    )

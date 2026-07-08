"""
Bayesian optimisation of MaxEnt hyperparameters on the SimpleStories dataset,
across seeds 1-3. Mirrors the realistic-800M maxent BO
(src/run/experiment/optimize/maxent/bo.py) but adapted for the stories model.

For each (seed, retain_target) pair the script runs one independent Optuna/TPE
study optimising (lr, alpha_retain). Every trial restarts maxent from the seed's
saved stories baseline checkpoint (hardlinked into the trial dir so the baseline
stage is skipped) and is scored with ``vsf_score`` -- symmetric core/retain/forget
compute-ratio MSE against the matching filtering reference for that (seed, retain).

Retain targets (stories aux labels = sorted()[:4]):
    ("core",)
    ("core", "a-deadline-or-time-limit")
    ("core", "alien-encounters")
    ("core", "bygone-eras")
    ("core", "cultural-traditions")

Unlike the realistic experiment (where base/ and filtering/ live in separate
trees), each stories seed is ONE run under results/stories/seed_N/<ts_id>/ whose
stats.jsonl holds both the baseline rows and the filtering rows (all 5 retain
sets). We auto-discover that ts_dir per seed.

Per-study outputs:
    results/optimize/maxent_stories/seed_N/<retain_key>/bo_trial_NNN/
    results/optimize/maxent_stories/seed_N/<retain_key>/bo_summary.json
    results/optimize/maxent_stories/bo_summary_all.json

Usage (single GPU is plenty for this model):
    export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=1 -m src.run.experiment.optimize.maxent_stories.bo
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
from src.run.experiment.config import GetStoriesConfig
from src.run.experiment.common import ROOT_DIR
from src.run.main import run as run_single

from analysis.common.load_data import load_stats_jsonl
from analysis.optimize.common import (
    fit_baseline_curves,
    build_filtering_reference,
    vsf_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed experiment constants  (EDIT search ranges / steps here)
# ---------------------------------------------------------------------------
SEEDS = (1, 2, 3)
STEPS = 2000                       # maxent steps per trial
BASELINE_LR = 5e-3                 # stories baseline lr (acc_mode=heterogeneous)
ACC_MODE = "heterogeneous"         # matches how the stories baseline trained

# Stories aux labels = sorted(all_labels)[:4]; retain targets = core + each (and core-only).
ALL_RETAIN_TARGETS: tuple[tuple[str, ...], ...] = (
    ("core",),
    ("core", "a-deadline-or-time-limit"),
    ("core", "alien-encounters"),
    ("core", "bygone-eras"),
    ("core", "cultural-traditions"),
)

# Search ranges. Stories trains at a much higher lr than realistic-800M and the
# MaxentConfig default alpha_retain (15) is far below realistic's (~300), so these
# are scaled for the small model -- tune as needed.
LR_RANGE = (3e-4, 1e-2)            # log-uniform
ALPHA_RETAIN_RANGE = (5.0, 300.0)  # uniform

STORIES_ROOT = ROOT_DIR / "stories"
BO_ROOT = ROOT_DIR / "optimize" / "maxent_stories"


def _retain_key(retain_target) -> str:
    return "_".join(retain_target)


# ---------------------------------------------------------------------------
# Auto-discovery: one ts_dir per seed holding baseline + filtering stages.
# ---------------------------------------------------------------------------

def _has_filtering_rows(stats_path: Path) -> bool:
    """True iff stats.jsonl has at least one filtering-stage row.

    Guards against picking a run dir that has a baseline but no filtering
    reference — e.g. a maxent-only unlearning run (results/stories/seed_N/<ts>/
    with baseline/ + maxent_NN/ but no filtering stage). Such a dir would
    otherwise be the max-name candidate and then break build_filtering_reference.
    """
    try:
        with open(stats_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("stage") or {}).get("name") == "filtering":
                    return True
    except OSError:
        return False
    return False


def _discover_run_dir(seed: int) -> Path:
    """Latest ts_dir under stories/seed_N/ that has baseline/checkpoint.pth,
    baseline/losses.pkl, stats.jsonl AND filtering-stage rows (so it is a real
    baseline+filtering run, not a maxent/routed-only output dir)."""
    seed_dir = STORIES_ROOT / f"seed_{seed}"
    cands = [
        d for d in seed_dir.iterdir()
        if d.is_dir()
        and (d / "stats.jsonl").exists()
        and (d / "baseline" / "losses.pkl").exists()
        and (d / "baseline" / "checkpoint.pth").exists()
        and _has_filtering_rows(d / "stats.jsonl")
    ]
    if not cands:
        raise FileNotFoundError(
            f"No usable stories run under {seed_dir} "
            "(need stats.jsonl with filtering rows + baseline/{losses.pkl,checkpoint.pth})"
        )
    return max(cands, key=lambda d: d.name)


def _split_baseline_filtering(stats_path: Path):
    """Return (baseline_df, filtering_df) from one run's stats.jsonl, split by stage."""
    df = load_stats_jsonl(stats_path, verbose=False)
    base_df = df[df["name"] == "baseline"].copy()
    filt_df = df[df["name"] == "filtering"].copy()
    return base_df, filt_df


def _available_filtering_targets(filt_df) -> set[tuple[str, ...]]:
    return {tuple(sorted(r)) for r in filt_df["retained"].dropna()
            if isinstance(r, (list, tuple))}


# ---------------------------------------------------------------------------
# Per-trial scaffolding.
# ---------------------------------------------------------------------------

def _stage_trial_baseline(study_root: Path, trial_id: str, baseline_src_dir: Path) -> None:
    """Hardlink baseline checkpoint + copy metadata so the trial's baseline stage
    is recognised as complete and maxent restarts from it. Rank 0 only."""
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


def _make_config(study_root, s3_prefix, trial_id, seed, lr, alpha_retain,
                 retain_target, do_elicit) -> ExperimentConfig:
    config = GetStoriesConfig()

    baseline_stage = BaselineConfig(
        lr=BASELINE_LR,
        acc_mode=ACC_MODE,
        num_train_evals=200,
        do_elicit=False,
    )
    maxent_stage = MaxentConfig(
        lr=lr,
        steps=STEPS,
        alpha_retain=alpha_retain,
        acc_mode=ACC_MODE,
        num_checkpoints=-1,
        num_train_evals=0,
        do_eval=True,
        do_elicit=do_elicit,
        retain_targets=[retain_target],
    )
    config.stages = [baseline_stage, maxent_stage]

    config.run.seed = seed
    config.run.cleanup_distributed = False
    config.run.find_unused_parameters = True
    config.run.res_root = study_root
    config.run.experiment_id = trial_id
    config.run.log_level = "DEBUG"
    config.run.compile = False   # tiny model: torch.compile recompiles per fresh trial model and dominates per-trial cost; eager is faster overall
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

def run_bo_for_target(seed, retain_target, run_dir, n_trials, n_startup,
                      do_elicit, baseline_df, filt_df, curves, max_steps) -> optuna.Study:
    key = _retain_key(retain_target)
    study_root = BO_ROOT / f"seed_{seed}" / key
    s3_prefix = f"optimize/maxent_stories/seed_{seed}/{key}"
    baseline_src_dir = run_dir / "baseline"

    ref_results = build_filtering_reference(baseline_df, filt_df, curves, max_steps)

    logger.info(
        f"[seed_{seed}/{key}] BO setup: run_dir={run_dir.name}, retain={retain_target}, "
        f"n_trials={n_trials}, study_root={study_root}"
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        alpha_retain = trial.suggest_float("alpha_retain", *ALPHA_RETAIN_RANGE)
        trial_id = f"bo_trial_{trial.number:03d}"
        _stage_trial_baseline(study_root, trial_id, baseline_src_dir)
        config = _make_config(study_root, s3_prefix, trial_id, seed, lr,
                              alpha_retain, retain_target, do_elicit)
        logger.info(f"[seed_{seed}/{key}] Trial {trial.number}: lr={lr:.3e}, "
                    f"alpha_retain={alpha_retain:.1f}, steps={STEPS}")
        try:
            run_single(config)
            trial_df = _load_trial_stats(study_root, trial_id)
            score, _ = vsf_score(trial_df, ref_results, baseline_df, curves, max_steps)
        except Exception as e:
            logger.error(f"[seed_{seed}/{key}] Trial {trial.number} failed: {e}", exc_info=True)
            raise optuna.TrialPruned(f"Trial failed: {e}")
        logger.info(f"[seed_{seed}/{key}] Trial {trial.number}: score={score:.4f}")
        return score

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup, seed=seed)
    study = optuna.create_study(
        study_name=f"maxent_stories_seed{seed}_{key}",
        direction="minimize", sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    if is_main_process():
        summary = {
            "seed": seed, "retain_target": retain_target, "run_dir": run_dir.name,
            "best_value": study.best_value, "best_params": study.best_params,
            "n_trials": len(study.trials),
            "all_trials": [
                {"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
                for t in study.trials
            ],
        }
        sp = study_root / "bo_summary.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"[seed_{seed}/{key}] Best: score={study.best_value:.4f}, params={study.best_params}")
    return study


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------

def run_bo(n_trials=30, n_startup=10, do_elicit=False,
           seeds=SEEDS, retain_keys=None) -> dict:
    studies = {}
    for seed in seeds:
        run_dir = _discover_run_dir(seed)
        baseline_df, filt_df = _split_baseline_filtering(run_dir / "stats.jsonl")
        curves, max_steps = fit_baseline_curves(run_dir / "baseline" / "losses.pkl")
        avail = _available_filtering_targets(filt_df)
        logger.info(f"[seed_{seed}] run={run_dir.name}; filtering targets: {sorted(avail)}")

        for rt in ALL_RETAIN_TARGETS:
            rt_list = list(rt)
            if retain_keys is not None and _retain_key(rt_list) not in retain_keys:
                continue
            if tuple(sorted(rt)) not in avail:
                raise FileNotFoundError(
                    f"No filtering reference for seed={seed} retain={rt}. Available: {sorted(avail)}"
                )
            logger.info(f"=== BO study: seed={seed} retain={rt_list} ===")
            studies[(seed, _retain_key(rt_list))] = run_bo_for_target(
                seed, rt_list, run_dir, n_trials, n_startup, do_elicit,
                baseline_df, filt_df, curves, max_steps,
            )

    if is_main_process() and studies:
        combined = {
            f"seed_{s}/{k}": {"best_value": st.best_value, "best_params": st.best_params,
                              "n_trials": len(st.trials)}
            for (s, k), st in studies.items()
        }
        cp = BO_ROOT / "bo_summary_all.json"
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(combined, indent=2, default=str))
        logger.info(f"Combined summary -> {cp}")
    return studies


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_startup", type=int, default=10)
    p.add_argument("--do_elicit", action="store_true",
                   help="Adversarial FT per trial (VSF ignores elicited rows; off by default).")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--retain_keys", nargs="+", default=None,
                   help="Subset of retain keys, e.g. core core_alien-encounters. Default: all 5.")
    a = p.parse_args()
    run_bo(n_trials=a.n_trials, n_startup=a.n_startup, do_elicit=a.do_elicit,
           seeds=tuple(a.seeds), retain_keys=a.retain_keys)

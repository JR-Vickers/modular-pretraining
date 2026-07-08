"""
Run maxent unlearning on top of an existing baseline checkpoint.

Unlike a vanilla resume, this script *does not* touch the original
baseline run directory. Instead, mirroring ``bo.py``, it stages a fresh
experiment layout under a separate ``maxent`` results subtree:

    <RES_ROOT>/<experiment_id>/baseline/        # hardlinked checkpoint + stage.json + losses.pkl
    <RES_ROOT>/<experiment_id>/maxent_01/...    # one maxent stage per retain target
    <RES_ROOT>/<experiment_id>/maxent_02/...
    <RES_ROOT>/<experiment_id>/maxent_03/...
    <RES_ROOT>/<experiment_id>/maxent_04/...
    <RES_ROOT>/<experiment_id>/maxent_05/...

where ``RES_ROOT = results/scaling/realistic/maxent/<param>/seed_<seed>``.

We run five maxent stages back-to-back, one per retain target:

    - ["core"]
    - ["core", "papers-biology"]
    - ["core", "code-lisp"]
    - ["core", "papers-cyber"]
    - ["core", "papers-nuclear"]

For each retain target, ``(lr, alpha_retain)`` is loaded from the matching
BO study's ``bo_summary.json`` under
``results/optimize/maxent/<param>/seed_<seed>/<retain_key>/bo_summary.json``.

Each invocation generates its own fresh, timestamped ``experiment_id`` (never
the baseline's), so re-running never overwrites or skip-resumes prior maxent
results. The per-seed reference baseline checkpoint is hardcoded (see
``BASELINE_EXPERIMENT_IDS``) and copied/hardlinked into the new experiment dir.

Usage:
    torchrun --nproc_per_node=8 --master_port=29500 \
        -m src.run.experiment.scaling.realistic.maxent.run \
        --model_size 800M --seed 1
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from src.run.util.config import ExperimentConfig
from src.run.util.distributed import is_main_process, barrier
from src.run.train.base import BaselineConfig
from src.run.train.maxent import MaxentConfig
from src.run.experiment.config import GetRealisticConfig
from src.run.experiment.common import (
    ROOT_DIR, parse_model_size, make_param_str, get_bs, get_lr,
)
from src.run.main import run as run_single
from src.run.util.tools import get_timestamp


# Retain targets to run maxent for (one stage per entry, in order).
RETAIN_TARGETS: list[list[str]] = [
    ["core"],
    ["core", "papers-biology"],
    ["core", "code-lisp"],
    ["core", "papers-cyber"],
    ["core", "papers-nuclear"],
]

MAXENT_STEPS = 2000

# Hardcoded reference baseline checkpoint (its experiment_id) per seed at 800M.
# Each maxent run copies/hardlinks base/<param>/seed_<S>/<id>/baseline into a
# fresh timestamped experiment dir — no auto-discovery, no id reuse.
BASELINE_EXPERIMENT_IDS: dict[int, str] = {
    1: "20260422084509038162",
    2: "20260504072224060283",
    3: "20260504213914888636",
}


def _retain_key(retain_target: list[str]) -> str:
    """Compact string id for a retain-target combo (matches bo.py)."""
    return "_".join(retain_target)


def _baseline_src_dir(n_params: int, seed: int, baseline_experiment_id: str) -> Path:
    return (
        ROOT_DIR / "scaling" / "realistic" / "base"
        / make_param_str(n_params) / f"seed_{seed}"
        / baseline_experiment_id / "baseline"
    )


def _maxent_res_root(n_params: int, seed: int) -> Path:
    return (
        ROOT_DIR / "scaling" / "realistic" / "maxent"
        / make_param_str(n_params) / f"seed_{seed}"
    )


def _bo_summary_path(n_params: int, seed: int, retain_target: list[str]) -> Path:
    return (
        ROOT_DIR / "optimize" / "maxent" / make_param_str(n_params)
        / f"seed_{seed}" / _retain_key(retain_target) / "bo_summary.json"
    )


def _load_bo_best_params(
    n_params: int, seed: int, retain_target: list[str],
) -> tuple[float, float]:
    """Return (lr, alpha_retain) from the matching BO study's bo_summary.json.

    Reads the per-(seed, retain_target) BO study, so each training seed uses
    the optima discovered for that same seed.
    """
    path = _bo_summary_path(n_params, seed, retain_target)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing BO summary for seed={seed} retain={retain_target} at {path}. "
            f"Run src.run.experiment.optimize.maxent.bo for this seed first."
        )
    summary = json.loads(path.read_text())
    best = summary["best_params"]
    return float(best["lr"]), float(best["alpha_retain"])


def _stage_baseline(
    baseline_src: Path, res_root: Path, experiment_id: str,
) -> None:
    """Hardlink checkpoint + copy metadata into the new experiment layout.

    Rank 0 only; other ranks wait at the barrier.
    """
    dst = res_root / experiment_id / "baseline"
    if is_main_process():
        dst.mkdir(parents=True, exist_ok=True)
        ckpt_src = baseline_src / "checkpoint.pth"
        ckpt_dst = dst / "checkpoint.pth"
        if ckpt_src.exists() and not ckpt_dst.exists():
            os.link(ckpt_src, ckpt_dst)
        for name in ("stage.json", "losses.pkl"):
            src = baseline_src / name
            if src.exists() and not (dst / name).exists():
                shutil.copy(src, dst / name)
    barrier()


def make_config(
    n_params: int,
    seed: int,
    res_root: Path,
    experiment_id: str,
    cleanup_distributed: bool,
    do_elicit: bool,
) -> ExperimentConfig:
    """Build a 1 baseline + 5 maxent stage config.

    The baseline stage is a *placeholder*: ``<res_root>/<experiment_id>/baseline/``
    has been pre-populated with a hardlinked ``checkpoint.pth`` and a
    completed ``stage.json``, so ``run_experiments`` will skip training and
    just load the checkpoint into ``baseline_model`` for the downstream
    maxent stages.
    """
    align = 64 if n_params < 5e9 else 32
    config = GetRealisticConfig(n_params, align=align)
    config.run.find_unused_parameters = False

    baseline_stage = BaselineConfig(
        lr=get_lr(n_params),
        num_checkpoints=100 if n_params > 800_000_000 else -1,
        num_train_evals=100,
        do_elicit=False,
    )

    maxent_stages: list[MaxentConfig] = []
    for retain_target in RETAIN_TARGETS:
        lr, alpha_retain = _load_bo_best_params(n_params, seed, retain_target)
        maxent_stages.append(MaxentConfig(
            lr=lr,
            steps=MAXENT_STEPS,
            alpha_retain=alpha_retain,
            num_checkpoints=-1,
            do_elicit=do_elicit,
            retain_targets=[retain_target],
        ))

    config.stages = [baseline_stage, *maxent_stages]

    config.run.target_effective_batch_size = get_bs(n_params)
    config.run.seed = seed
    config.run.cleanup_distributed = cleanup_distributed
    config.run.res_root = res_root
    config.run.experiment_id = experiment_id
    config.run.log_level = "DEBUG"
    config.run.compile = True
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = (
        f"scaling/realistic/maxent/{make_param_str(n_params)}/seed_{seed}"
    )
    return config


def run(
    n_params: int,
    seed: int,
    cleanup_distributed: bool = True,
    do_elicit: bool = True,
) -> None:
    """Stage the reference baseline into a fresh experiment dir, then run 5 maxent stages."""
    param_str = make_param_str(n_params)
    lr = get_lr(n_params)
    eff_bs = get_bs(n_params)

    if seed not in BASELINE_EXPERIMENT_IDS:
        raise KeyError(
            f"No hardcoded reference baseline for seed={seed}; "
            f"known seeds: {sorted(BASELINE_EXPERIMENT_IDS)}"
        )
    baseline_experiment_id = BASELINE_EXPERIMENT_IDS[seed]
    baseline_src = _baseline_src_dir(n_params, seed, baseline_experiment_id)
    if not (baseline_src / "checkpoint.pth").exists():
        raise FileNotFoundError(
            f"Reference baseline checkpoint not found: {baseline_src}/checkpoint.pth"
        )

    res_root = _maxent_res_root(n_params, seed)
    # Fresh, timestamped experiment_id (NOT the baseline's) so each run writes
    # to its own dir and never overwrites or skip-resumes prior maxent results.
    # This mirrors the default RunConfig.experiment_id (also get_timestamp):
    # generated per-rank here, but setup() broadcasts rank 0's value to all
    # ranks, and only rank 0 stages the baseline, so the staged dir matches
    # the broadcast id. No changes to main/setup are required.
    experiment_id = get_timestamp()

    print(
        f"[scaling.realistic.maxent] N={n_params} ({param_str}) "
        f"lr={lr:.3e} eff_bs={eff_bs} seed={seed} "
        f"baseline_ref={baseline_experiment_id} experiment_id={experiment_id}"
    )

    _stage_baseline(baseline_src, res_root, experiment_id)

    config = make_config(
        n_params=n_params,
        seed=seed,
        res_root=res_root,
        experiment_id=experiment_id,
        cleanup_distributed=cleanup_distributed,
        do_elicit=do_elicit,
    )
    run_single(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_size", type=str, default="800M",
        help="Model size (default 800M; hardcoded reference baselines are 800M).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no_elicit", action="store_true",
        help="Disable adversarial FT on each maxent stage.",
    )

    args = parser.parse_args()
    run(
        n_params=parse_model_size(args.model_size.upper()),
        seed=args.seed,
        do_elicit=not args.no_elicit,
    )

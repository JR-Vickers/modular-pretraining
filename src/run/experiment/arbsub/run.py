"""
eval_arbsub-only re-evaluation of trained GR-MoE and LoRA 800M checkpoints
across seeds 1-3.

For each (method, seed):
  - locate the latest trained checkpoint at
    results/scaling/realistic/<method>/800M/seed_N/<ts>/routed/checkpoint.pth
  - hardlink it into a fresh res dir at
    results/arbsub/800M/seed_N/<method>/routed/checkpoint.pth
  - rebuild the model, restore the checkpoint, run do_eval against the full
    arbsub-expanded set of retain targets (N base + (N-1)-out + all-on)
  - write stats.jsonl under results/arbsub/800M/seed_N/<method>/

No training, no adversarial elicitation — just loss evaluations.

Usage (single-node):
    export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.arbsub.run
"""
import argparse
import logging
import os
import shutil
from pathlib import Path

import torch

from src.model.config import RoutedModelConfig
from src.model.lora import LoRATransformer
from src.model.moe import MoETransformer
from src.model.utils import make_model
from src.run.eval import do_eval
from src.run.experiment.common import (
    ROOT_DIR, get_bs, get_lr, make_param_str,
)
from src.run.experiment.config import GetRealisticConfig
from src.run.main import get_retain_targets
from src.run.train.routed import OrderedConfig, UnorderedConfig
from src.run.util.config import ExperimentConfig, setup
from src.run.util.distributed import (
    barrier, cleanup_distributed, is_main_process,
)
from src.run.util.s3 import stop_watcher, sync_to_s3
from src.run.util.state import (
    mark_stage_completed, restore_partial,
)

logger = logging.getLogger(__name__)

MODEL_SIZE = 800e6
SEEDS = (1, 2, 3)
METHODS = ("grmoe", "lora")
PARAM_STR = make_param_str(MODEL_SIZE)

# Match scaling/realistic/{grmoe,lora}/run.py exactly.
GRMOE = dict(
    core_param_prc=1.0, aux_param_prc=0.1,
    aux_factor={"code-lisp": 4.0, "papers-biology": 3.0,
                "papers-nuclear": 3.0, "papers-cyber": 3.0},
    robust_prc=0.2, aux_route_prc=0.5,
)
LORA = dict(
    core_param_prc=1.0, aux_param_prc=0.1,
    aux_factor={"code-lisp": 2.0, "papers-biology": 1.0,
                "papers-nuclear": 1.0, "papers-cyber": 1.0},
    core_aux_ratio=1.0,
)


# ---------------------------------------------------------------------------
# Source-checkpoint discovery + staging.
# ---------------------------------------------------------------------------

def _source_checkpoint(method: str, seed: int) -> Path:
    """Latest trained routed checkpoint for (method, seed) at MODEL_SIZE."""
    seed_dir = (ROOT_DIR / "scaling" / "realistic" / method
                / PARAM_STR / f"seed_{seed}")
    candidates = [
        d for d in seed_dir.iterdir()
        if d.is_dir() and (d / "routed" / "checkpoint.pth").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No routed/checkpoint.pth under {seed_dir}/<ts_id>/"
        )
    return max(candidates, key=lambda d: d.name) / "routed" / "checkpoint.pth"


def _stage_checkpoint(stage_dir: Path, src_ckpt: Path) -> None:
    """Hardlink src_ckpt into <stage_dir>/checkpoint.pth (rank 0 only)."""
    if is_main_process():
        stage_dir.mkdir(parents=True, exist_ok=True)
        dst = stage_dir / "checkpoint.pth"
        if not dst.exists():
            try:
                os.link(src_ckpt, dst)
            except OSError:
                shutil.copy(src_ckpt, dst)
    barrier()


# ---------------------------------------------------------------------------
# Config builder.
# ---------------------------------------------------------------------------

def _make_config(
    method: str, seed: int, *, cleanup_distributed_flag: bool,
) -> ExperimentConfig:
    align = 64  # 800M < 5e9
    config = GetRealisticConfig(MODEL_SIZE, align=align)

    common_kwargs = dict(
        lr=get_lr(MODEL_SIZE),
        num_checkpoints=-1,
        equal_compute=True,
        num_train_evals=0,
        do_elicit=False,        # no adversarial FT
        acc_mode="uniform",     # matches original 800M runs
        label_prc=1.0,
        eval_arbsub=True,       # the whole point
    )

    if method == "grmoe":
        stage = UnorderedConfig(
            model=RoutedModelConfig.from_base(
                config.model, arch="moe",
                core_param_prc=GRMOE["core_param_prc"],
                aux_param_prc=GRMOE["aux_param_prc"]),
            aux_factor=GRMOE["aux_factor"],
            robust_prc=GRMOE["robust_prc"],
            aux_route_prc=GRMOE["aux_route_prc"],
            **common_kwargs,
        )
    elif method == "lora":
        stage = OrderedConfig(
            model=RoutedModelConfig.from_base(
                config.model, arch="lora",
                core_param_prc=LORA["core_param_prc"],
                aux_param_prc=LORA["aux_param_prc"]),
            aux_factor=LORA["aux_factor"],
            core_aux_ratio=LORA["core_aux_ratio"],
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    config.stages = [stage]
    config.run.target_effective_batch_size = get_bs(MODEL_SIZE)
    config.run.seed = seed
    config.run.cleanup_distributed = cleanup_distributed_flag
    # Standard timestamped layout (mirrors scaling/realistic): method lives in
    # res_root, experiment_id defaults to a timestamp, so res_dir is
    # .../arbsub/<method>/<size>/seed_N/<timestamp>/ and each run is isolated
    # (reruns land in a fresh dir instead of appending to one another).
    config.run.res_root = ROOT_DIR / "arbsub" / method / PARAM_STR / f"seed_{seed}"
    config.run.log_level = "DEBUG"
    config.run.compile = True
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = f"arbsub/{method}/{PARAM_STR}/seed_{seed}"
    return config


# ---------------------------------------------------------------------------
# Eval-only orchestrator (mirrors run_routed's prep, skips training+elicit).
# ---------------------------------------------------------------------------

def _arbsub_eval_configs(config: ExperimentConfig, stage) -> list[dict]:
    """Mirror run_routed's eval-config building (with eval_arbsub=True)."""
    labels = config.run.labels
    aux_labels = config.data.aux.labels
    retain_targets = get_retain_targets(aux_labels)
    extra = [sorted(set(labels) - {x}) for x in aux_labels] + [labels]
    retain_targets += extra
    return [
        {
            "expert_labels": sorted(retained),
            "log": {
                "retained": sorted(retained),
                "finetune": None,
                "elicited": False,
            },
        }
        for retained in retain_targets
    ]


def run_arbsub_eval(method: str, seed: int, cleanup: bool) -> None:
    """Run eval_arbsub-only evaluation for one (method, seed)."""
    config = _make_config(method, seed, cleanup_distributed_flag=cleanup)
    src_ckpt = _source_checkpoint(method, seed)

    try:
        config = setup(config)
        log = config.run.logger
        log.info(f"=== arbsub eval: method={method} seed={seed} ===")
        log.info(f"Source checkpoint: {src_ckpt}")
        log.info(f"Output res_dir: {config.run.res_dir}")

        [stage] = config.stages
        _stage_checkpoint(stage.res_dir, src_ckpt)

        # Build model (DDP-wrapped + compiled).
        labels = config.run.labels
        if stage.model.arch == "moe":
            model_cls = MoETransformer
        elif stage.model.arch == "lora":
            model_cls = LoRATransformer
        else:
            raise ValueError(f"Unsupported arch: {stage.model.arch}")
        model = make_model(model_cls, stage.model, config.run,
                           extra_args={"labels": labels})

        # Load the staged checkpoint.
        model, _ = restore_partial(model, stage, config)
        model.eval()

        # Run do_eval for every (retain target) combo from eval_arbsub.
        eval_configs = _arbsub_eval_configs(config, stage)
        log.info(f"Running {len(eval_configs)} eval passes "
                 f"(arbsub-expanded retain targets)")
        for ec in eval_configs:
            do_eval(stage=stage, model=model, config=config, **ec)

        mark_stage_completed(stage, config)
        barrier()

        stop_watcher()
        sync_to_s3(config)
        log.info(f"Finished {method}/seed_{seed}. See {config.run.res_dir}")

    finally:
        stop_watcher()
        barrier()
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        if config.run.cleanup_distributed:
            cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args()

    # Sequential (method, seed) plan; cleanup_distributed only on the last.
    plan = [(m, s) for m in args.methods for s in args.seeds]
    for i, (method, seed) in enumerate(plan):
        is_last = (i == len(plan) - 1)
        run_arbsub_eval(method, seed, cleanup=is_last)


if __name__ == "__main__":
    main()

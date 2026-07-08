"""
Capability-titration eval of trained GR-MoE / FT-LoRA 800M checkpoints (AGI-2032).

Mirrors ``src/run/experiment/arbsub/run.py``: for each (method, seed) we rebuild
the routed model, restore the trained checkpoint, and evaluate test loss --- but
instead of binary retain-target evals we sweep a float forward-mask weight (the
*titration level* ``t`` in [0, 1]) on one aux module at a time:

    h_out = core(x) + t * aux(x)        (the slow path in MoE / LoRALinear)

``t = 0`` ablates the aux capability, ``t = 1`` fully enables it; at the
endpoints the float mask equals the binary expert mask used at training time.
For each (method, aux, t) we log test cross-entropy on the aux's own domain
(should drop as t rises) and on core (should stay flat).

The eval itself reuses the training-time protocol exactly: ``setup()`` builds
the same distributed test loaders (num_processes = num GPUs), and the eval loop
mirrors ``src.run.eval.eval_loss`` (full loader, reduce across ranks). So losses
are directly comparable to the original GRAM/FT-LoRA evals.

No training, no adversarial elicitation -- just loss evaluations.

Usage (single-node, 8 GPUs):
    export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.titration.run

Writes results/titration/<method>/800M/seed_N/<timestamp>/stats.jsonl.
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
from src.run.experiment.common import ROOT_DIR, get_lr, make_param_str
from src.run.experiment.config import GetRealisticConfig
from src.run.train.routed import OrderedConfig, UnorderedConfig
from src.run.util.config import ExperimentConfig, setup
from src.run.util.distributed import (
    barrier, cleanup_distributed, is_main_process, reduce_tensor,
)
from src.run.util.s3 import stop_watcher, sync_to_s3
from src.run.util.state import mark_stage_completed, restore_partial
from src.run.util.tools import get_batch, json_safe, log_line

logger = logging.getLogger(__name__)

MODEL_SIZE = 800e6
SEEDS = (1, 2, 3)
METHODS = ("grmoe", "lora")
TITRATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
PARAM_STR = make_param_str(MODEL_SIZE)

# Match scaling/realistic/{grmoe,lora}/run.py exactly (copied from arbsub/run.py).
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
# Source-checkpoint discovery + staging (same as arbsub/run.py).
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
        raise FileNotFoundError(f"No routed/checkpoint.pth under {seed_dir}/<ts_id>/")
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
# Config builder (mirrors arbsub/run.py, retargeted to results/titration/).
# ---------------------------------------------------------------------------

def _make_config(method: str, seed: int, *, cleanup_distributed_flag: bool) -> ExperimentConfig:
    config = GetRealisticConfig(MODEL_SIZE, align=64)

    common_kwargs = dict(
        lr=get_lr(MODEL_SIZE),
        num_checkpoints=-1,
        equal_compute=True,
        num_train_evals=0,
        do_elicit=False,        # no adversarial FT
        acc_mode="uniform",     # matches original 800M runs
        label_prc=1.0,
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
    # Pin the effective batch size to the 800M training value (672) rather than
    # get_bs(MODEL_SIZE)=641. This only affects the test-set token budget
    # (rounded to a multiple of eff_bs); pinning it makes the test set identical
    # to the original training eval regardless of how many GPUs we run on.
    config.run.target_effective_batch_size = 672
    config.run.seed = seed
    config.run.cleanup_distributed = cleanup_distributed_flag
    # Standard timestamped layout (mirrors scaling/realistic): method lives in
    # res_root, experiment_id defaults to a timestamp, so res_dir is
    # .../titration/<method>/<size>/seed_N/<timestamp>/ and each run is isolated
    # (reruns land in a fresh dir instead of appending to one another).
    config.run.res_root = ROOT_DIR / "titration" / method / PARAM_STR / f"seed_{seed}"
    config.run.log_level = "DEBUG"
    # Eval-only with a float forward mask (the slow MoE/LoRA path); skip
    # torch.compile to avoid recompiles/graph-breaks on that path. The speedup
    # comes from data-parallel sharding across the 8 GPUs, not compile.
    config.run.compile = False
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = f"titration/{method}/{PARAM_STR}/seed_{seed}"
    return config


# ---------------------------------------------------------------------------
# Titration eval: mirrors src.run.eval.eval_loss, but with a float mask.
# ---------------------------------------------------------------------------

def _titration_mask(labels: list[str], titrated_aux: str, t: float, device) -> torch.Tensor:
    """Per-module forward weights: core = 1, the titrated aux = t, rest = 0."""
    w = torch.zeros(len(labels), device=device)
    w[labels.index("core")] = 1.0
    w[labels.index(titrated_aux)] = float(t)
    return w


@torch.inference_mode()
def _titration_eval(model, config: ExperimentConfig, data_label: str,
                    mask: torch.Tensor) -> float:
    """Full-test-set mean CE with a fixed float forward mask, averaged across
    ranks. Same structure as src.run.eval.eval_loss (the only change is a
    precomputed float mask instead of a binary get_exp_mask)."""
    barrier()
    loader = config.run.loaders[data_label]["test"]
    loader.reset(epoch=0)
    n = len(loader)
    total = 0.0
    model.eval()
    for _ in range(n):
        x, y, _ = get_batch(loader)
        total += model(tokens=x, targets=y, fwd_mask=mask, bck_mask=mask)[1].item()
    loss = total / max(n, 1)
    return reduce_tensor(torch.tensor(loss, device=config.run.device)).item()


def run_titration_eval(method: str, seed: int, titrations, cleanup: bool) -> None:
    """Titration sweep for one (method, seed)."""
    config = _make_config(method, seed, cleanup_distributed_flag=cleanup)
    src_ckpt = _source_checkpoint(method, seed)

    try:
        config = setup(config)
        log = config.run.logger
        log.info(f"=== titration eval: method={method} seed={seed} ===")
        log.info(f"Source checkpoint: {src_ckpt}")

        [stage] = config.stages
        _stage_checkpoint(stage.res_dir, src_ckpt)

        labels = config.run.labels
        aux_labels = config.data.aux.labels
        model_cls = MoETransformer if stage.model.arch == "moe" else LoRATransformer
        model = make_model(model_cls, stage.model, config.run,
                           extra_args={"labels": labels})
        model, _ = restore_partial(model, stage, config)
        model.eval()

        device = config.run.device
        log_fp = config.run.res_dir / "stats.jsonl"
        for aux in aux_labels:
            for t in titrations:
                mask = _titration_mask(labels, aux, t, device)
                active = [lab for lab, w in zip(labels, mask.tolist()) if w > 0]
                for data_label in (aux, "core"):
                    loss = _titration_eval(model, config, data_label, mask)
                    if is_main_process():
                        log_line({
                            "stage": json_safe(stage),
                            "function": "titration_eval",
                            "name": method,
                            "seed": seed,
                            "data_label": data_label,
                            "loss": loss,
                            "titration": t,
                            "aux": aux,
                            "expert_labels": active,
                            "split": "test",
                        }, log_fp)
                        log.info(f"[{method}] aux={aux} t={t:.2f} {data_label} loss={loss:.4f}")

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
    parser.add_argument("--titrations", nargs="+", type=float, default=list(TITRATIONS))
    args = parser.parse_args()

    plan = [(m, s) for m in args.methods for s in args.seeds]
    for i, (method, seed) in enumerate(plan):
        is_last = (i == len(plan) - 1)
        run_titration_eval(method, seed, args.titrations, cleanup=is_last)


if __name__ == "__main__":
    main()

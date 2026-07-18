"""
export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.main

main.py — multiclass-gradient-routing pipeline.

Architecture
------------
The pipeline runs a sequence of *stages*, each producing checkpoints in its
own subdirectory under the results root.  Stages are defined as plain dicts
in a ``stages`` list (see the CLI block at the bottom for examples).

Execution flow::

    run()                   # entry point: sets up DDP, calls setup()
      -> run_experiments()  # iterates over stages, dispatches to runners
          -> run_baseline   # train from scratch on all data
          -> run_routed     # gradient-routing (moe / lora / demix)
          -> run_filtering  # retrain without target data (per target group)
          -> run_coreftaux  # core pretrain + per-label fine-tune
          -> run_unlearning # post-hoc unlearning (rmu / ascent / maxent)

Each stage runner is self-contained: it creates the model, delegates training
to the appropriate ``do_*`` function, and marks progress via ``stage.json``.
Auto-resume is handled inside the training loops by ``CheckpointManager``.

Preemption
----------
On SIGTERM (Slurm preemption), a global flag is set.  CheckpointManager
detects it via broadcast, saves a ``checkpoint_step-N.pth``, and the
process exits cleanly with ``sys.exit(0)``.  On requeue,
``run_experiments`` reads ``stage.json`` to skip completed stages, and
training loops call ``ckpt.load_latest()`` to resume from the
highest-step checkpoint.

Adversarial fine-tuning
-----------------------
Several stages optionally run "adversarial fine-tuning" (``ft_forget=True``):
after the main training, the model is fine-tuned on each forget-target label
individually and evaluated to measure how easily the model can re-learn the
information.  This is an *elicitation* metric.
"""

import dataclasses
import gc
import json
from pathlib import Path
from typing import Callable, Dict, Optional
import warnings
import torch

from src.model.base import BaseTransformer
from src.model.config import Transformer
from src.model.moe import MoETransformer
from src.model.lora import LoRATransformer
from src.model.demix import DemixTransformer
from src.model.utils import copy_model, make_model
from src.run.eval import do_eval
from src.run.util.tools import json_safe, labels_to_str
from src.run.util.config import ExperimentConfig, StageConfig, setup
from src.run.util.s3 import sync_to_s3, stop_watcher
from src.run.util.state import restore_partial, restore_partial_state
from src.run.util.distributed import (
    barrier,
    cleanup_distributed,
    get_raw_model,
    is_main_process,
)
from src.run.util.state import (
    get_completed_iterations,
    is_stage_completed,
    mark_iteration_completed,
    mark_stage_completed,
)

from src.run.train.base import BaselineConfig, FilteringConfig, do_train
from src.run.train.coreftaux import CoreftauxConfig, do_coreftaux
from src.run.train.finetune import do_finetune
from src.run.train.maxent import do_maxent
from src.run.train.ascent import do_gradient_ascent
from src.run.train.rmu import RmuConfig, do_rmu
from src.run.train.routed import (
    RoutedConfig,
    OrderedConfig,
    UnorderedConfig,
    do_routed_ordered,
    do_routed_unordered,
)
from src.run.train.demix import DemixConfig, do_demix

warnings.filterwarnings("ignore", message=r"(?s).*Online softmax is disabled.*", category=UserWarning)

def run_experiment(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    func: Callable,
    func_args: Optional[dict] = None,
    eval_configs: Optional[list[dict]] = None,
) -> Transformer:
    """
    Modify a model in some way, then evaluate the effects of the modification.

    Args:
        stage: Typed stage configuration.
        model: Model to train (may be DDP-wrapped).
        config: Run-level configuration (loaders, device, logger, etc.).
        func: Training function (e.g. do_train, do_coreftaux, do_finetune).
        func_args: Extra kwargs forwarded to *func*.
        eval_configs: List of eval kwarg dicts.  Each dict is unpacked into
            do_eval as a separate call.  Defaults to a single empty-dict call.

    Returns:
        The model after training.
    """
    if func_args is None:
        func_args = dict()
    if eval_configs is None:
        eval_configs = [{}]

    model = func(
        stage=stage,
        model=model,
        config=config,
        **func_args,
    )

    for ec in eval_configs:
        do_eval(
            stage=stage,
            model=model,
            config=config,
            **ec,
        )

    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_elicitation(
    stage: StageConfig,
    cur_dir: Path | None,
    model: Transformer,
    config: ExperimentConfig,
    data_labels: set[str] | list[str],
    expert_labels: list[str] | None = None,
    log_extra: dict | None = None,
) -> Transformer:
    """Elicitation: copy model per label, finetune, eval, cleanup."""

    model = get_raw_model(model).to("cpu", dtype=config.run.dtype)
    save_dir = cur_dir / "elicit"
    save_dir.mkdir(parents=True, exist_ok=True)

    for label in sorted(data_labels):

        ft_model = copy_model(model, config.run)

        log_args = {"finetune": label, "elicited": True}
        if log_extra:
            log_args.update(log_extra)

        do_finetune(
            stage=stage,
            model=ft_model,
            config=config,
            data_labels=[label],
            expert_labels=expert_labels,
            log_args=log_args,
        )

        del ft_model
        torch.cuda.empty_cache()

    return model


def get_retain_targets(aux_labels: list[str]) -> list[list[str]]:
    """Get the list of retain targets for the auxiliary labels."""
    return [["core"]] + [["core", x] for x in aux_labels]


def get_routed_retain_targets(stage: RoutedConfig, config: ExperimentConfig) -> list[list[str]]:
    """Resolve routed evaluation profiles, including explicit smoke overrides."""
    labels = config.run.labels
    if stage.retain_targets is not None:
        return stage.retain_targets
    if stage.model.arch == "demix":
        return [["core"]] + [[x] for x in sorted(config.data.aux.labels)]
    retain_targets = get_retain_targets(config.data.aux.labels)
    if stage.eval_arbsub:
        retain_targets += [
            sorted(set(labels) - {x}) for x in config.data.aux.labels
        ] + [labels]
    return retain_targets
    

# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_baseline(
    stage: BaselineConfig,
    config: ExperimentConfig
) -> Transformer:
    """Train the baseline model. Returns the model on CPU."""

    logger = config.run.logger
    labels = config.run.labels

    logger.info("BASELINE START")

    model = make_model(BaseTransformer, config.model, config.run)
    model, state = restore_partial(model, stage, config) # resume from partial training

    model = run_experiment(
        stage=stage,
        model=model,
        config=config,
        func=do_train,
        func_args={
            "train_labels": sorted(labels),
            "state": state
        },
        eval_configs=[{"log": {
            "retained": sorted(labels),
            "finetune": None,
            "elicited": False,
        }}],
    )

    if stage.do_elicit:
        import gc
        # Free GPU memory before elicitation (state holds checkpoint optimizer states)
        del state
        model = get_raw_model(model).to("cpu", dtype=config.run.dtype)
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("BASELINE - ADVERSARIAL FT START")
        aux_labels = config.data.aux.labels
        run_elicitation(
            stage=stage,
            cur_dir=stage.res_dir,
            model=model,
            config=config,
            data_labels=aux_labels,
            log_extra={"retained": sorted(labels)})

    mark_stage_completed(stage, config)
    barrier()

    return model


def run_unlearning(
    stage: StageConfig,
    config: ExperimentConfig,
    baseline_model: Transformer,
) -> None:
    """Post-hoc unlearning (rmu / ascent / maxent).

    Unlearning is fast (~1% of training time), so there's no mid-training
    checkpointing or resume.  We just save the final model to a subdir
    named by the retained capabilities and use iteration tracking to skip
    completed iterations if the job restarts.
    """

    logger = config.run.logger
    labels = config.run.labels
    completed_iterations = get_completed_iterations(stage)
    retain_targets = get_retain_targets(config.data.aux.labels)
    if stage.retain_targets is not None:
        retain_targets = stage.retain_targets
        logger.info(f"Using stage-level retain_targets override ({len(retain_targets)} targets)")

    logger.info(f"UNLEARNING - {stage.name} START")
    assert baseline_model is not None, "Baseline model is required for posthoc unlearning"

    unlearn_funcs: Dict[str, Callable] = {
        "rmu": do_rmu,
        "ascent": do_gradient_ascent,
        "maxent": do_maxent,
    }

    for retained in retain_targets:
        removed = sorted(set(labels) - set(retained))
        if not removed:
            continue

        iter_name = labels_to_str(retained)

        if iter_name in completed_iterations:
            logger.info(f"Skipping completed unlearning iteration: {iter_name}")
            continue

        logger.info(f"Unlearning: retaining {sorted(retained)}, removing {removed}")

        iter_dir = stage.res_dir / iter_name

        model = copy_model(baseline_model, config.run)
        func_args: dict = {"data_labels": removed}
        if isinstance(stage, RmuConfig):
            func_args["frozen_model"] = copy_model(model, config.run).eval()

        model = run_experiment(
            stage=stage, model=model, config=config,
            func=unlearn_funcs[stage.name], func_args=func_args,
            eval_configs=[{"log": {
                "retained": sorted(retained), 
                "finetune": None, 
                "elicited": False
            }}],
        )

        if "frozen_model" in func_args:
            del func_args["frozen_model"]
            torch.cuda.empty_cache()

        # Persist the unlearned model
        if is_main_process():
            iter_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = iter_dir / "checkpoint.pth"
            tmp_path = ckpt_path.with_suffix(".pth.tmp")
            torch.save({"model": get_raw_model(model).state_dict()}, str(tmp_path))
            tmp_path.replace(ckpt_path)
            logger.info(f"Saved unlearned checkpoint: {ckpt_path}")
        barrier()

        if stage.do_elicit:
            model = get_raw_model(model).to("cpu", dtype=config.run.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"UNLEARNING - {stage.name} - ADVERSARIAL FT START")
            model = run_elicitation(
                stage=stage,
                cur_dir=iter_dir,
                model=model,
                config=config,
                data_labels=removed,
                log_extra={"retained": sorted(retained)},
            )
            model = model.to(config.run.device, dtype=config.run.dtype)

        barrier()

        del model
        torch.cuda.empty_cache()

        mark_iteration_completed(stage, iter_name, config)
        barrier()

    mark_stage_completed(stage, config)
    barrier()


def run_filtering(
    stage: FilteringConfig,
    config: ExperimentConfig,
) -> None:
    """Data-filtering stage: retrain from scratch on only the retained labels."""

    logger = config.run.logger
    labels = config.run.labels

    logger.info("FILTERING START")

    retain_targets = get_retain_targets(config.data.aux.labels)
    if stage.retain_targets is not None:
        retain_targets = stage.retain_targets
        logger.info(f"Using stage-level retain_targets override ({len(retain_targets)} targets)")

    completed_iterations = get_completed_iterations(stage)
    if completed_iterations:
        logger.info(f"Filtering: iterations already completed: {completed_iterations}")

    for retained in retain_targets:

        removed = sorted(set(labels) - set(retained))
        if not removed:
            continue

        iter_name = labels_to_str(retained)

        if iter_name in completed_iterations:
            logger.info(f"Skipping completed filtering iteration: {iter_name}")
            continue

        logger.info(f"Filtering: retaining {retained}, removing {removed}")

        model = make_model(BaseTransformer, config.model, config.run)

        iter_dir = stage.res_dir / iter_name
        iter_stage = dataclasses.replace(stage, res_dir=iter_dir)
        model, state = restore_partial(model, iter_stage, config)

        model = run_experiment(
            stage=iter_stage,
            model=model,
            config=config,
            func=do_train,
            func_args={"train_labels": retained, "state": state},
            eval_configs=[{"log": {
                "retained": sorted(retained),
                "finetune": None,
                "elicited": False
            }}],
        )

        if stage.do_elicit:
            import gc
            # Free ALL GPU memory: optimizer state, DDP buffers, model
            del state
            model = get_raw_model(model).to("cpu", dtype=config.run.dtype)
            del model
            gc.collect()
            torch.cuda.empty_cache()

            # Reload clean model from checkpoint for elicitation
            model = make_model(BaseTransformer, config.model, config.run)
            ckpt_state = restore_partial_state(iter_stage, config)
            if ckpt_state and "model" in ckpt_state:
                get_raw_model(model).load_state_dict(ckpt_state.pop("model"))
            del ckpt_state  # discard optimizer state — not needed for elicitation
            gc.collect()
            torch.cuda.empty_cache()
            model = get_raw_model(model).to("cpu", dtype=config.run.dtype)

            logger.info("FILTERING - ADVERSARIAL FT START")
            model = run_elicitation(
                stage=stage,
                cur_dir=iter_dir,
                model=model,
                config=config,
                data_labels=removed,
                log_extra={"retained": sorted(retained)},
            )

        del model
        torch.cuda.empty_cache()

        mark_iteration_completed(stage, iter_name, config)
        barrier()

    mark_stage_completed(stage, config)
    barrier()


def run_coreftaux(
    stage: CoreftauxConfig,
    config: ExperimentConfig,
) -> None:
    """Model branching: core pretraining + aux fine-tuning."""

    logger = config.run.logger
    aux_labels = config.data.aux.labels

    completed_iterations = get_completed_iterations(stage)
    if completed_iterations:
        logger.info(f"Coreftaux: iterations already completed: {completed_iterations}")

    logger.info("CORE FT AUX - BASE PHASE START")

    model = make_model(BaseTransformer, config.model, config.run)

    core_dir = stage.res_dir / "core"
    core_stage = dataclasses.replace(stage, res_dir=core_dir)
    model, core_state = restore_partial(model, core_stage, config)

    if "core" in completed_iterations:
        logger.info("Skipping completed coreftaux iteration: core")
    else:
        model = run_experiment(
            stage=core_stage,
            model=model,
            config=config,
            func=do_coreftaux,
            func_args={
                "phase": "base",
                "data_label": "core",
                "state": core_state,
            },
            eval_configs=[{"log": {
                "retained": ["core"],
                "finetune": None,
                "elicited": False,
            }}],
        )

        if core_stage.do_elicit:
            model = get_raw_model(model).to("cpu", dtype=config.run.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"CORE FT AUX - BASE PHASE - ADVERSARIAL FT START")
            run_elicitation(
                stage=core_stage,
                cur_dir=core_dir,
                model=model,
                config=config,
                data_labels=aux_labels,
                log_extra={"retained": ["core"]},
            )

        mark_iteration_completed(stage, "core", config)
        barrier()

    model = get_raw_model(model).to("cpu", dtype=config.run.dtype)

    for label in aux_labels:

        retained = ["core", label]
        iter_name = labels_to_str(retained)
        removed = sorted(set(aux_labels) - {label})

        if iter_name in completed_iterations:
            logger.info(f"Skipping completed coreftaux iteration: {iter_name}")
            continue

        ft_model = copy_model(model, config.run)

        logger.info(f"CORE FT AUX - AUX PHASE: {label}")

        iter_dir = stage.res_dir / iter_name
        iter_stage = dataclasses.replace(stage, res_dir=iter_dir)
        ft_model, ft_state = restore_partial(ft_model, iter_stage, config)

        ft_model = run_experiment(
            stage=iter_stage,
            model=ft_model,
            config=config,
            func=do_coreftaux,
            func_args={
                "phase": "ft",
                "data_label": label,
                "state": ft_state,
            },
            eval_configs=[{"log": {    
                "retained": sorted(retained),
                "finetune": None,
                "elicited": False,
            }}],
        )

        if stage.do_elicit:
            ft_model = get_raw_model(ft_model).to("cpu", dtype=config.run.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"CORE FT AUX - AUX PHASE: {label} - ADVERSARIAL FT START")
            run_elicitation(
                stage=stage,
                cur_dir=iter_dir,
                model=ft_model,
                config=config,
                data_labels=removed,
                log_extra={"retained": sorted(retained)},
            )

        mark_iteration_completed(stage, iter_name, config)
        barrier()

        del ft_model
        torch.cuda.empty_cache()

    mark_stage_completed(stage, config)
    barrier()

    del model
    torch.cuda.empty_cache()


def run_routed(
    stage: RoutedConfig,
    config: ExperimentConfig,
) -> None:
    """Routed (gradient-routing) stage: demix / moe / lora."""

    logger = config.run.logger
    labels = config.run.labels

    logger.info("ROUTED START")

    func = None

    if stage.model.arch == "moe":
        model = make_model(MoETransformer, stage.model, config.run, extra_args={"labels": labels})

    elif stage.model.arch == "lora":
        model = make_model(LoRATransformer, stage.model, config.run, extra_args={"labels": labels})

    elif stage.model.arch == "demix":
        model = make_model(DemixTransformer, stage.model, config.run, extra_args={"labels": labels})

    else:
        raise ValueError(f"Unsupported routed arch: {stage.model.arch}")

    if isinstance(stage, DemixConfig):
        func = do_demix

    elif isinstance(stage, OrderedConfig):
        func = do_routed_ordered

    elif isinstance(stage, UnorderedConfig):
        func = do_routed_unordered

    else:
        raise ValueError(f"Unknown routed config type: {type(stage).__name__}")

    retain_targets = get_routed_retain_targets(stage, config)
    if stage.retain_targets is not None:
        logger.info(f"Using stage-level retain_targets override ({len(retain_targets)} targets)")

    eval_configs = []
    for retained in retain_targets:
        eval_configs.append({
            "expert_labels": sorted(retained),
            "log": {
                "retained": sorted(retained),
                "finetune": None,
                "elicited": False
        }})

    model, state = restore_partial(model, stage, config)

    model = run_experiment(
        stage=stage, 
        model=model,
        config=config,
        func=func,
        func_args={"state": state},
        eval_configs=eval_configs,
    )

    # Adversarial FT: separate loop because it involves model copies and cleanup
    if stage.do_elicit:
        model = get_raw_model(model).to("cpu", dtype=config.run.dtype)
        gc.collect()
        torch.cuda.empty_cache()

        for ec in eval_configs:

            retained = ec["expert_labels"]
            if stage.model.arch == "demix":
                # Single active expert (mirrors grmoe): elicit the *other* aux
                # domains. Core is always retained, and the active expert's own
                # domain is excluded, so removed = aux_labels - {active expert}.
                # e.g. [core]->a,b,c,d ; [a]->b,c,d ; [b]->a,c,d ; ...
                removed = sorted(set(config.data.aux.labels) - set(retained))
            else:
                removed = sorted(set(labels) - set(retained))
            if not removed: continue

            logger.info("ROUTED - ADVERSARIAL FT START")
            iter_dir = stage.res_dir / labels_to_str(retained)

            model = run_elicitation(
                stage=stage,
                cur_dir=iter_dir,
                model=model,
                config=config,
                data_labels=removed,
                log_extra={"retained": sorted(retained)},
                expert_labels=ec["expert_labels"],
            )

    del model
    torch.cuda.empty_cache()

    mark_stage_completed(stage, config)
    barrier()


# ---------------------------------------------------------------------------
# Pipeline dispatcher
# ---------------------------------------------------------------------------

def run_experiments(config: ExperimentConfig) -> None:
    """Execute the full multi-stage pipeline.

    Stages are sorted so "baseline" always runs first (downstream stages
    may need the baseline model as a starting point).  Directory names are
    computed up front by get_stage_dirs() so that resume can match stages
    to their on-disk directories deterministically.
    """

    device = config.run.device
    logger = config.run.logger

    # Baseline must run first — other stages may clone from its checkpoint.
    stages = config.stages
    stages = sorted(stages, key=lambda c: 0 if c.name == "baseline" else 1)
    baseline_model = None

    # Check if any unlearning stage still needs the baseline model
    _unlearning_names = {"rmu", "ascent", "maxent"}
    _needs_baseline = any(
        s.name in _unlearning_names and not is_stage_completed(s)
        for s in stages
    )

    for stage in stages:

        if is_stage_completed(stage):

            logger.info(f"Skipping completed stage {stage.name} @ {stage.res_dir}")

            if stage.name == "baseline" and baseline_model is None and _needs_baseline:
                baseline_ckpt = stage.res_dir / "checkpoint.pth"
                if baseline_ckpt.exists():
                    logger.info(f"Loading baseline model from {baseline_ckpt} for downstream stages")
                    baseline_model = make_model(BaseTransformer, config.model, config.run)
                    checkpoint = torch.load(str(baseline_ckpt), map_location=device, weights_only=False)
                    get_raw_model(baseline_model).load_state_dict(checkpoint['model'])
                    del checkpoint
                else:
                    logger.warning(f"Baseline checkpoint not found at {baseline_ckpt}, cannot load")
            continue

        stage_dict = json_safe(stage)
        logger.info(f"Stage [{stage.name}] start: {json.dumps(stage_dict, default=str, ensure_ascii=False)}")

        if stage.name == "baseline":
            baseline_model = run_baseline(stage, config)

        elif stage.name in ("rmu", "ascent", "maxent"):
            run_unlearning(stage, config, baseline_model)

        elif stage.name == "filtering":
            run_filtering(stage, config)

        elif stage.name == "coreftaux":
            run_coreftaux(stage, config)

        elif stage.name == "routed":
            run_routed(stage, config)

        else:
            raise ValueError(f"Unknown stage name: {stage.name}")


def run(config: ExperimentConfig):
    """Run the gradient-routing pipeline.

    Args:
        config: An ExperimentConfig instance.

    Returns:
        DataFrame of eval stats (empty if no stats were written).
    """
    
    try:
        config = setup(config)

        logger = config.run.logger
        loaders = config.run.loaders
        for key, loader in loaders.items():
            logger.info(f"Loader {key} train: {len(loader['train'])}, test: {len(loader['test'])}")

        run_experiments(config)

        # Flush watcher, then do a final full sync to catch anything missed
        stop_watcher()
        sync_to_s3(config)

        logger.info("-" * 40)
        logger.info(f"Finished. See {config.run.res_dir}")

    finally:
        stop_watcher()  # ensure cleanup even on exception
        barrier()
        # Clear torch.compile / dynamo caches so sequential runs in run_all
        # don't accumulate stale guards and compiled graphs from prior model sizes.
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        if config.run.cleanup_distributed:
            cleanup_distributed()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    torch.cuda.empty_cache()

    root_dir = Path("src").absolute()
    res_root = root_dir.parent / "results" / "test-runs" / "realistic"

    from src.run.experiment.config import GetRealisticConfig

    n_params = 50e6
    config = GetRealisticConfig(n_params)
    power_laws_path = Path(__file__).resolve().parents[2] / "analysis" / "optimize" / "base" / "power_laws.json"
    power_laws = json.loads(power_laws_path.read_text())
    LR = power_laws["lr"]["coef"] * (n_params ** power_laws["lr"]["exp"])
    BS = round(power_laws["bs"]["coef"] * (n_params ** power_laws["bs"]["exp"]))

    config.run.target_effective_batch_size = BS
    config.run.seed = 1
    config.run.log_level = "DEBUG"
    config.run.res_root = res_root
    config.run.compile = True
    base_model = config.model

    config.stages = [
        BaselineConfig(
            num_train_evals=0,
            do_elicit=False,
            lr=LR,
            acc_mode="uniform",
        ),
    ]

    run(config)

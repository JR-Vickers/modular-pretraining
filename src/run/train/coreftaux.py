"""
Core-then-fine-tune-aux (CoreFTAux) training loop.

This implements a two-phase training strategy for measuring knowledge
compartmentalization without architecture-level routing:

Phase 1 — Core pretraining (phase="base"):
    Train a dense model on core data only, producing a foundation model
    that has never seen any aux-label data.

Phase 2 — Per-label fine-tuning (phase="ft"):
    Starting from the core checkpoint, fine-tune on each aux label separately
    (with core data mixed in to prevent catastrophic forgetting).  This
    produces N independent models, one per label, that can be evaluated to
    measure how much each label's knowledge transfers or interferes.

The optimizer state persists across phases (passed via ``state``), but
the LR scheduler is reset to a fresh warmup→constant→decay cycle for
each phase.

Data mixing in Phase 2 is controlled by ``ft_aux_prc`` and ``base_prc``:
the batch sequence is split into a base segment (core only) followed by
an aux+core segment, with the aux fraction within that segment set by
``ft_aux_prc``.
"""

import pickle
import contextlib
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Literal
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from src.model.base import BaseTransformer
from src.run.util.config import ExperimentConfig, StageConfig, use_fused_adamw
from src.run.util.distributed import barrier, broadcast_object, is_main_process, get_rank
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.preemption import is_preempted
from src.run.util.tools import get_batch, log_batch_counts, set_seeds
from src.run.util.state import save_checkpoint, should_save
from src.run.eval import eval_loss


@dataclass
class CoreftauxConfig(StageConfig):
    """Configuration for the core-then-fine-tune-aux stage."""
    name: str = "coreftaux"
    aux_factor: float | dict[str, float] = 1.0
    core_aux_ratio: float = 1.0
    label_prc: float = 1.0
    ft_lr_factor: float = 0.25


def do_coreftaux(
    stage: CoreftauxConfig,
    model: BaseTransformer,
    config: ExperimentConfig,
    phase: Literal["base", "ft"] = "base",
    data_label: str = "core",
    state: dict | None = None,
) -> BaseTransformer:
    """Train a dense model with ordered core-then-aux data mixing.

    Called twice per label in the coreftaux stage:
    1. Base phase (phase="base"): train on core data (optionally mixed with
       a fraction of aux data via base_aux_prc).
    2. FT phase (phase="ft", data_label="biology"): continue from the base
       checkpoint, mixing core and aux batches controlled by ft_aux_prc.

    The optimizer state is persisted across phases via ``state["opts"]``,
    but the LR scheduler resets to a fresh warmup→constant→decay cycle
    for each phase.

    Args:
        stage: CoreFTAux stage config.
        model: Dense transformer to train.
        config: Experiment config.
        phase: "base" for core pretraining, "ft" for per-label fine-tuning.
        data_label: Which label to fine-tune on (required for phase="ft").
        state: Resume state dict. Carries optimizer state between phases.

    Returns:
        Trained model.
    """

    # unpack config/stage
    acc_steps = config.run.accumulation_steps
    loaders = config.run.loaders
    epochs = config.run.epochs
    logger = config.run.logger
    adam_betas = config.run.adam_betas
    labels = config.run.labels
    aux_labels = config.data.aux.labels
    seed = config.run.seed
    is_ddp = config.run.is_ddp
    warmup_prc = config.run.warmup_prc
    decay_prc = config.run.decay_prc

    ft_lr_factor = stage.ft_lr_factor
    lr = stage.lr if phase == "base" else stage.lr * ft_lr_factor
    aux_factor = stage.aux_factor # what percent of the aux data do you use
    core_aux_ratio = stage.core_aux_ratio # what percent of the ft phase is made of core data

    num_evals = stage.num_train_evals
    num_checkpoints = stage.num_checkpoints
    res_dir = stage.res_dir if phase == "base" else stage.res_dir / data_label

    all_labels = config.run.labels
    label_prc = stage.label_prc
    acc_mode = stage.acc_mode

    logger.info(f"---- Begin CoreFTAux | Phase: {phase} | Data Label: {data_label} ----")
    logger.debug(f"acc_mode: {acc_mode}, label_prc: {label_prc}, lr: {lr}")

    set_seeds(seed)

    model.train()

    if isinstance(aux_factor, float):
        factor = aux_factor
        aux_factor = {label: factor for label in aux_labels}

    assert all(val > 0.0 for val in aux_factor.values()), f"all aux_factor values must be > 0.0"
    assert 0 <= core_aux_ratio, f"0 <= core_aux_ratio"
    assert 0 <= label_prc <= 1, f"0 <= label_prc <= 1"
    assert lr > 0, "LR must be provided"

    if phase == "ft":
        assert data_label != "core", "data_label cannot be 'core' for ft phase"

    if state is None:
        state = dict()

    # --- calculate batches ---

    len_all = len(loaders["all"]["train"])
    len_all_labeled = round(len_all * label_prc)
    len_all_unlabeled = len_all - len_all_labeled

    len_aux = 0
    for label, factor in aux_factor.items():
        len_aux += round(len(loaders[label]["train"]) * label_prc * factor)
    assert len_aux <= len_all_labeled, f"len_aux <= len_all_labeled, ({len_aux} <= {len_all_labeled})"

    len_aux_for_ft = len_aux
    len_core_for_ft = round(len_aux_for_ft * core_aux_ratio)
    len_ft = len_aux_for_ft + len_core_for_ft
    len_base = len_all_labeled - len_ft

    labeled_batches = {}
    labeled_batches["core"] = ["core"] * len_base

    aux_proportions = {label: len(loaders[label]["train"]) * aux_factor[label] for label in aux_labels}
    aux_proportions = {k: v / sum(aux_proportions.values()) for k, v in aux_proportions.items()}

    for label in aux_labels:

        cur_aux_prc = aux_proportions[label]
        cur_aux_ft_samples = round(len_aux_for_ft * cur_aux_prc)
        cur_core_ft_samples = round(len_core_for_ft * cur_aux_prc)

        cur_ft_batches = []
        cur_ft_batches += [label]  * cur_aux_ft_samples
        cur_ft_batches += ["core"] * cur_core_ft_samples

        labeled_batches[label] = cur_ft_batches

    unlabeled_batches = []
    for label in all_labels:
        proportion = len(loaders[label]["train"]) / len_all
        unlabeled_batches += [label] * round(proportion * len_all_unlabeled)

    logger.info(
        f"\nlen_all: {len_all}, len_all_labeled: {len_all_labeled}, len_all_unlabeled: {len_all_unlabeled},"
        f"\nlen_base: {len_base}, len_ft: {len_ft},"
        f"\nlen_core_for_ft: {len_core_for_ft}, len_aux_for_ft: {len_aux_for_ft},"
        f"\naux_factor: { {k: round(v, 4) for k, v in aux_factor.items()} }, core_aux_ratio: {round(core_aux_ratio, 4)},"
    )

    batches = labeled_batches[data_label]
    if phase == "base":
        batches += unlabeled_batches

    batches = broadcast_object(batches, src=0)
    
    if acc_mode == "uniform":
        random.seed(seed)
        count = Counter(batches)
        temp = []
        for label in count.keys():
            temp += [label] * round(count[label] / acc_steps)
        random.shuffle(temp)
        batches = []
        for label in temp:
            batches += [label] * acc_steps
        batches = broadcast_object(batches, src=0)

    else: #heterogeneous
        random.seed(get_rank())
        random.shuffle(batches) #different per rank
        if len(batches) % acc_steps != 0: #if not multiple of acc_steps, truncate
            batches = batches[:-(len(batches) % acc_steps)]

    log_batch_counts(batches, logger)

    # --- do training ---

    # calculate total steps
    num_total_steps = (len(batches) // acc_steps) * epochs
    logger.info(f"num_total_steps: {num_total_steps}")
    eval_freq = max(1, round(num_total_steps / num_evals)) if num_evals > 0 else -1
    checkpoint_freq = max(1, round(num_total_steps / num_checkpoints)) if num_checkpoints > 0 else -1
    resume_step = state.get("step", -1)

    # Training already complete — return immediately
    if resume_step >= num_total_steps:
        logger.info(f"Training already complete (resume_step={resume_step} >= num_total_steps={num_total_steps}), skipping")
        return model

    # setup optimizer
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, fused=use_fused_adamw(config.run.device), betas=adam_betas
    )

    # restore optimizer
    if "opts" in state:
        opt.load_state_dict(state["opts"]["all"])
        opt.param_groups[0]['lr'] = lr
        del state["opts"]

    # setup losses
    losses = {"train": {}, "val": {}}
    for label in labels:
        losses["train"][label] = []
        losses["val"][label] = []

    # restore losses (each entry is a (step, loss) tuple)
    if "losses" in state:
        for label in losses["train"]:
            losses["train"][label] = [tuple(x) for x in state["losses"]["train"].get(label, [])]
        for label in losses["val"]:
            losses["val"][label] = [tuple(x) for x in state["losses"]["val"].get(label, [])]

        logger.info(f"Restored {sum(len(v) for v in losses['train'].values())} train loss entries")
        logger.info(f"Restored {sum(len(v) for v in losses['val'].values())} val loss entries")

    end_factor = 1e-8 / lr
    warmup_steps = round(warmup_prc * num_total_steps)
    decay_steps = round(decay_prc * num_total_steps)
    constant_steps = num_total_steps - warmup_steps - decay_steps

    logger.info(f"warmup_steps: {warmup_steps}")
    logger.info(f"constant_steps: {constant_steps}")
    logger.info(f"decay_steps: {decay_steps}")

    # LambdaLR: three-phase schedule (warmup → constant → decay)
    def lr_lambda(current_step):

        if current_step < warmup_steps:
            if current_step == 0:
                logger.info("LR Scheduler: Warmup Start")
            return end_factor + (1.0 - end_factor) * (current_step / warmup_steps)

        elif current_step < warmup_steps + constant_steps:
            if current_step == warmup_steps:
                logger.info("LR Scheduler: Constant Start")
            return 1.0

        else:
            if current_step == warmup_steps + constant_steps:
                logger.info("LR Scheduler: Decay Start")
            if decay_steps == 0:
                return end_factor
            decay_progress = (current_step - warmup_steps - constant_steps) / decay_steps
            return 1.0 - (1.0 - end_factor) * decay_progress

    scheduler = LambdaLR(opt, lr_lambda)
    cur_lr = scheduler.get_last_lr()[0]
    opt.param_groups[0]['lr'] = cur_lr

    # restore scheduler position from step count
    if resume_step > 0:
        scheduler.last_epoch = resume_step
        cur_lr = scheduler.get_lr()[0]
        scheduler._last_lr = [cur_lr]
        opt.param_groups[0]['lr'] = cur_lr
        logger.info(f"Restored scheduler to step {resume_step}, LR: {cur_lr:.6e}")

    # setup progress bar
    pbar = tqdm(total=num_total_steps, **get_tqdm_kwargs(logger, ncols=150))

    # training loop
    step = 0
    for epoch_idx in range(epochs):

        # reset loader for each epoch
        for label in all_labels:
            loaders[label]["train"].reset(epoch_idx)

        for batch_idx in range(len(batches)):

            access_idx = batch_idx
            if acc_mode == "uniform":
                access_idx = batch_idx // acc_steps * acc_steps

            loader_name = batches[access_idx]
            loader = loaders[loader_name]["train"]

            x, y, batch_label = get_batch(loader)

            if step < resume_step:
                step += 1
                continue

            acc_idx = batch_idx % acc_steps
            is_last_acc = acc_idx == acc_steps - 1
            no_sync_ctx = model.no_sync() if (is_ddp and not is_last_acc) else contextlib.nullcontext()

            with no_sync_ctx:

                _, loss = model.forward(
                    tokens=x,
                    targets=y,
                )

                scaled_loss = loss / acc_steps
                scaled_loss.backward()

            # train loss logging
            cur_loss = loss.item()
            losses["train"][batch_label].append((step, cur_loss))

            # update progress bar
            label_idx = labels.index(batch_label)
            desc_str = f"AC {acc_idx + 1}/{acc_steps} | LR: {cur_lr:.2e} | L: {cur_loss:.2f} | LB: {label_idx}"
            pbar.set_description(desc_str)
            pbar.refresh()

            if is_last_acc:

                step += 1
                pbar.update()

                # step the optimizer
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

                # update learning rate
                scheduler.step()
                cur_lr = scheduler.get_last_lr()[0]
                opt.param_groups[0]['lr'] = cur_lr

                # measure validation loss
                if eval_freq > 0 and step % eval_freq == 0:
                    model.eval()
                    for label in labels:
                        val_loss = eval_loss(
                            model, config, 
                            data_label=label, 
                            num_batches=50, 
                            shuffle_seed=step,
                        )
                        losses["val"][label].append((step, val_loss))
                    model.train()

                # logger printout
                if (step == 1) or (step % 1000 == 0) or (step == num_total_steps):

                    logger.info(f"Step: {step}, LR: {cur_lr:.2e}")

                    train_loss_str = ""
                    for label in losses["train"].keys():
                        label_str = label.upper()
                        loss_slice = [v for _, v in losses["train"][label][-10:]]
                        avg_train_loss = np.mean(loss_slice) if len(loss_slice) > 0 else float('nan')
                        train_loss_str += f"{label_str}: {avg_train_loss:.2f} "
                    logger.info(f"Train Loss: {train_loss_str}")

                    if eval_freq > 0:
                        val_loss_str = ""
                        for label in losses["val"].keys():
                            label_str = label.upper()
                            val_loss = losses["val"][label][-1][1] if len(losses["val"][label]) > 0 else float('nan')
                            val_loss_str += f"{label_str}: {val_loss:.2f} "
                        logger.info(f"Val Loss: {val_loss_str}")

                # save checkpoint periodically, at final step, or on preemption
                if should_save(step, num_total_steps, checkpoint_freq):

                    ckpt_state = {
                        'opts': {'all': opt.state_dict()},
                        'step': step,
                        'total_steps': num_total_steps,
                        'data_label': data_label,
                        'losses': {
                            "train": {label: np.array(vals) for label, vals in losses["train"].items()},
                            "val": {label: np.array(vals) for label, vals in losses["val"].items()},
                        },
                    }
                    
                    save_checkpoint(
                        stage=stage,
                        model=model,
                        state=ckpt_state,
                        config=config,
                    )

                    if is_preempted():
                        logger.warning("Preemption detected — checkpoint saved, exiting")
                        sys.exit(0)

                    # Dump per-label loss curves as pickle for post-hoc analysis / plotting
                    if is_main_process() and step == num_total_steps:
                        res_dir.mkdir(parents=True, exist_ok=True)
                        losses_path = res_dir / "losses.pkl"
                        losses_path.write_bytes(pickle.dumps(ckpt_state["losses"]))

        # sync after epoch
        barrier()

    # close progress bar
    pbar.close()

    return model

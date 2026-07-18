"""
Heterogeneous training loop.
Standard (non-routed) training loop.

This is the simplest training loop in the pipeline — a single optimizer over
all model parameters, no expert routing, no per-label gradient control.  Used
for baseline pretraining and the data-filtering stage (retrain from scratch
on a subset of labels).

Key design points:

- **Single AdamW optimizer** with a three-phase LR schedule:
  10% warmup (linear ramp), 80% constant, 10% linear decay.
- **Loss pickle dumps**: per-label loss histories are saved alongside
  checkpoints for offline analysis and plotting.
- **CombinedDataLoader**: when training on multiple labels, batches are
  drawn from each label's loader in proportion to dataset size (interleaved).
"""

import pickle
import sys
import contextlib

from dataclasses import dataclass
import random
from typing import Iterable
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter

from src.model.base import BaseTransformer
from src.run.util.config import ExperimentConfig, StageConfig, use_fused_adamw
from src.run.util.tools import get_batch, log_batch_counts, set_seeds
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.distributed import broadcast_object, is_main_process, barrier, get_rank
from src.run.util.preemption import is_preempted
from src.run.util.state import save_checkpoint, should_save
from src.run.eval import eval_loss


@dataclass
class BaselineConfig(StageConfig):
    """Configuration for the baseline (dense) training stage."""
    name: str = "baseline"
    label_prc: float = 1.0


@dataclass
class FilteringConfig(StageConfig):
    """Configuration for the data-filtering evaluation stage."""
    name: str = "filtering"
    retain_targets: list[list[str]] | None = None
    label_prc: float = 1.0


def do_train(
    stage: StageConfig,
    model: BaseTransformer,
    config: ExperimentConfig,
    train_labels: Iterable[str],
    state: dict | None = None,
) -> BaseTransformer:
    """
    Train a transformer model on specified train data labels.

    Args:
        stage: Stage configuration
        model: Model to train
        config: Run configuration
        train_labels: Data labels to train on
        state: State to resume from (overridden by auto-resume if checkpoint exists)
    
    Returns:
        Trained model
    """

    # unpack run config
    acc_steps = config.run.accumulation_steps
    loaders = config.run.loaders
    epochs = config.run.epochs
    logger = config.run.logger
    adam_betas = config.run.adam_betas
    warmup_prc = config.run.warmup_prc
    decay_prc = config.run.decay_prc
    data_labels = config.run.labels
    seed = config.run.seed
    is_ddp = config.run.is_ddp
    all_labels = config.run.labels

    #unpack stage config
    num_checkpoints = stage.num_checkpoints
    num_evals = stage.num_train_evals
    res_dir = stage.res_dir
    lr = stage.lr
    label_prc = stage.label_prc
    acc_mode = stage.acc_mode

    logger.info(f"---- Begin Train | Train Labels: {train_labels} ----")
    logger.debug(f"acc_mode: {acc_mode}, label_prc: {label_prc}")

    assert train_labels != ["all"]

    model.train()

    train_labels = sorted(train_labels)
    
    if state is None:
        state = dict()

    batches = []
    for label in all_labels:
        len_label = len(loaders[label]["train"])
        labeled_len = round(len_label * label_prc)
        unlabeled_len = len_label - labeled_len
        batches += [label] * unlabeled_len
        if label in train_labels:
            batches += [label] * labeled_len

    batches = broadcast_object(batches, src=0)

    if acc_mode == "uniform":
        counter = Counter(batches)
        temp = []
        for value, count in counter.items():
            temp += [value] * round(count / acc_steps)
        random.seed(seed)
        random.shuffle(temp)
        batches = []
        for value in temp:
            batches += [value] * acc_steps
        batches = broadcast_object(batches, src=0)

    else: #heterogeneous
        random.seed(get_rank())
        random.shuffle(batches) #different per rank
        if len(batches) % acc_steps != 0: #if not multiple of acc_steps, truncate
            batches = batches[:-(len(batches) % acc_steps)]

    log_batch_counts(batches, logger)
    set_seeds(seed)

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

    # setup losses
    losses = {"train": {}, "val": {}}
    for label in data_labels:
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
    
    # setup scheduler
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
    pbar = tqdm(total = num_total_steps, **get_tqdm_kwargs(logger, ncols=150))

    # training loop
    step = 0
    for epoch_idx in range(epochs):

        # reset loader for each epoch
        for label in data_labels:
            loaders[label]["train"].reset(epoch_idx)

        for batch_idx in range(len(batches)):

            loader_name = batches[batch_idx]
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
            label_idx = data_labels.index(batch_label)
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
                    for label in data_labels:
                        val_loss = eval_loss(
                            model, config, 
                            data_label=label, 
                            num_batches=50, 
                            shuffle_seed=step)
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

                # save checkpoint periodically or at final step
                if should_save(step, num_total_steps, checkpoint_freq):

                    ckpt_state = {
                        'opts': {'all': opt.state_dict()},
                        'step': step,
                        'total_steps': num_total_steps,
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

                    if is_main_process() and step == num_total_steps:
                        res_dir.mkdir(parents=True, exist_ok=True)
                        losses_path = res_dir / "losses.pkl"
                        losses_path.write_bytes(pickle.dumps(ckpt_state["losses"]))
            
        # sync after epoch
        barrier()

    # close progress bar
    pbar.close()

    return model

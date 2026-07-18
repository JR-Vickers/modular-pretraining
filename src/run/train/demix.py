"""
DEMix training loop.

Analogue of ``src/run/train/routed.py`` for the DEMix method. Data is routed to
experts by domain label (hard, document-level routing): every micro-batch of a
domain trains that domain's expert plus the shared trunk. There is no
ordered/unordered phase structure and no compute-equalisation — DEMix simply
trains each expert on its own domain and the trunk on everything.

Partial labeling (``label_prc`` < 1.0): only a ``label_prc`` fraction of each
domain's data has a known domain label and is routed to its own expert; the
rest is treated as unknown-domain and routed to the core (generic) expert
(``(label, ("core",), ("core",))``) — the data still comes from that domain's
loader, only the routing falls back to core. The shared trunk trains on all of
it. ``label_prc = 1.0`` (default) means fully labeled.

The mechanics mirror ``do_routed`` exactly so this stays compatible with the
rest of the pipeline:

* batches are ``(loader_name, fwd_experts, bck_experts)`` tuples;
* ``acc_mode`` ("heterogeneous" | "uniform") shapes the accumulation windows;
* one optimizer per label (one full-size expert each), **plus** one for the
  shared trunk ("SHARED" — the non-expert params: embeddings, final norm,
  unembed, and per-block attention + norms; there is no MLP for SHARED);
* after each accumulation window the SHARED optimizer is **always** stepped
  (the trunk trains on every domain), and each *expert* optimizer is stepped
  only if its parameters received a gradient (detected via ``p.grad is not
  None``) — i.e. only on windows that contained its domain. This is what makes
  heterogeneous windows (micro-batches routing to different experts) correct.
"""

from __future__ import annotations

import contextlib
import pickle
import random
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from src.model.config import ModelConfig, RoutedModelConfig, Transformer
from src.model.demix import DemixTransformer
from src.run.eval import eval_loss
from src.run.train.routed import RoutedConfig
from src.run.util.config import ExperimentConfig, use_fused_adamw
from src.run.util.distributed import (
    barrier,
    broadcast_object,
    get_raw_model,
    is_main_process,
)
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.preemption import is_preempted
from src.run.util.state import save_checkpoint, should_save
from src.run.util.tools import get_batch, get_exp_mask, log_batch_counts, set_seeds


# --------------------------------------------------------------------------- #
# DEMix stage config                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class DemixConfig(RoutedConfig):
    """DEMix routed stage (one full-size expert per domain)."""

    name: str = "routed"
    model: ModelConfig = field(
        default_factory=lambda: RoutedModelConfig(arch="demix")
    )
    label_prc: float = 1.0


# Pseudo-label for the shared (non-expert) trunk.
SHARED = "SHARED"


# --------------------------------------------------------------------------- #
# training                                                                    #
# --------------------------------------------------------------------------- #


def do_demix(
    stage: DemixConfig,
    model: Transformer,
    config: ExperimentConfig,
    state: dict | None = None,
) -> Transformer:
    """DEMix training: hard per-domain routing of data to experts.

    Each domain's data trains its own expert; the shared trunk trains on every
    domain. Returns the trained model.
    """

    # unpack config
    acc_steps = config.run.accumulation_steps
    loaders = config.run.loaders
    epochs = config.run.epochs
    logger = config.run.logger
    adam_betas = config.run.adam_betas
    data_labels = config.run.labels
    seed = config.run.seed
    warmup_prc = config.run.warmup_prc
    decay_prc = config.run.decay_prc
    is_ddp = config.run.is_ddp
    lr = stage.lr
    num_evals = stage.num_train_evals
    num_checkpoints = stage.num_checkpoints
    res_dir = stage.res_dir
    acc_mode = stage.acc_mode

    logger.info("---- Begin DEMix ----")
    logger.debug(f"acc_mode: {acc_mode}")

    raw_model = get_raw_model(model)
    assert isinstance(raw_model, DemixTransformer), "raw_model must be a DemixTransformer"

    assert lr > 0, "LR must be provided"
    assert 0 <= stage.label_prc <= 1, "0 <= label_prc <= 1"

    set_seeds(seed)

    # ------------------------------------------------------------------ #
    # build domain-routed batches.                                       #
    #   - labeled fraction (label_prc): data whose domain is known ->     #
    #     routed to its own expert,  (label, (label,), (label,)).         #
    #   - unlabeled fraction (1 - label_prc): domain unknown -> routed to  #
    #     the core (generic) expert, (label, ("core",), ("core",)). The    #
    #     real data label is kept as element 0 so the sequence is still    #
    #     drawn from that domain's loader; only the routing falls back to  #
    #     core. SHARED trains on both (it is always stepped).              #
    # core data routes to the core expert either way.                     #
    # ------------------------------------------------------------------ #
    n_labeled_total = 0
    n_unlabeled_total = 0
    batches: list[tuple[str, tuple, tuple]] = []
    for label in data_labels:
        total = len(loaders[label]["train"])
        n_labeled = round(total * stage.label_prc)
        n_unlabeled = total - n_labeled
        batches += [(label, (label,), (label,))] * n_labeled
        batches += [(label, ("core",), ("core",))] * n_unlabeled
        n_labeled_total += n_labeled
        n_unlabeled_total += n_unlabeled
    logger.info(
        f"label_prc={stage.label_prc}: labeled batches={n_labeled_total}, "
        f"unlabeled (routed to core)={n_unlabeled_total}"
    )

    # Shape accumulation windows (mirrors do_routed).
    random.seed(seed)
    if acc_mode == "uniform":
        # Each window contains a single label, repeated acc_steps times.
        count = Counter(batches)
        temp = []
        for batch, c in count.items():
            temp += [batch] * round(c / acc_steps)
        random.shuffle(temp)
        windowed = []
        for batch in temp:
            windowed += [batch] * acc_steps
        batches = windowed
    else:  # heterogeneous
        random.shuffle(batches)
        if len(batches) % acc_steps != 0:  # truncate to a multiple of acc_steps
            batches = batches[: -(len(batches) % acc_steps)]

    batches = broadcast_object(batches, src=0)
    batch_groups = [batches]

    log_batch_counts(batches, logger)
    count_by_label = {label: 0 for label in data_labels}
    for b in batches:
        count_by_label[b[0]] += 1
    logger.info(f"count_by_label: {count_by_label}")

    model.train()

    # validate batches
    for l_name, fwd_experts, bck_experts in batches:
        assert l_name in data_labels
        assert all(e in data_labels for e in fwd_experts)
        assert all(e in data_labels for e in bck_experts)

    if state is None:
        state = dict()

    # optimizer labels: one per expert, plus the shared trunk
    opt_labels = list(data_labels) + [SHARED]
    label_to_idx = {label: i for i, label in enumerate(data_labels)}

    # total optimizer steps
    num_total_steps = sum(len(bg) // acc_steps for bg in batch_groups) * epochs
    logger.info(f"num_total_steps: {num_total_steps}")
    eval_freq = max(1, num_total_steps // num_evals) if num_evals > 0 else -1
    checkpoint_freq = max(1, num_total_steps // num_checkpoints) if num_checkpoints > 0 else -1
    resume_step = state.get("step", -1)

    # setup optimizers (one per expert + SHARED; selective stepping = modularity)
    opts = {}
    for label in opt_labels:
        params = list(raw_model.get_params(label))
        opts[label] = torch.optim.AdamW(
            params, lr=lr, fused=use_fused_adamw(config.run.device), betas=adam_betas
        )

    # restore optimizers
    if "opts" in state:
        for label in opts.keys():
            if label in state["opts"]:
                opts[label].load_state_dict(state["opts"][label])
                opts[label].param_groups[0]["lr"] = lr
            else:
                logger.warning(f"Label {label} not found in state['opts']")
        del state["opts"]

    # setup losses
    losses = {"train": {}, "val": {}}
    for label in data_labels:
        losses["train"][label] = []
        losses["val"][label] = []

    # restore losses (each entry is a (step, loss) tuple)
    if "losses" in state:
        for label in losses["train"].keys():
            losses["train"][label] = [tuple(x) for x in state["losses"]["train"].get(label, [])]
        for label in losses["val"].keys():
            losses["val"][label] = [tuple(x) for x in state["losses"]["val"].get(label, [])]
        logger.info(f"Restored {sum(len(v) for v in losses['train'].values())} train loss entries")
        logger.info(f"Restored {sum(len(v) for v in losses['val'].values())} val loss entries")

    # setup scheduler — single warmup -> constant -> decay over all steps.
    end_factor = 1e-8 / lr
    group_len = num_total_steps
    warmup = max(1, round(warmup_prc * group_len))
    decay = max(1, round(decay_prc * group_len))
    constant = max(0, group_len - warmup - decay)
    logger.info(f"LR schedule: warmup={warmup}, constant={constant}, decay={decay}")

    def lr_lambda(current_step):
        if current_step < warmup:
            if current_step == 0:
                logger.info(f"LR Scheduler: warmup (step {current_step})")
            return end_factor + (1.0 - end_factor) * (current_step / warmup)
        elif current_step < warmup + constant:
            if current_step == warmup:
                logger.info(f"LR Scheduler: constant (step {current_step})")
            return 1.0
        else:
            if current_step == warmup + constant:
                logger.info(f"LR Scheduler: decay (step {current_step})")
            progress = min(1.0, (current_step - warmup - constant) / decay)
            return 1.0 - (1.0 - end_factor) * progress

    # Schedule is driven off the SHARED optimizer (it is stepped every window).
    scheduler = LambdaLR(opts[SHARED], lr_lambda)
    cur_lr = scheduler.get_last_lr()[0]
    for label in opts.keys():
        opts[label].param_groups[0]["lr"] = cur_lr

    # restore scheduler position from step count
    if resume_step > 0:
        scheduler.last_epoch = resume_step
        cur_lr = scheduler.get_lr()[0]
        scheduler._last_lr = [cur_lr]
        for label in opts.keys():
            opts[label].param_groups[0]["lr"] = cur_lr
        logger.info(f"Restored scheduler to step {resume_step}, LR: {cur_lr:.4e}")

    # memory diagnostics (before first training step)
    if is_main_process() and config.run.device.type == "cuda":
        mem_summary = torch.cuda.memory_summary()
        logger.info("CUDA memory before training loop:")
        for line in mem_summary.split("\n"):
            logger.info(line)

    # training loop
    step = 0
    for batches in batch_groups:

        group_epoch_steps = len(batches) // acc_steps
        group_total_steps = group_epoch_steps * epochs
        pbar = tqdm(total=group_total_steps, **get_tqdm_kwargs(logger, ncols=150))

        logger.info(
            f"Batch group start, LR: {cur_lr:.4e}, batches: {len(batches)}, "
            f"epochs: {epochs}, group_total_steps: {group_total_steps}"
        )

        for epoch_idx in range(epochs):

            for label in data_labels:
                loaders[label]["train"].reset(epoch_idx)

            for batch_idx in range(len(batches)):

                access_idx = batch_idx
                if acc_mode == "uniform":
                    access_idx = batch_idx // acc_steps * acc_steps

                batch = batches[access_idx]
                loader_name, experts_forward, experts_backward = batch
                loader = loaders[loader_name]["train"]
                x, y, batch_label = get_batch(loader)

                if step < resume_step:
                    step += 1
                    continue

                fwd_mask = get_exp_mask(data_labels, experts_forward, device=x.device)
                bck_mask = get_exp_mask(data_labels, experts_backward, device=x.device)

                acc_idx = batch_idx % acc_steps
                is_last_acc = acc_idx == acc_steps - 1
                no_sync_ctx = model.no_sync() if (is_ddp and not is_last_acc) else contextlib.nullcontext()

                with no_sync_ctx:

                    _, loss = model.forward(
                        tokens=x,
                        targets=y,
                        fwd_mask=fwd_mask,
                        bck_mask=bck_mask,
                    )

                    scaled_loss = loss / acc_steps
                    scaled_loss.backward()

                # train loss logging
                cur_loss = loss.item()
                losses["train"][batch_label].append((step, cur_loss))

                # update progress bar
                num_subsets = len(data_labels)
                exp_for_str = ",".join([str(label_to_idx[e]) for e in experts_forward])
                exp_for_str = exp_for_str.ljust(num_subsets * 2 - 1)
                exp_bck_str = ",".join([str(label_to_idx[e]) for e in experts_backward])
                exp_bck_str = exp_bck_str.ljust((num_subsets + 1) * 2 - 1)
                label_idx = data_labels.index(batch_label)
                desc_str = (
                    f"AC {acc_idx + 1}/{acc_steps} | LR: {cur_lr:.2e} | L: {cur_loss:.2f} | "
                    f"LB: {label_idx} | EF: {exp_for_str} | EB: {exp_bck_str}"
                )
                pbar.set_description(desc_str)
                pbar.refresh()

                if is_last_acc:

                    step += 1
                    pbar.update()

                    # experts that received gradient this window (domain-routed),
                    # plus SHARED which is *always* stepped (the shared trunk
                    # trains on every domain in DEMix).
                    exp_to_step = [
                        label
                        for label in data_labels
                        if any(p.grad is not None for p in raw_model.get_params(label))
                    ]
                    opts_to_step = exp_to_step + [SHARED]

                    # clip the union
                    params_to_clip = [p for e in opts_to_step for p in raw_model.get_params(e)]
                    if params_to_clip:
                        torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)

                    # step the domain experts that got gradient + always SHARED
                    for label in opts_to_step:
                        opts[label].step()

                    for opt in opts.values():
                        opt.zero_grad(set_to_none=True)

                    # update learning rate
                    scheduler.step()
                    cur_lr = scheduler.get_last_lr()[0]
                    for label in opts.keys():
                        opts[label].param_groups[0]["lr"] = cur_lr

                    # measure validation loss (each domain on its own expert)
                    if eval_freq > 0 and step % eval_freq == 0:
                        model.eval()
                        for label in data_labels:
                            val_loss = eval_loss(
                                model,
                                config,
                                data_label=label,
                                num_batches=50,
                                expert_labels=(label,),
                                shuffle_seed=step,
                            )
                            losses["val"][label].append((step, val_loss))
                        model.train()

                    # memory diagnostics (after first step)
                    if (
                        step == resume_step + 2
                        and is_main_process()
                        and config.run.device.type == "cuda"
                    ):
                        mem_summary = torch.cuda.memory_summary()
                        logger.info("CUDA memory after first training step:")
                        for line in mem_summary.split("\n"):
                            logger.info(line)

                    # logger printout
                    if (step == 1) or (step % 1000 == 0) or (step == num_total_steps):
                        train_loss_str = ""
                        for label in losses["train"].keys():
                            label_str = label.upper()
                            loss_slice = [v for _, v in losses["train"][label][-100:]]
                            avg_train_loss = np.mean(loss_slice) if len(loss_slice) > 0 else float("nan")
                            train_loss_str += f"{label_str}: {avg_train_loss:.2f} "
                        logger.info(f"Step: {step}, LR: {cur_lr:.4e}")
                        logger.info(f"Train Loss: {train_loss_str}")

                        if eval_freq > 0:
                            val_loss_str = ""
                            for label in losses["val"].keys():
                                label_str = label.upper()
                                val_loss = losses["val"][label][-1][1] if len(losses["val"][label]) > 0 else float("nan")
                                val_loss_str += f"{label_str}: {val_loss:.2f} "
                            logger.info(f"Val Loss: {val_loss_str}")

                    # save checkpoint periodically or at final step
                    if should_save(step, num_total_steps, checkpoint_freq):

                        ckpt_state = {
                            "opts": {label: opt.state_dict() for label, opt in opts.items()},
                            "step": step,
                            "total_steps": num_total_steps,
                            "losses": {
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

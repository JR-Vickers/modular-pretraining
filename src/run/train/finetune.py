"""
Adversarial fine-tuning for elicitation testing.

This is NOT a training stage that produces a model for downstream use.
Instead, it measures how easily a trained model can *re-learn* forgotten or
compartmentalized knowledge, which is the key metric for evaluating the
effectiveness of gradient routing and unlearning methods.

The procedure:
1. Copy the model being evaluated.
2. Fine-tune the copy on the target label for the full target-label dataset.
3. Log per-step validation loss to a pickle file for offline analysis.
4. At ``ft_eval_prc`` of the way through, snapshot the val loss and write
   it to stats.jsonl as the canonical elicited loss.
"""

from __future__ import annotations

import contextlib
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from sympy import N
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from src.model.config import Transformer
from src.run.eval import eval_loss
from src.run.util.config import ExperimentConfig, StageConfig, use_fused_adamw
from src.run.util.tools import get_batch, get_exp_mask, log_line
from src.run.util.tools import json_safe, labels_to_str
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.dataloader import InterleavedDataLoader
from src.run.util.distributed import barrier, get_raw_model, get_rank, get_world_size, is_main_process, broadcast_object


def do_finetune(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    data_labels: Iterable[str],
    expert_labels: Optional[Iterable[str]] = None,
    log_args: Optional[dict[str, Any]] = None,
    num_seq: int = 512,
    num_rounds: int = 200,
    patience: int = 10,
) -> tuple[Transformer, dict]:
    """
    Finetune a model on specified data labels with optional expert routing.
    
    Returns:
        Finetuned model
    """

    loaders = config.run.loaders
    logger = config.run.logger
    labels = config.run.labels
    log_fp = config.run.res_dir / "stats.jsonl"
    lr = stage.lr / 4
    mbs_size = config.run.micro_batch_size
    is_ddp = config.run.is_ddp
    data_labels = list(data_labels)

    logger.info(f"---- Begin FT | Experts: {expert_labels} | Data: {data_labels} ----")
    logger.debug(f"num_seq: {num_seq}, num_rounds: {num_rounds}, patience: {patience}")

    assert all(label in labels for label in data_labels), f"all data labels must be in {labels}"

    raw_model = get_raw_model(model)
    model_type = type(raw_model).__name__
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, fused=use_fused_adamw(config.run.device)
    )

    loaders = [config.run.loaders[label]["train"] for label in data_labels]
    loader = InterleavedDataLoader(loaders, weighted=False)
    loader.reset(0)

    patience_count = 0
    best_idx = 0
    losses = {lab: [] for lab in data_labels}

    # Each rank builds its own (num_seq, T-1) CUDA tensor on its local
    # device. Shapes/dtypes are identical across ranks (loaders share B and
    # T), but values differ since the loader RNG is per-rank. We then
    # broadcast rank 0's tensor in-place via NCCL so all ranks end up with
    # the same data on their own cuda:N — no pickle / host roundtrip and
    # no cuda:0 device leak (the bug that previously crashed ranks > 0).
    chunks_x, chunks_y = [], []
    while sum(c.shape[0] for c in chunks_x) < num_seq:
        x, y, _ = get_batch(loader)
        chunks_x.append(x)
        chunks_y.append(y)
    data_x = torch.cat(chunks_x, dim=0)[:num_seq].contiguous()
    data_y = torch.cat(chunks_y, dim=0)[:num_seq].contiguous()

    if is_ddp:
        dist.broadcast(data_x, src=0)
        dist.broadcast(data_y, src=0)

    world_size = get_world_size()
    rank = get_rank()
    if num_seq % world_size != 0:
        num_seq = (num_seq // world_size) * world_size
        data_x = data_x[:num_seq]
        data_y = data_y[:num_seq]
    per_gpu = num_seq // world_size
    local_x = data_x[rank * per_gpu : (rank + 1) * per_gpu]
    local_y = data_y[rank * per_gpu : (rank + 1) * per_gpu]
    num_steps = (per_gpu + mbs_size - 1) // mbs_size

    logger.debug(f"local_x: {local_x.shape}, local_y: {local_y.shape}, num_steps: {num_steps}, per_gpu: {per_gpu}, mbs_size: {mbs_size}")

    pbar = tqdm(range(num_rounds), **get_tqdm_kwargs(logger, desc=f"FT", ncols=150))
    for round_idx in pbar:
       
        for i in range(0, per_gpu, mbs_size):

            x = local_x[i:i+mbs_size]
            y = local_y[i:i+mbs_size]

            is_last_acc = i + mbs_size >= per_gpu
            no_sync_ctx = model.no_sync() if (is_ddp and not is_last_acc) else contextlib.nullcontext()

            with no_sync_ctx:
                if model_type in ("MoETransformer", "LoRATransformer", "DemixTransformer"):
                    fwd_mask = get_exp_mask(labels, expert_labels, device=x.device)
                    bck_mask = get_exp_mask(labels, expert_labels, device=x.device)
                    loss = model(x, targets=y, fwd_mask=fwd_mask, bck_mask=bck_mask)[1]
                else:
                    loss = model(x, targets=y)[1]

                loss = loss / num_steps
                loss.backward()

            pbar.set_description(f"Data {i}-{min(i+mbs_size, per_gpu)}/{per_gpu} (x{world_size}) | L {loss:.4f}")

        opt.step()
        opt.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        with torch.inference_mode():

            for label in data_labels:
                val_loss = eval_loss(
                    model,
                    config,
                    data_label=label,
                    expert_labels=expert_labels,
                    num_batches=100
                )
                losses[label].append(val_loss)

            last_losses = [losses[lab][-1] for lab in data_labels]
            val_loss = sum(last_losses) / len(last_losses)

            old_losses = [losses[lab][best_idx] for lab in data_labels]
            old_loss = sum(old_losses) / len(old_losses)

            if val_loss <= old_loss:
                best_idx = round_idx
                logger.info(f"New best validation loss: {val_loss:.4f} @ step {round_idx}")
                patience_count = 0
            else:
                patience_count += 1

        if patience_count > patience:
            logger.info(f"Stop FT @ {round_idx+1}: no val improvement for {patience} steps")
            break

    if is_main_process():
        for label in data_labels:
            loss = losses[label][best_idx]
            entry = {
                "stage": json_safe(stage),
                "function": "do_finetune",
                "data_label": label,
                "expert_labels": expert_labels,
                "loss": loss,
                "ft_step": best_idx,
                "ft_type": "single_batch",
            }
            if log_args:
                entry.update(log_args)
            log_line(entry, log_fp)
        logger.info(
            f"Wrote stats.jsonl entries at FT step {best_idx} "
        )

    barrier()

    return model


def do_finetune_old(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    data_labels: Iterable[str],
    save_dir: Path,
    expert_labels: Optional[Iterable[str]] = None,
    log_args: Optional[dict[str, Any]] = None,
) -> Transformer:
    """Fine-tune a model on specified data labels over the full dataset."""

    loaders = config.run.loaders
    logger = config.run.logger
    labels = config.run.labels
    log_fp = config.run.res_dir / "stats.jsonl"
    acc_steps = config.run.accumulation_steps
    stats_prc = stage.elicit_eval_prc
    num_evals = max(1, stage.elicit_num_evals)
    lr = stage.lr / 2

    data_labels = list(data_labels)
    logger.info(f"---- Begin FT | Experts: {expert_labels} | Data: {data_labels} ----")
    assert all(x in labels for x in data_labels), f"all data_labels must be in {labels}"

    raw_model = get_raw_model(model)
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, fused=use_fused_adamw(config.run.device)
    )

    loaders = [config.run.loaders[label]["train"] for label in data_labels]
    loader = InterleavedDataLoader(loaders, weighted=False)
    loader.reset()

    num_batches = len(loader)
    num_steps = num_batches // acc_steps
    eval_every = max(1, num_steps // num_evals)
    stats_step = max(1, int(num_steps * stats_prc))
    losses: dict[str, list[tuple[int, float]]] = {lab: [] for lab in data_labels}

    pbar = tqdm(range(num_batches), **get_tqdm_kwargs(logger, desc="FT", ncols=150))
    for batch_idx in pbar:

        if batch_idx % acc_steps == 0:
            batch_loss = 0.0

        x, y, _ = get_batch(loader)

        if raw_model.config.arch in ("moe", "lora", "demix"):
            fwd_mask = get_exp_mask(labels, expert_labels, device=x.device)
            bck_mask = get_exp_mask(labels, expert_labels, device=x.device)
            loss = model(x, targets=y, fwd_mask=fwd_mask, bck_mask=bck_mask)[1]
        else:
            loss = model(x, targets=y)[1]

        loss = loss / acc_steps
        batch_loss += loss.item()
        loss.backward()

        if batch_idx % acc_steps == acc_steps - 1:
            step = (batch_idx + 1) // acc_steps
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % eval_every == 0 or step == num_steps or step == stats_step:

                model.eval()
                torch.cuda.empty_cache()
                with torch.inference_mode():

                    for label in data_labels:
                        val_loss = eval_loss(
                            model=model,
                            config=config,
                            data_label=label,
                            expert_labels=expert_labels,
                            num_batches=10,
                        )
                        losses[label].append((step, val_loss))

                    val_str = " | ".join(
                        f"{lab}: {losses[lab][-1][1]:.4f}" for lab in data_labels
                    )
                    logger.info(
                        f"FT step {step}/{num_steps} | train: {batch_loss:.4f} | {val_str}"
                    )
                model.train()

            # Write stats.jsonl entry at the snapshot step
            if step == stats_step:
                if is_main_process():
                    for label in data_labels:
                        snap_loss = losses[label][-1][1]
                        entry = {
                            "stage": json_safe(stage),
                            "function": "do_finetune",
                            "data_label": label,
                            "expert_labels": expert_labels,
                            "loss": snap_loss,
                            "ft_step": step,
                            "ft_num_steps": num_steps,
                            "ft_type": "multi_batch",
                        }
                        if log_args:
                            entry.update(log_args)
                        log_line(entry, log_fp)
                    logger.info(
                        f"Wrote stats.jsonl entries at FT step {step} "
                        f"(stats_prc={stats_prc})"
                    )

    # Write per-label loss trajectory to pkl
    if is_main_process():
        label_str = labels_to_str(data_labels)
        pkl_path = save_dir / label_str / "losses.pkl"
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        pkl_data = {lab: np.array(vals) for lab, vals in losses.items()}
        pkl_path.write_bytes(pickle.dumps(pkl_data))
        logger.info(f"Wrote elicitation losses to {pkl_path}")

    barrier()
    return model

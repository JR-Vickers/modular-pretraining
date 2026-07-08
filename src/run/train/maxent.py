"""
Maximum-entropy (MaxEnt) unlearning.

Post-hoc method that pushes the model's output distribution on forget-target
data towards uniform, while preserving performance on retained data via a
weighted cross-entropy loss.

The total loss is::

    L = KL(P_forget || Uniform) + alpha * CE(P_retain, Y_retain)

where alpha (``me_alpha_retain``) controls the strength of the retain
regularisation.  Higher alpha preserves more retain performance at the cost
of slower unlearning.

Memory optimization: ``kl_uniform_chunked`` computes KL divergence to the
uniform distribution in chunks over the vocabulary dimension, avoiding
materialising a (B, T, V) softmax tensor in memory.
"""

from dataclasses import dataclass
from math import log
from typing import Iterable
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F

from src.model.base import BaseTransformer
from src.run.util.config import ExperimentConfig, StageConfig
from src.run.util.tools import get_batch
from src.run.util.dataloader import InterleavedDataLoader
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.distributed import barrier, get_raw_model, reduce_tensor


@dataclass
class MaxentConfig(StageConfig):
    """Configuration for maximum-entropy unlearning."""
    name: str = "maxent"
    lr: float = 1e-4
    alpha_retain: float = 15.0
    steps: int = 2000
    early_stop_thresh: float | None = 2.0

def kl_uniform_chunked(logits: torch.Tensor, logV: float, chunk: int = 2048) -> torch.Tensor:
    """Compute KL divergence to uniform in chunks to save memory.

    Args:
        logits: (B, T, V) fp16 tensor.
        logV:   log(|V|)
        chunk:  number of vocab entries processed at once.
    Returns:
        kl: (B, T) tensor, per-token KL(P||U)
    """

    B, T, Vocab = logits.shape
    device = logits.device

    # max logits for numerical stability (B,T,1)
    max_l = logits.max(dim=-1, keepdim=True).values

    exp_sum = torch.zeros((B, T), dtype=torch.float32, device=device)
    exp_logits_times_logits_sum = torch.zeros_like(exp_sum)

    for start in range(0, Vocab, chunk):
        end = min(start + chunk, Vocab)
        l_chunk = logits[..., start:end]
        exp_l_chunk = (l_chunk - max_l).exp()  # (B,T,C)
        exp_sum += exp_l_chunk.sum(dim=-1)
        exp_logits_times_logits_sum += (exp_l_chunk * l_chunk).sum(dim=-1)

    logZ = exp_sum.log() + max_l.squeeze(-1)  # (B,T)
    kl = logV + exp_logits_times_logits_sum / exp_sum - logZ  # (B,T)
    return kl


def do_maxent(
    stage: StageConfig,
    model: BaseTransformer,
    config: ExperimentConfig,
    data_labels: Iterable[str],
) -> BaseTransformer:
    """
    Maximum entropy unlearning: push forget set towards uniform distribution.
    
    Args:
        model: Model to unlearn from
        config: Run configuration
        data_labels: Data labels to forget
        alpha_retain: Weight for retain loss
        lr: Learning rate
        steps: Number of optimization steps
        early_stop_thresh: Early stopping threshold for KL divergence
    
    Returns:
        Unlearned model
    """

    # unpack run config
    loaders = config.run.loaders
    logger = config.run.logger
    alpha_retain = stage.alpha_retain
    lr = stage.lr
    steps = stage.steps
    early_stop_thresh = stage.early_stop_thresh

    logger.info(f"---- Begin MaxEnt | Data Labels: {data_labels} ----")
    logger.debug(f"alpha_retain: {alpha_retain}, lr: {lr}, steps: {steps}, early_stop_thresh: {early_stop_thresh}")

    assert all(label in loaders.keys() for label in data_labels), f"data labels must be in {loaders.keys()}"
    assert "core" not in data_labels, "core cannot be in unlearning data labels"

    # Get raw model for accessing model config
    raw_model = get_raw_model(model)
    
    model.train()

    retain_labels = sorted( set(config.data.aux.labels) - set(data_labels) | set(config.data.core.labels) )
    logger.debug(f"retain_labels: {retain_labels}")

    forget_subloaders = [loaders[label]["train"] for label in data_labels]
    retain_subloaders = [loaders[label]["train"] for label in retain_labels]
    
    forget_loader = InterleavedDataLoader(forget_subloaders, weighted=False)
    retain_loader = InterleavedDataLoader(retain_subloaders, weighted=False)

    forget_loader.reset()
    retain_loader.reset()

    V = raw_model.config.vocab_size
    logV_const = log(V)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

    tot_me_loss = 0.0
    tot_unif_loss = 0.0
    tot_retain_loss = 0.0

    pbar = tqdm(range(steps), **get_tqdm_kwargs(logger, desc=f"MAXENT", ncols=100))

    for step in pbar:
        # Halve each batch and concatenate so forget + retain fit in one
        # forward pass — the model sees both in a single call, then we
        # split the logits to compute separate losses.
        frg_x, frg_y, _ = get_batch(forget_loader)
        frg_x = frg_x[:len(frg_x)//2]
        frg_y = frg_y[:len(frg_y)//2]
        # 3:1 core-to-aux retain ratio — core is the bulk of retained knowledge
        # so it gets more representation in the retain loss.
        if random.random() < 0.75:
            loader = loaders["core"]["train"]
        else:
            if len(retain_labels) == 1:
                loader = loaders["core"]["train"]
            else:
                label = random.choice([x for x in retain_labels if x != "core"])
                loader = loaders[label]["train"]

        ret_x, ret_y, _ = get_batch(loader)
        ret_x = ret_x[:len(ret_x)//2]
        ret_y = ret_y[:len(ret_y)//2]

        combo_x = torch.cat([frg_x, ret_x], dim=0)

        combo_logits = model(combo_x)[0]
        frg_logits = combo_logits[:len(frg_x)]
        ret_logits = combo_logits[len(frg_x):]

        kl_token = kl_uniform_chunked(frg_logits, logV_const)  # (B,T)
        unif_loss = kl_token.mean()  #  L_uniform(θ)

        ret_loss = F.cross_entropy(
            ret_logits.view(-1, ret_logits.size(-1)),
            ret_y.reshape(-1),
            ignore_index=-1000,
            reduction="mean",
        )

        loss = unif_loss + alpha_retain * ret_loss

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        tot_me_loss += loss.item()
        tot_unif_loss += unif_loss.item()
        tot_retain_loss += ret_loss.item()

        pbar.set_description(f"MAXENT U:{unif_loss.item():.3f} Retain:{ret_loss.item():.3f}")

        # ---------------- optional early stopping on uniformity ---------------
        steps_done = step + 1
        if early_stop_thresh is not None:
            # Synchronize early stop decision across all ranks to prevent deadlock
            unif_loss_synced = reduce_tensor(
                torch.tensor(unif_loss.item(), device=config.run.device)
            ).item()
            if unif_loss_synced <= early_stop_thresh:
                logger.info(f"Early stop MAXENT at step {step+1}: KL→U ≤ {early_stop_thresh}")
                break

    logger.info(
        f"Finished MaxEnt — Avg KL→U: {tot_unif_loss/steps_done:.3f}, "
        f"Avg Retain CE: {tot_retain_loss/steps_done:.3f}"
    )
    
    barrier()

    return model

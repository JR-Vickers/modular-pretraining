from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch

from src.model.base import BaseTransformer
from src.run.eval import eval_loss
from src.run.util.config import ExperimentConfig, StageConfig
from src.run.util.tools import get_batch
from src.run.util.dataloader import InterleavedDataLoader
from tqdm import tqdm
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.distributed import barrier, reduce_tensor


@dataclass
class AscentConfig(StageConfig):
    """Configuration for gradient-ascent unlearning."""
    name: str = "ascent"
    alpha_retain: float = 10.0
    steps: int = 100
    early_stop_thresh: float | None = 6.0


# --------------------------------------------------------------------------- #
# Gradient Ascent                                                             #
# --------------------------------------------------------------------------- #

def do_gradient_ascent(
    stage: StageConfig,
    model: BaseTransformer,
    config: ExperimentConfig,
    data_labels: Iterable[str],
) -> BaseTransformer:
    """
    Gradient ascent unlearning: maximize loss on forget set while minimizing on retain set.
    
    Supports both single-GPU and multi-GPU (DDP) training.
    
    Args:
        model: Model to unlearn from (may be DDP-wrapped)
        config: Run configuration
        data_labels: Data labels to forget
        alpha_retain: Weight for retain loss
        lr: Learning rate
        steps: Number of optimization steps
        early_stop_thresh: Early stopping threshold for forget validation loss
    
    Returns:
        Unlearned model (in same wrapped/unwrapped state as input)
    """

    # unpack run config
    loaders = config.run.loaders
    logger = config.run.logger
    alpha_retain = stage.alpha_retain
    lr = stage.lr
    steps = stage.steps
    early_stop_thresh = stage.early_stop_thresh

    logger.info(f"---- Begin Gradient Ascent | Data Labels: {data_labels} ----")

    assert all(label in loaders.keys() for label in data_labels), f"data labels must be in {loaders.keys()}"
    assert "core" not in data_labels, "core cannot be in unlearning data labels"

    model.train()

    retain_labels = sorted(set(config.data.aux.labels) - set(data_labels)) + ["core"]

    forget_subloaders = [loaders[label]["train"] for label in data_labels]
    retain_subloaders = [loaders[label]["train"] for label in retain_labels]

    retain_loader = InterleavedDataLoader(retain_subloaders, weighted=False)
    forget_loader = InterleavedDataLoader(forget_subloaders, weighted=False)

    retain_loader.reset()
    forget_loader.reset()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

    total_ga_loss = 0.0
    total_forget_loss = 0.0
    total_retain_loss = 0.0
    forget_val_loss = None

    pbar = tqdm(range(steps), **get_tqdm_kwargs(logger, desc=f"ASCENT", ncols=100))

    for step in pbar:

        ret_x, ret_y, _ = get_batch(retain_loader)
        ret_x = ret_x[:len(ret_x)//2]
        ret_y = ret_y[:len(ret_y)//2]

        frg_x, frg_y, _ = get_batch(forget_loader)
        frg_x = frg_x[:len(frg_x)//2]
        frg_y = frg_y[:len(frg_y)//2]

        # Combine batches for single forward pass to reduce memory usage
        combo_x = torch.cat([frg_x, ret_x], dim=0)
        combo_logits = model(combo_x)[0]

        # Split logits to compute separate losses
        frg_logits = combo_logits[:len(frg_x)]
        ret_logits = combo_logits[len(frg_x):]

        # Calculate forget loss (we want to maximize this)
        frg_loss = torch.nn.functional.cross_entropy(
            frg_logits.view(-1, frg_logits.size(-1)),
            frg_y.reshape(-1),
            ignore_index=-1000,
            reduction="mean",
        )

        # Calculate retain loss (we want to minimize this)
        ret_loss = torch.nn.functional.cross_entropy(
            ret_logits.view(-1, ret_logits.size(-1)),
            ret_y.reshape(-1),
            ignore_index=-1000,
            reduction="mean",
        )

        # Gradient Ascent Objective: Minimize (alpha * retain_loss - forget_loss)
        # This is equivalent to maximizing (forget_loss - alpha * retain_loss)
        combined_loss = alpha_retain * ret_loss - frg_loss

        total_ga_loss += combined_loss.item()
        total_forget_loss += frg_loss.item()
        total_retain_loss += ret_loss.item()

        pbar.set_description(
            f"ASCENT Forget: {frg_loss.item():.4f}, Retain: {ret_loss.item():.4f}"
        )

        combined_loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        if early_stop_thresh is not None:

            with torch.inference_mode():

                # Evaluate forget validation loss
                arr = []
                for label in data_labels:
                    temp = eval_loss(
                        model=model,
                        config=config,
                        data_label=label,
                        num_batches=50,
                    )
                    arr.append(temp)

                forget_val_loss = sum(arr) / len(arr)
                
                # Synchronize early stop decision across all ranks to prevent deadlock
                forget_val_loss = reduce_tensor(
                    torch.tensor(forget_val_loss, device=config.run.device)
                ).item()

                if forget_val_loss >= early_stop_thresh:  # Early stopping
                    logger.info(
                        f"Early stopping ASCENT at step {step+1} due to "
                        f"forget_val_loss >= {early_stop_thresh} (Loss: {forget_val_loss:.4f})"
                    )
                    break

    actual_completed_steps = step + 1

    avg_ga_loss = total_ga_loss / actual_completed_steps
    avg_forget_loss = total_forget_loss / actual_completed_steps
    avg_retain_loss = total_retain_loss / actual_completed_steps

    logger.info(
        f"Finished ASCENT | data labels: {data_labels} | Avg Combined: {avg_ga_loss:.4f}, "
        f"Forget Train: {avg_forget_loss:.4f}, Retain Train: {avg_retain_loss:.4f}"
    )
    
    barrier()

    return model
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import contextlib
import torch

from src.model.base import BaseTransformer
from src.run.util.config import ExperimentConfig, StageConfig
from src.run.util.tools import get_batch
from src.run.util.dataloader import InterleavedDataLoader
from tqdm import tqdm
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.distributed import barrier, get_raw_model, is_distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class RmuConfig(StageConfig):
    """Configuration for Representation Misdirection Unlearning."""
    name: str = "rmu"
    c: float = 100.0
    alpha_retain: float = 200.0
    steps: int = 500


# --------------------------------------------------------------------------- #
# RMU                                                                         #
# --------------------------------------------------------------------------- #


def do_rmu(
    stage: StageConfig,
    model: BaseTransformer,
    config: ExperimentConfig,
    data_labels: Iterable[str],
    frozen_model: BaseTransformer,
) -> BaseTransformer:
    """
    Representation Misdirection for Unlearning (RMU).
    
    Args:
        model: Model to unlearn from
        config: Run configuration
        data_labels: Data labels to forget
        frozen_model: Frozen copy of original model
        act_layer_inds: Layer indices to apply RMU
        c: Scaling coefficient for forget loss
        alpha_retain: Weight for retain loss
        lr: Learning rate
        steps: Number of optimization steps
    
    Returns:
        Unlearned model
    """

    # unpack run config
    loaders = config.run.loaders
    device = config.run.device
    logger = config.run.logger
    c = stage.c
    alpha_retain = stage.alpha_retain
    lr = stage.lr
    steps = stage.steps
    act_layer_inds: List[int] = list(range(config.model.num_layers - 2))

    logger.info(f"---- Begin RMU | Data Labels: {data_labels} ----")

    assert all(label in loaders.keys() for label in data_labels), f"data labels must be in {loaders.keys()}"
    assert "core" not in data_labels, "core cannot be in unlearning data labels"

    # Get raw models for accessing config and blocks
    raw_model = get_raw_model(model)

    model.train()
    frozen_model.eval()  # Frozen model should be in eval mode

    act_layer_inds = sorted(set(act_layer_inds))
    assert len(act_layer_inds) > 0, "no indices provided"
    assert max(act_layer_inds) < raw_model.config.num_layers, "max ind greater than n_layer"

    # Collect parameters from the specified layers for optimization.
    params = []
    for layer_idx in act_layer_inds:
        params.append(raw_model.blocks[layer_idx].mlp.c_proj.weight)

    retain_labels = sorted(set(config.data.aux.labels) - set(data_labels)) + ["core"]

    forget_subloaders = [loaders[label]["train"] for label in data_labels]
    retain_subloaders = [loaders[label]["train"] for label in retain_labels]

    forget_loader = InterleavedDataLoader(forget_subloaders, weighted=False)
    retain_loader = InterleavedDataLoader(retain_subloaders, weighted=False)

    forget_loader.reset()
    retain_loader.reset()

    opt = torch.optim.AdamW(params, lr=lr, fused=True)

    # one control vector per label to unlearn (synced across GPUs)
    control_vecs = {}
    for label in data_labels:
        random_vec = torch.rand(raw_model.config.embed_dim, device=device, dtype=torch.bfloat16)
        # Broadcast from rank 0 to ensure all GPUs use the same control vector
        if is_distributed():
            dist.broadcast(random_vec, src=0)
        control_vecs[label] = (random_vec / torch.norm(random_vec)) * c

    pbar = tqdm(range(steps), **get_tqdm_kwargs(logger, desc=f"RMU", ncols=100))
    
    # Check if model is DDP-wrapped
    is_ddp = is_distributed() and isinstance(model, DDP)
    
    for _ in pbar:

        frg_x, _, cur_label = get_batch(forget_loader)
        frg_x = frg_x[:len(frg_x)//2]

        ret_x, _, _ = get_batch(retain_loader)
        ret_x = ret_x[:len(ret_x)//2]

        with torch.no_grad():
            ret_act_frozen = frozen_model(ret_x, targets=None, stop_at_layer=max(act_layer_inds))[0]

        # Combine batches for single forward pass to reduce memory usage
        combo_x = torch.cat([frg_x, ret_x], dim=0)
        
        # Use no_sync context to prevent DDP from complaining about unused parameters
        # when using stop_at_layer (which doesn't use all layers)
        maybe_no_sync = model.no_sync() if is_ddp else contextlib.nullcontext()
        
        with maybe_no_sync:
            combo_act = model(combo_x, targets=None, stop_at_layer=max(act_layer_inds))[0]

            # Split activations
            frg_act = combo_act[:len(frg_x)]
            ret_act = combo_act[len(frg_x):]
            
            cur_vec = control_vecs[cur_label]
            frg_loss = torch.nn.functional.mse_loss(
                frg_act, cur_vec.view(1, 1, -1).expand_as(frg_act)
            )

            ret_loss = torch.nn.functional.mse_loss(ret_act, ret_act_frozen)
            ret_loss = alpha_retain * ret_loss
            
            loss = frg_loss + ret_loss

            loss.backward()
            
        # Manual gradient synchronization if using DDP (only for params we're optimizing)
        if is_ddp:
            for param in params:
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)
        
        opt.step()
        opt.zero_grad(set_to_none=True)

        pbar.set_description(f"RMU Forget: {frg_loss.item():.4f}, Retain: {ret_loss.item():.4f}")
    
    barrier()
    
    return model

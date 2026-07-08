"""
Minimal distributed training utilities for torchrun.

This module provides lightweight utilities for multi-GPU training with PyTorch DDP.
Designed to work with torchrun launcher only - no custom wrappers or abstractions.

Usage:
    # Single GPU
    python src/run/main.py --epochs 10
    
    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 src/run/main.py --epochs 10
"""

from __future__ import annotations

import atexit
import glob
import logging
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# NCCL 2.27.5 shm workaround; set at import time so tests can assert it.
os.environ.setdefault("NCCL_SHM_DISABLE", "1")

_logger = logging.getLogger(__name__)
_atexit_registered = False


# ============================================================================
# NCCL Resource Cleanup
# ============================================================================

def _cleanup_nccl_shm() -> None:
    """Remove stale NCCL shared memory segments from /dev/shm.

    Previous torchrun runs that crashed or were killed can leave
    ``/dev/shm/nccl-*`` files behind.  When the next run starts, NCCL
    tries to attach to these corrupted segments and fails with
    ``ncclSystemError``.  Cleaning them before ``init_process_group``
    prevents this.

    Safe to call from every rank concurrently — concurrent removes on
    the same file are harmless (one succeeds, the rest get ENOENT which
    is caught).
    """
    stale = glob.glob("/dev/shm/nccl-*")
    removed = 0
    for path in stale:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    if removed:
        _logger.info(f"Cleaned {removed} stale NCCL shm segment(s)")


def _atexit_cleanup() -> None:
    """Best-effort cleanup registered via atexit.

    Ensures ``destroy_process_group`` and NCCL shm cleanup run even on
    uncaught exceptions or ``sys.exit()``.  ``finally`` blocks in runner
    code are the primary cleanup path; this is the safety net.
    """
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    _cleanup_nccl_shm()


# ============================================================================
# Distributed Setup & Status
# ============================================================================

def setup_distributed() -> None:
    """
    Initialize distributed training.
    Assumes launched with torchrun which sets RANK, WORLD_SIZE, LOCAL_RANK env vars.
    Does nothing if not launched with torchrun (single-GPU mode).
    Idempotent - safe to call multiple times.
    """
    global _atexit_registered

    if not is_distributed_launch():
        return
    
    # Skip if already initialized (allows reuse across multiple run() calls)
    if dist.is_initialized():
        return

    # Clean stale NCCL shm from previous crashed runs.
    # Every rank cleans independently — concurrent removes are harmless.
    _cleanup_nccl_shm()
    
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # NCCL 2.27.5 has a bug in its shared-memory transport that causes
    # UnicodeDecodeError / ncclSystemError when GPUs span multiple PCIe
    # domains (SYS topology).  SHM is only used for intra-node comms —
    # multi-node always uses network sockets — so disabling it has no
    # effect on multi-node performance.  Intra-node falls back to P2P
    # over PCIe which is still fast.
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    
    # Set CUDA device BEFORE any CUDA operations
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend="nccl", 
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    if not _atexit_registered:
        atexit.register(_atexit_cleanup)
        _atexit_registered = True
    
    if rank == 0:
        print(f"Initialized distributed training: {get_world_size()} GPUs")


def is_distributed_launch() -> bool:
    """Check if launched with torchrun by looking for environment variables."""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def is_distributed() -> bool:
    """Check if currently running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank (global). Returns 0 for single-GPU."""
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get('RANK', 0))


def get_local_rank() -> int:
    """Get local rank within this node. Returns 0 for single-GPU.
    
    Use this for device_ids in DDP wrapper, as CUDA_VISIBLE_DEVICES
    remaps GPUs so LOCAL_RANK corresponds to the visible device index.
    """
    return int(os.environ.get('LOCAL_RANK', 0))


def get_world_size() -> int:
    """Get total number of processes. Returns 1 for single-GPU."""
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get('WORLD_SIZE', 1))


def is_main_process() -> bool:
    """Check if this is rank 0 (main process)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes. Does nothing in single-GPU mode."""
    if is_distributed():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def cleanup_distributed() -> None:
    """Clean up distributed process group and NCCL resources.

    Primary cleanup path — called from ``finally`` blocks in runners.
    ``_atexit_cleanup`` is the safety net for cases where this isn't reached.
    Safe to call in single-GPU mode or multiple times.
    """
    if is_distributed():
        dist.destroy_process_group()
    _cleanup_nccl_shm()


# ============================================================================
# Model Unwrapping Helper
# ============================================================================

def get_raw_model(model: nn.Module) -> nn.Module:
    """
    Get underlying model, unwrapping DDP and torch.compile if needed.
    
    Handles wrapping order: DDP(CompiledModel(BaseModel))
    (compile-before-DDP is the PyTorch recommended pattern)
    
    Use this when you need to access:
    - model.config
    - Custom methods like model.get_params(), model.ablate()
    - Model attributes like model.model_type
    
    Examples:
        raw_model = get_raw_model(model)
        embed_dim = raw_model.config.embed_dim
        params = raw_model.get_params('core')
    """
    # Unwrap DDP first (outermost wrapper)
    if isinstance(model, DDP):
        model = model.module
    
    # Then unwrap torch.compile (check if it has _orig_mod attribute)
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    
    return model


# ============================================================================
# Collective Operations
# ============================================================================

def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    All-reduce a tensor across processes.
    
    Use this to aggregate metrics (loss, accuracy) across GPUs.
    Does nothing in single-GPU mode.
    
    Example:
        loss_tensor = torch.tensor(loss, device=model.device)
        avg_loss = reduce_tensor(loss_tensor).item()
    """
    if not is_distributed():
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= get_world_size()
    return tensor


def broadcast_object(obj: object, src: int = 0) -> object:
    """
    Broadcast any picklable object from source rank to all ranks.
    
    Use this to synchronize random state, sampled labels, etc.
    Does nothing in single-GPU mode.
    
    Example:
        # Main process samples, others receive the same value
        label = random.choice(labels) if is_main_process() else None
        label = broadcast_object(label)  # Now all ranks have same label
    """
    if not is_distributed():
        return obj
    
    obj_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def all_reduce_dict(data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    All-reduce a dictionary of tensors across all processes.
    
    Args:
        data_dict: Dictionary of tensors to reduce
        
    Returns:
        Dictionary with reduced tensors
    """
    if not (dist.is_available() and dist.is_initialized()):
        return data_dict
    
    world_size = get_world_size()
    if world_size == 1:
        return data_dict
    
    # Reduce each tensor in the dictionary
    for key, tensor in data_dict.items():
        if torch.is_tensor(tensor):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            data_dict[key] = tensor / world_size
    
    return data_dict
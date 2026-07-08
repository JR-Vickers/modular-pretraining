"""
Pure utility functions with no model or heavy framework dependencies.

This module can be safely imported by any module in the project (including
model/config.py and dataloader.py) without circular imports.

Model-dependent helpers (make_model, copy_model, etc.) live in model_utils.py.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING

import numpy as np
import torch

from src.run.util.distributed import is_main_process

if TYPE_CHECKING:
    from src.run.util.dataloader import DataLoader


def json_safe(obj):
    """Recursively convert dataclasses/Paths to JSON-safe types, skipping non-serializable fields."""
    if is_dataclass(obj) and not isinstance(obj, type):
        from dataclasses import fields
        return {f.name: json_safe(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[k] = json_safe(v)
            except (TypeError, NotImplementedError):
                out[k] = repr(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)



def labels_to_str(labels: Iterable[str]) -> str:
    """
    Sort labels by appending 'core' first, then sorting the rest alphabetically.
    'core' always comes first, remaining labels sorted alphabetically.
    E.g. {"core", "biology"} → "core_biology", {"core"} → "core".
    """
    labels = set(labels)
    parts = []
    if "core" in labels:
        parts.append("core")
        labels.discard("core")
    parts.extend(sorted(labels))
    return "_".join(parts)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)



def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_line(msg: dict, log_fp: Path) -> None:
    if is_main_process():
        with open(log_fp, "a") as f:
            f.write(json.dumps(msg, default=str) + "\n")


def get_batch(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, str | None]:
    batch, label = loader.next_batch()
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y, label


def get_exp_mask(
    labels: list[str],
    selected_labels: Optional[Iterable[str]],
    device: torch.device,
) -> torch.Tensor:
    """Create a boolean expert selection mask of length len(labels)."""
    K = len(labels)
    mask = torch.zeros(K, device=device, dtype=torch.bool)

    if selected_labels is None:
        mask[:] = True
    else:
        label_set = set(labels)
        for e in selected_labels:
            assert e in label_set, f"Unknown expert label '{e}' not in {labels}"
            mask[labels.index(e)] = True

    return mask


def log_batch_counts(batches: list[tuple[str, tuple, tuple] | str], logger: logging.Logger) -> None:

    if not batches:
        logger.info("Batch group is empty (0 batches)")
        return

    if type(batches[0]) == tuple:
        assert len(batches[0]) == 3
        batches = sorted(batches, key=lambda x: (x[1], x[0], x[2]))
    else:
        batches = sorted(batches)

    batch_counts = {}
    for batch in batches:
        if batch not in batch_counts:
            batch_counts[batch] = 0
        batch_counts[batch] += 1
    for batch, count in batch_counts.items():
        logger.info(f"Batch [{batch}] count: {count}")

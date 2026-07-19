"""Reproducible symmetric weight-only fake quantization.

This module deliberately implements a weight-grid perturbation rather than an
integer inference path. Quantized weights are immediately dequantized to fp32.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any, Iterable, Literal

import torch
from torch import nn


Granularity = Literal["per_tensor", "per_channel"]
GROUPS = ("core_mlp", "aux_modules", "attention", "embeddings")


def fake_quantize_tensor(
    weight: torch.Tensor,
    bit_width: int,
    granularity: Granularity = "per_channel",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and dequantize a matrix using a symmetric narrow signed grid."""
    if not isinstance(bit_width, int) or isinstance(bit_width, bool) or bit_width < 2:
        raise ValueError("bit_width must be an integer greater than or equal to 2")
    if granularity not in ("per_tensor", "per_channel"):
        raise ValueError(f"Unsupported granularity: {granularity}")
    if weight.ndim != 2:
        raise ValueError(f"Expected a matrix weight, got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError("Expected a floating-point weight")

    source = weight.detach().to(dtype=torch.float32)
    qmax = 2 ** (bit_width - 1) - 1
    if granularity == "per_tensor":
        scale = source.abs().amax().reshape(1) / qmax
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    else:
        scale = source.abs().amax(dim=1, keepdim=True) / qmax
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(source / safe_scale).clamp(-qmax, qmax)
    dequantized = quantized * scale
    return dequantized, scale


def classify_matrix_weights(model: nn.Module) -> dict[str, str]:
    """Assign every matrix parameter to exactly one Phase 3 parameter group."""
    is_gram = hasattr(model, "labels") and any(".moe.experts." in name for name, _ in model.named_parameters())
    assignments: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2:
            continue
        matches: list[str] = []
        if name in ("embed.weight", "unembed.weight"):
            matches.append("embeddings")
        if ".attn." in name:
            matches.append("attention")
        if is_gram and ".moe.experts." in name:
            try:
                expert_index = int(name.split(".moe.experts.", 1)[1].split(".", 1)[0])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Cannot classify GRAM matrix weight {name}") from exc
            matches.append("core_mlp" if expert_index == 0 else "aux_modules")
        if not is_gram and ".mlp." in name:
            matches.append("core_mlp")
        if len(matches) != 1:
            reason = "overlapping" if matches else "unclassified"
            raise ValueError(f"{reason} matrix weight {name}: {matches}")
        assignments[name] = matches[0]
    if not assignments:
        raise ValueError("Model has no eligible matrix weights")
    if not is_gram and "aux_modules" in assignments.values():
        raise ValueError("Dense models cannot contain auxiliary-module weights")
    return assignments


def _error_statistics(source: torch.Tensor, quantized: torch.Tensor, scale: torch.Tensor) -> dict[str, Any]:
    source = source.detach().to(dtype=torch.float32)
    error = quantized - source
    source_l2 = torch.linalg.vector_norm(source)
    error_l2 = torch.linalg.vector_norm(error)
    relative_l2 = 0.0 if source_l2.item() == 0 else (error_l2 / source_l2).item()
    return {
        "element_count": source.numel(),
        "scale_count": scale.numel(),
        "scale_min": scale.min().item(),
        "scale_mean": scale.mean().item(),
        "scale_max": scale.max().item(),
        "mae": error.abs().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "max_absolute_error": error.abs().max().item(),
        "relative_l2_error": relative_l2,
        "source_l2": source_l2.item(),
        "error_l2": error_l2.item(),
        "absolute_error_sum": error.abs().sum().item(),
        "squared_error_sum": error.square().sum().item(),
    }


def _aggregate_statistics(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(item["element_count"] for item in items)
    scale_count = sum(item["scale_count"] for item in items)
    squared_error = sum(item["squared_error_sum"] for item in items)
    source_l2_squared = sum(item["source_l2"] ** 2 for item in items)
    error_l2_squared = sum(item["error_l2"] ** 2 for item in items)
    return {
        "parameter_count": len(items),
        "element_count": count,
        "scale_count": scale_count,
        "scale_min": min(item["scale_min"] for item in items),
        "scale_mean": sum(item["scale_mean"] * item["scale_count"] for item in items) / scale_count,
        "scale_max": max(item["scale_max"] for item in items),
        "mae": sum(item["absolute_error_sum"] for item in items) / count,
        "rmse": math.sqrt(squared_error / count),
        "max_absolute_error": max(item["max_absolute_error"] for item in items),
        "relative_l2_error": 0.0 if source_l2_squared == 0 else math.sqrt(error_l2_squared / source_l2_squared),
    }


def quantize_model_copy(
    model: nn.Module,
    bit_width: int,
    granularity: Granularity = "per_channel",
    selected_groups: Iterable[str] = GROUPS,
) -> tuple[nn.Module, dict[str, Any]]:
    """Return an independently quantized fp32 model and detailed error statistics."""
    assignments = classify_matrix_weights(model)
    selected = tuple(dict.fromkeys(selected_groups))
    unknown = set(selected) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown quantization groups: {sorted(unknown)}")
    available = set(assignments.values())
    unavailable = set(selected) - available
    if unavailable:
        raise ValueError(f"Groups unavailable for this model: {sorted(unavailable)}")
    if not selected:
        raise ValueError("At least one quantization group must be selected")

    quantized_model = copy.deepcopy(model).to(dtype=torch.float32)
    source_parameters = dict(model.named_parameters())
    target_parameters = dict(quantized_model.named_parameters())
    per_parameter: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with torch.no_grad():
        for name, group in assignments.items():
            if group not in selected:
                continue
            source = source_parameters[name].detach()
            dequantized, scale = fake_quantize_tensor(source, bit_width, granularity)
            target_parameters[name].copy_(dequantized.to(device=target_parameters[name].device))
            stats = {"group": group, **_error_statistics(source, dequantized, scale)}
            per_parameter[name] = stats
            grouped[group].append(stats)
    per_group = {group: _aggregate_statistics(items) for group, items in grouped.items()}
    return quantized_model, {
        "bit_width": bit_width,
        "granularity": granularity,
        "selected_groups": list(selected),
        "per_parameter": per_parameter,
        "per_group": per_group,
        "overall": _aggregate_statistics(list(per_parameter.values())),
    }

"""Run or benchmark the eager FP32, single-device Phase 1 GRAM smoke test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.stories_utils import download_missing_stories, validate_stories_data
from src.model.config import ModelConfig, RoutedModelConfig
from src.model.moe import MoETransformer
from src.run.experiment.config import GetStoriesConfig
from src.run.main import run
from src.run.train.routed import UnorderedConfig
from src.run.util.config import resolve_device, resolve_dtype, use_fused_adamw
from src.run.util.tools import get_timestamp


DEFAULT_TOKEN_BUDGET = 10_000_000
DEFAULT_CORE_TOKENS = 9_156_912
DEFAULT_AUX_TOKENS = 843_088
REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = REPO_ROOT / "src/data/stories"


def model_config(shape: str) -> ModelConfig:
    if shape == "paper":
        return ModelConfig(
            ctx_len=256, vocab_size=4096, num_layers=8, embed_dim=512,
            mlp_dim=2048, num_heads=8, num_key_value=2,
            attn_bias=True, eos_token_id=1,
        )
    if shape == "small":
        return ModelConfig(
            ctx_len=256, vocab_size=4096, num_layers=4, embed_dim=256,
            mlp_dim=1024, num_heads=8, num_key_value=2,
            attn_bias=True, eos_token_id=1,
        )
    raise ValueError(f"Unknown model shape: {shape}")


def split_token_budget(token_budget: int) -> tuple[int, int]:
    if token_budget <= 0:
        raise ValueError("token budget must be positive")
    core = round(token_budget * DEFAULT_CORE_TOKENS / DEFAULT_TOKEN_BUDGET)
    return core, token_budget - core


def routed_model_config(shape: str) -> RoutedModelConfig:
    return RoutedModelConfig.from_base(
        model_config(shape), arch="moe", core_param_prc=1.0, aux_param_prc=0.1
    )


def make_smoke_config(args: argparse.Namespace):
    config = GetStoriesConfig()
    config.model = model_config(args.model_shape)
    core_tokens, aux_tokens = split_token_budget(args.token_budget)
    config.data.core.method = "count"
    config.data.core.limit = core_tokens
    config.data.aux.method = "count"
    config.data.aux.limit = aux_tokens

    config.run.res_root = REPO_ROOT / "results/stories_smoke/seed_1"
    config.run.seed = 1
    config.run.device = args.device
    config.run.dtype = args.dtype
    config.run.compile = False
    config.run.cleanup_distributed = False
    config.run.s3_bucket = None
    config.run.s3_prefix = None
    config.run.epochs = 1
    config.run.target_effective_batch_size = -1
    config.run.micro_batch_size = 16
    config.run.accumulation_steps = 8
    config.run.max_num_test_sequences = 128
    config.run.limit_eval_sequences = True
    config.run.model_shape = args.model_shape
    config.run.model_shape_note = (
        "Matches the 26M dense / 32.57M GRAM paper shape"
        if args.model_shape == "paper"
        else "Timed Phase 1 fallback; not a paper-replication result"
    )
    config.run.nominal_token_budget = args.token_budget

    labels = ["core"] + sorted(config.data.aux.labels)
    profiles = [labels] + [sorted(set(labels) - {label}) for label in labels[1:]]
    config.stages = [
        UnorderedConfig(
            model=routed_model_config(args.model_shape),
            robust_prc=0.5,
            aux_route_prc=0.3,
            num_train_evals=0,
            num_checkpoints=2,
            do_elicit=False,
            lr=5e-3,
            equal_compute=True,
            acc_mode="heterogeneous",
            retain_targets=profiles,
        )
    ]
    return config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    labels = ["core", "aux_1", "aux_2", "aux_3", "aux_4"]
    model = MoETransformer(routed_model_config(args.model_shape), labels).to(device, dtype=dtype)
    optimizers = {
        label: torch.optim.AdamW(
            list(model.get_params(label)),
            lr=5e-3,
            fused=use_fused_adamw(device),
            betas=(0.9, 0.95),
        )
        for label in labels
    }
    micro_batch, accumulation_steps = 16, 8
    cfg = model.config
    micro_step = 0

    def route_masks(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic approximation of the 91.6/8.4 data mix and p_cr/p_as."""
        fwd = torch.zeros(len(labels), dtype=torch.bool, device=device)
        bck = torch.zeros(len(labels), dtype=torch.bool, device=device)
        if index % 12 == 0:  # approximately 8.4% auxiliary data
            auxiliary_index = 1 + (index // 12) % 4
            fwd[0] = True
            fwd[auxiliary_index] = True
            bck[auxiliary_index] = True
            if (index // 12) % 10 < 3:  # p_as = 0.3
                bck[0] = True
        elif index % 2 == 0:  # p_cr = 0.5 on core data
            auxiliary_index = 1 + (index // 2) % 4
            fwd[0] = True
            fwd[auxiliary_index] = True
            bck[0] = True
            bck[auxiliary_index] = True
        else:
            fwd[0] = True
            bck[0] = True
        return fwd, bck

    def effective_batch() -> None:
        nonlocal micro_step
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation_steps):
            tokens = torch.randint(
                0, cfg.vocab_size, (micro_batch, cfg.ctx_len), device=device
            )
            fwd_mask, bck_mask = route_masks(micro_step)
            _, loss = model(tokens, tokens, fwd_mask=fwd_mask, bck_mask=bck_mask)
            (loss / accumulation_steps).backward()
            micro_step += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer in optimizers.values():
            has_gradient = any(
                parameter.grad is not None
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            if has_gradient:
                optimizer.step()

    effective_batch()  # compile/cache/allocation warmup
    synchronize(device)
    started = time.perf_counter()
    for _ in range(10):
        effective_batch()
    synchronize(device)
    elapsed = time.perf_counter() - started

    tokens_per_batch = micro_batch * accumulation_steps * cfg.ctx_len
    projected_seconds = elapsed / 10 * (args.token_budget / tokens_per_batch)
    run_dir = REPO_ROOT / "results/stories_smoke/seed_1" / get_timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    output = {
        "device": str(device),
        "dtype": str(dtype),
        "model_shape": args.model_shape,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "synthetic_routing": "91.6/8.4 core/aux mix with p_cr=0.5 and p_as=0.3",
        "warmup_effective_batches": 1,
        "timed_effective_batches": 10,
        "timed_seconds": elapsed,
        "seconds_per_effective_batch": elapsed / 10,
        "tokens_per_effective_batch": tokens_per_batch,
        "nominal_token_budget": args.token_budget,
        "projected_seconds": projected_seconds,
        "projected_hours": projected_seconds / 3600,
        "paper_shape_allowed_by_six_hour_rule": projected_seconds <= 6 * 3600,
    }
    path = run_dir / "benchmark_projection.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"Wrote {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--model-shape", choices=("paper", "small"), default="paper")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="mps")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Validate local shards without downloading missing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark_only:
        benchmark(args)
        return
    if not args.skip_download:
        downloaded = download_missing_stories(DATA_DIR)
        print(f"Downloaded {len(downloaded)} missing stories shards")
    print(json.dumps(validate_stories_data(DATA_DIR), indent=2))
    run(make_smoke_config(args))


if __name__ == "__main__":
    main()

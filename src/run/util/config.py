"""
Run configuration and setup.

This module is the single entry point for initialising everything a training
run needs: CUDA device, tokenizer, data loaders, model config, results
directory, and logging.  The ``setup()`` function is called once at the start
of ``run()`` in main.py and returns a dict of ``RunConfig`` + ``ModelConfig``.

Data budget
-----------
Two mechanisms control how much data each label sees:

- **core_batch_limit**: can be an int (exact count), ``"optimal"`` (Chinchilla
  scaling law heuristic: ~20 tokens per parameter), or None (use all data).
- **aux_batch_limit**: can be an int, a float (fraction of core batch count),
  or None (use all data).
- **core_dist / aux_dist**: per-label proportions within core and aux budgets.
  Must each sum to 1.0.

These are resolved into per-label sequence budgets and passed to ``make_loaders()``.
"""

from __future__ import annotations

from typing import Literal
import json
import logging
import subprocess
import time
import warnings
import math
import numpy as np
from dotenv import load_dotenv
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.model.config import ModelConfig
from src.run.util.dataloader import (
    DataLoader,
    make_loaders,
    get_labels_token_count,
    auto_detect_categories,
)
from src.run.util.tools import ensure_dir, get_timestamp, set_seeds, json_safe
from src.run.util.logger import setup_logger
from src.run.util.distributed import (
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    broadcast_object,
    is_distributed,
    setup_distributed,
    is_distributed_launch,
)
from src.run.util.preemption import setup_preemption
from src.run.util.s3 import setup_s3, sync_from_s3, start_watcher
from src.model.base import BaseTransformer, assert_cuda_gqa_equivalence

# --------------------------------------------------------------------------- #
# global log suppression for TorchDynamo recompilation warnings               #
# --------------------------------------------------------------------------- #

# Silence TorchDynamo recompilation warnings without setting invalid TORCH_LOGS
warnings.filterwarnings(
    "ignore",
    message=r".*torch\._dynamo.*recompile_limit.*",
    category=UserWarning,
)

# Silence Dynamo warnings about DDP's _broadcast_coalesced (can't trace through DDP internals)
warnings.filterwarnings(
    "ignore",
    message=r".*_broadcast_coalesced.*",
    category=UserWarning,
)

# Reduce logger verbosity for torch._dynamo in the current process
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


# --------------------------------------------------------------------------- #
# dataclasses                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class StageConfig:
    """Base configuration shared by all stage types."""
    name: str = ""
    num_checkpoints: int = -1 # number of checkpoints to save during training
    lr: float = 1e-4
    res_dir: Path = field(default_factory=Path)
    acc_mode: Literal["uniform", "heterogeneous"] = "uniform"

    #eval args
    do_eval: bool = True #do eval on test set?
    do_sample: bool = False #generate samples?
    num_train_evals: int = 0 #number of test evals during training
    do_elicit: bool = True #do adversarial ft?
    elicit_eval_prc: float = 0.5 #what point during adversarial ft to save stats.jsonl entry?
    elicit_num_evals: int = 100 #number of validation evaluations during adversarial ft

    # Override default retain-target iteration for unlearning/filtering stages.
    # If None, get_retain_targets(aux_labels) is used.
    retain_targets: list[list[str]] | None = None

    @property
    def state_path(self) -> Path:
        return self.res_dir / "stage.json"

@dataclass
class DataLabelConfig:
    """Configuration for data labels."""
    labels: list[str] = field(default_factory=list)
    # "optimal"/"total"/"count" all cap num_tok at this dtype's own available
    # tokens (downsample-only). "dataset" instead targets ``limit`` x the FULL
    # dataset (every label's tokens) and is NOT capped -- requesting more than a
    # side's available tokens is honoured by upsampling (repeating) it in the
    # loader. Used to pin a fixed core/aux composition independent of how many
    # labels fall on each side (see experiment/auxnum/run.py).
    method: Literal["optimal", "total", "count", "dataset"] = "total" #method of determining data budget
    limit: float | int = 1.0
    dist: dict[str, float] = field(default_factory=dict) # maps label to proportion of budget
    num_tok: int = -1

@dataclass
class DataConfig:
    """Configuration for data limits."""
    dirs: list[Path] = field(default_factory=list)
    core: DataLabelConfig = field(default_factory=DataLabelConfig)
    aux: DataLabelConfig = field(default_factory=DataLabelConfig)

@dataclass
class RunConfig:
    """Shared onfiguration for runtime over all stages."""
    target_effective_batch_size: int = 128
    effective_batch_size: int = -1
    micro_batch_size: int = -1
    accumulation_steps: int = -1
    max_num_test_sequences: int = 10000
    limit_eval_sequences: bool = False
    eval_all_labels: bool = False #eval every loader label?
    micro_batch_anchors: tuple[tuple[float, int], tuple[float, int]] = ((400e6, 23), (5e9, 2))
    epochs: int = 1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    warmup_prc: float = 0.02
    decay_prc: float = 0.1
    seed: int = 42
    compile: bool = True
    find_unused_parameters: bool = True
    is_ddp: bool = False
    cleanup_distributed: bool = True
    log_level: str = "INFO"
    num_gpus: int = 1
    process_id: int = -1
    num_base_params: int = -1
    device: str | torch.device = "auto"
    dtype: str | torch.dtype = "auto"
    logger: logging.Logger = field(default_factory=logging.getLogger)
    labels: list[str] = field(default_factory=list)
    loaders: dict[str, DataLoader] = field(default_factory=dict)
    res_root: Path = field(default_factory=Path)
    experiment_id: str = field(default_factory=get_timestamp)
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    model_shape: str | None = None
    model_shape_note: str | None = None
    nominal_token_budget: int | None = None

    @property
    def res_dir(self) -> Path:
        return self.res_root / self.experiment_id

@dataclass
class ExperimentConfig:
    """Typed configuration for a full experiment run.

    This is the top-level config passed to ``run()`` / ``setup()``.
    Use ``dataclasses.replace()`` to derive per-run variants, or subclass
    with overridden defaults for a specific experiment template
    (see ``StoriesConfig`` / ``RealisticConfig`` in orchestrate/config).
    """

    stages: list[StageConfig] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig = field(default_factory=RunConfig)
    label_prc: float = 1.0


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

_DTYPE_NAMES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve a requested runtime device without silently changing it."""
    name = str(requested).lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {requested!r}")
    return device


def resolve_dtype(requested: str | torch.dtype, device: torch.device) -> torch.dtype:
    """Resolve the model dtype; automatic mode preserves CUDA BF16 behavior."""
    if isinstance(requested, torch.dtype):
        dtype = requested
    else:
        name = requested.lower()
        if name == "auto":
            return torch.bfloat16 if device.type == "cuda" else torch.float32
        try:
            dtype = _DTYPE_NAMES[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {requested!r}") from exc

    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 training is not supported on CPU; use float32 or bfloat16")
    return dtype


def use_fused_adamw(device: str | torch.device) -> bool:
    """The fused AdamW implementation is retained only for CUDA runs."""
    return torch.device(device).type == "cuda"


def validate_stages(stages: list[StageConfig]) -> None:
    """Validate stage dependencies."""
    stage_names = [s.name for s in stages]
    acceptable_stages = [
        'baseline',
        'rmu',
        'ascent',
        'maxent',
        'filtering',
        'coreftaux',
        'routed',
    ]

    assert all(stage in acceptable_stages for stage in stage_names), f"Invalid stages: {stage_names}"

    if any(x in stage_names for x in ["rmu", "gradient_ascent", "maxent"]):
        assert "baseline" in stage_names, "Baseline model is required for posthoc unlearning"


def setup_tokenizer(metadata_path: Path, logger: logging.Logger) -> AutoTokenizer:
    """Setup tokenizer from metadata."""
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {metadata_path}. Please run data prep.")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    tokenizer_name = metadata["all"].get("tokenizer")
    if tokenizer_name is None:
        tokenizer_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    vocab_size = len(tokenizer)
    logger.info(f"Tokenizer vocabulary: {vocab_size}")

    metadata_vocab_size = metadata["all"].get("vocab_size")
    if metadata_vocab_size is not None:
        assert vocab_size == metadata_vocab_size, f"Vocab size mismatch: tokenizer={vocab_size}, metadata={metadata_vocab_size}"
    else:
        logger.warning(f"No vocab_size in metadata.json — skipping validation (tokenizer has {vocab_size})")

    return tokenizer


def get_git_info() -> tuple[str, str]:
    """Return (branch, commit) for the repo containing this file.

    Falls back to "unknown" for either field if git is unavailable or the
    working directory is not a git repo (e.g. shipped as a tarball).
    """
    repo_dir = Path(__file__).resolve().parent
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo_dir,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return "unknown"

    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["rev-parse", "HEAD"])
    return branch, commit


def get_stage_dirs(stages: list[StageConfig]) -> list[str]:
    """Compute directory names for each stage, deduplicating when necessary.

    Single-occurrence names get their plain name (e.g. "baseline").
    Duplicate names get a postfix: "routed_01", "routed_02", etc.

    Returns a list parallel to the input stages list.
    """
    names = [s.name for s in stages]
    counts = Counter(names)
    seen: dict[str, int] = {}
    dirs = []
    for name in names:
        if counts[name] == 1:
            dirs.append(name)
        else:
            idx = seen.get(name, 0) + 1
            seen[name] = idx
            dirs.append(f"{name}_{idx:02d}")
    return dirs


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #


def setup(config: ExperimentConfig) -> ExperimentConfig:
    """
    Takes a partially filled ExperimentConfig, fills missing values, and initializes environment.
    """
    load_dotenv(override=True)

    device = resolve_device(config.run.device)
    dtype = resolve_dtype(config.run.dtype, device)
    if device.type != "cuda" and is_distributed_launch():
        raise RuntimeError("MPS and CPU runs must be launched as a single process (without torchrun)")

    setup_distributed()
    setup_preemption()
    setup_s3(config)

    assert "core" not in config.data.aux.labels, "core cannot be an aux label"
    assert len(config.data.dirs) > 0, "data_dirs must be provided"

    validate_stages(config.stages)

    # Synchronize experiment ID across ranks (timestamps differ per process)
    config.run.experiment_id = broadcast_object(
        config.run.experiment_id if is_main_process() else None, src=0
    )

    if config.run.seed == -1:
        seed = int(time.time()) if is_main_process() else None
        seed = broadcast_object(seed, src=0)
        config.run.seed = seed

    set_seeds(config.run.seed)

    # Backend setup. Keep the existing CUDA settings unchanged.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.set_float32_matmul_precision("high")
        assert_cuda_gqa_equivalence(device)
    config.run.device = device
    config.run.dtype = dtype
    config.run.is_ddp = is_distributed()

    # Only create directory on main process to avoid duplicates in multigpu mode
    if is_main_process():
        ensure_dir(config.run.res_dir)
    
    # Ensure all processes wait for directory creation and use the same path
    barrier()

    # Pull existing files from S3 before checking stage.json
    sync_from_s3(config)
    barrier()

    #define the stage directories
    stage_dirs = get_stage_dirs(config.stages)
    for idx, stage_dir in enumerate(stage_dirs):
        config.stages[idx].res_dir = config.run.res_dir / stage_dir

    # Write initial stage.json for each stage if it doesn't exist yet
    if is_main_process():
        for stage in config.stages:
            state_path = stage.state_path
            if not state_path.exists():
                data = {"completed": False, "completed_iterations": [], "stage": json_safe(stage)}
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(data, indent=2))

    # Start background S3 watcher (after sync_down + stage.json init so
    # the initial snapshot doesn't trigger re-uploads of existing files)
    start_watcher(config)
    barrier()

    # Setup logger
    config.run.process_id = get_rank()
    log_file = config.run.res_dir / "training.log"
    logger = setup_logger(
        name=f"training_{config.run.experiment_id}",
        log_file=log_file,
        level=config.run.log_level,
        process_id=config.run.process_id,
    )
    config.run.logger = logger

    branch, commit = get_git_info()
    logger.info(f"Git branch: {branch}, commit: {commit}")

    # Get number of GPUs
    config.run.num_gpus = get_world_size()
    logger.info(f"Number of processes: {config.run.num_gpus}")
    logger.info(f"Runtime device: {config.run.device}, dtype: {config.run.dtype}")

    # Setup tokenizer
    dataset_metadata_path = config.data.dirs[0] / "metadata.json"
    tokenizer = setup_tokenizer(dataset_metadata_path, logger)
    config.model.tokenizer = tokenizer
    config.model.eos_token_id = tokenizer.eos_token_id
    for stage in config.stages:
        if stage.name == "routed":
            stage.model.tokenizer = tokenizer
            stage.model.eos_token_id = tokenizer.eos_token_id

    #round vocab size to nearest multiple of 64
    vocab_size = len(tokenizer)
    config.model.vocab_size = 64 * ((vocab_size + 63) // 64)
    logger.info(f"Set model vocab size to nearest multiple of 64 of tokenizer size: {config.model.vocab_size}")

    temp_model = BaseTransformer(config.model)
    num_base_params = sum(p.numel() for p in temp_model.parameters())
    config.run.num_base_params = num_base_params
    del temp_model

    # Derive acc steps and micro batch size from effective batch size
    target_eff = config.run.target_effective_batch_size

    if target_eff == -1:

        assert config.run.micro_batch_size > 0, "micro batch size must be provided if target effective batch size is not provided"
        assert config.run.accumulation_steps > 0, "accumulation steps must be provided if target effective batch size is not provided"
        assert config.run.num_gpus > 0, "number of GPUs must be provided if target effective batch size is not provided"
        eff_bs = config.run.micro_batch_size * config.run.accumulation_steps * config.run.num_gpus
        config.run.effective_batch_size = eff_bs

    else:

        num_gpus = config.run.num_gpus
        if num_gpus > target_eff:
            logger.warning(f"Number of GPUs {num_gpus} is greater than effective batch size {target_eff}")

        # Compute max micro batch size from anchors (upper bound from GPU memory)
        (N1, B1), (N2, B2) = config.run.micro_batch_anchors
        exp = math.log(B2 / B1) / math.log(N2 / N1)
        coef = B1 / (N1 ** exp)
        max_micro = max(1, round(coef * config.run.num_base_params ** exp))

        # Find largest micro batch size <= max_micro that hits the target exactly,
        # falling back to within 5% if no exact divisor exists.
        best_eff, best_micro, best_acc = num_gpus, 1, 1
        for m in range(max_micro, 0, -1):
            if target_eff % (m * num_gpus) == 0:
                a = target_eff // (m * num_gpus)
                best_eff, best_micro, best_acc = target_eff, m, a
                break
        else:
            for m in range(max_micro, 0, -1):
                a = max(1, round(target_eff / (m * num_gpus)))
                candidate = m * a * num_gpus
                if abs(candidate - target_eff) <= 0.05 * target_eff:
                    best_eff, best_micro, best_acc = candidate, m, a
                    break
        micro_batch_size = best_micro
        acc_steps = best_acc
        eff_batch_size = best_eff

        config.run.effective_batch_size = eff_batch_size
        config.run.accumulation_steps = acc_steps
        config.run.micro_batch_size = micro_batch_size

    logger.info(
        f"target effective batch size: {config.run.target_effective_batch_size}, "
        f"actual effective batch size: {config.run.effective_batch_size}, "
        f"micro batch size: {config.run.micro_batch_size}, "
        f"accumulation steps: {config.run.accumulation_steps}, "
        f"num gpus: {config.run.num_gpus}"
    )

    data_dirs = [Path(d) for d in config.data.dirs]
    config.data.dirs = data_dirs

    categories = {}
    for data_dir in config.data.dirs:
        categories.update(auto_detect_categories(data_dir))
    all_labels = sorted(categories.keys())
    assert len(all_labels) > 0, "no labels found in data directories"
    
    core_labels = config.data.core.labels
    if not core_labels:
        core_labels = sorted(set(all_labels) - set(config.data.aux.labels))
        config.data.core.labels = core_labels

    config.run.labels = ["core"] + sorted(config.data.aux.labels)

    for dtype in ("core", "aux"):

        label_cfg = getattr(config.data, dtype)

        # --------- token limit ---------
        max_tok = get_labels_token_count(
            data_dirs=config.data.dirs,
            labels=label_cfg.labels,
        )
        logger.info(f"{dtype} total number of tokens: {max_tok}")

        if label_cfg.method in ("optimal", "total"):
            assert 0 <= label_cfg.limit <= 1.0, f"{dtype} limit must be less than or equal to 1.0"
        elif label_cfg.method == "count":
            assert label_cfg.limit > 0, f"{dtype} limit must be greater than 0"
        elif label_cfg.method == "dataset":
            assert 0 <= label_cfg.limit <= 1.0, f"{dtype} limit must be in [0, 1.0]"

        if label_cfg.method == "optimal":

            # Chinchilla scaling law heuristic: ~20 tokens per parameter is roughly
            # compute-optimal for dense transformers. We convert to per-rank
            # sequence count so the sampled sequence stream is batch-size invariant.
            logger.info(f"Baseline model has {config.run.num_base_params:,} parameters")
            optimal_tok = config.run.num_base_params * 20
            num_tok = optimal_tok * label_cfg.limit
            logger.info(
                f"{dtype}: chinchilla optimal tokens: {optimal_tok:,}, "
                f"requested proportion: {label_cfg.limit:.4f}, "
                f"requested number of tokens: {num_tok:,}, "
            )
            if num_tok > max_tok:
                logger.warning(
                    f"Requested {dtype} sequence limit {num_tok} is greater than max {max_tok}, taking min"
                )
                num_tok = min(num_tok, max_tok)

            label_cfg.num_tok = num_tok

        elif label_cfg.method == "total":
            num_tok = int(round(label_cfg.limit * max_tok))
            logger.info(f"{dtype}: {label_cfg.limit:.4f} x {max_tok} = {num_tok}")
            label_cfg.num_tok = num_tok

        elif label_cfg.method == "count":
            num_tok = label_cfg.limit
            if num_tok > max_tok:
                logger.warning(f"Requested {dtype} count {num_tok} is greater than max {max_tok}, taking min")
                num_tok = min(num_tok, max_tok)
            logger.info(f"{dtype}: {label_cfg.limit} tokens")
            label_cfg.num_tok = num_tok

        elif label_cfg.method == "dataset":
            # Target a fraction of the FULL dataset (all labels), NOT capped at
            # this side's own availability: if the request exceeds max_tok the
            # shortfall is made up by upsampling (repetition) in make_loaders.
            total_all_tok = get_labels_token_count(
                data_dirs=config.data.dirs,
                labels=all_labels,
            )
            num_tok = int(round(label_cfg.limit * total_all_tok))
            logger.info(
                f"{dtype}: {label_cfg.limit:.4f} x full-dataset {total_all_tok:,} "
                f"= {num_tok:,} ({'upsample' if num_tok > max_tok else 'downsample'} "
                f"from {max_tok:,} available)"
            )
            label_cfg.num_tok = num_tok

        else:
            raise ValueError(f"Unknown data method: {label_cfg.method}")

        if len(label_cfg.dist.keys()) == 0:
            labels = label_cfg.labels
            dist = {}
            for label in labels:
                dist[label] = 1.0 / len(labels)
            label_cfg.dist = dist

        assert abs(sum(label_cfg.dist.values()) - 1.0) < 1e-10, "dist must sum to 1.0"
        assert all(x in label_cfg.dist.keys() for x in label_cfg.labels), "all labels must be in dist"
  
    label_counts = {}
    
    for label in config.data.core.labels:
        label_counts[label] = round(config.data.core.dist[label] * config.data.core.num_tok)

    for label in config.data.aux.labels:
        label_counts[label] = round(config.data.aux.dist[label] * config.data.aux.num_tok)

    # Only labels whose budget came from the uncapped "dataset" method may be
    # upsampled (repeated) past their available tokens; everything else keeps
    # the historical downsample-only behaviour.
    upsample_labels: set[str] = set()
    for dtype in ("core", "aux"):
        label_cfg = getattr(config.data, dtype)
        if label_cfg.method == "dataset":
            upsample_labels.update(label_cfg.labels)


    target_max_test_seq = config.run.max_num_test_sequences
    eff_bs = config.run.effective_batch_size
    max_test_sequences = int(max(eff_bs, np.ceil(target_max_test_seq / eff_bs) * eff_bs))
    max_test_tokens = max_test_sequences * config.model.ctx_len

    # Setup data loaders
    loaders = make_loaders(
        data_dirs=data_dirs,
        aux_labels=config.data.aux.labels,
        core_labels=config.data.core.labels,
        B=config.run.micro_batch_size,
        T=config.model.ctx_len,
        num_processes=config.run.num_gpus,
        seed=config.run.seed,
        device=device,
        pin_memory=device.type == "cuda",
        label_token_counts=label_counts,
        max_num_test=max_test_tokens,
        upsample_labels=upsample_labels,
    )

    config.run.loaders = loaders

    # Save configuration info
    if is_main_process():
        config_dump = json_safe(config)
        config_text = json.dumps(config_dump, indent=4)
        config_path = config.run.res_dir / "config.json"
        config_path.write_text(config_text)
        logger.debug(f"Saved config to {config_path}")
        logger.debug(config_text)

    return config

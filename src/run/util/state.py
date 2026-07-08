"""
Unified run state management: checkpoints + stage progress state.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

from src.run.util.distributed import get_raw_model, is_main_process, barrier, broadcast_object
from src.run.util.preemption import is_preempted, cancel_forced_exit
from src.run.util.config import StageConfig, ExperimentConfig
from src.run.util.s3 import get_s3_manager
from src.run.util.tools import json_safe


def stage_rel_dir(stage: StageConfig, config: ExperimentConfig) -> str | None:
    try:
        return str(stage.res_dir.relative_to(config.run.res_dir))
    except Exception:
        return None


_STEP_RE = re.compile(r"^checkpoint_step-(\d+)\.pth$")


def parse_step(name: str) -> int:
    if name == "checkpoint.pth":
        return 10**18
    m = _STEP_RE.match(name)
    return int(m.group(1)) if m else -1


def find_latest_local(stage_dir: Path) -> Path | None:
    if not stage_dir.exists():
        return None
    final = stage_dir / "checkpoint.pth"
    if final.exists():
        return final
    best: Path | None = None
    best_step = -1
    for fp in stage_dir.glob("checkpoint_step-*.pth"):
        cur_step = parse_step(fp.name)
        if cur_step > best_step:
            best = fp
            best_step = cur_step
    return best


def list_local_candidates(stage_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not stage_dir.exists():
        return out
    for fp in stage_dir.glob("checkpoint*.pth"):
        if not fp.is_file():
            continue
        step = parse_step(fp.name)
        if step >= 0:
            out[step] = fp
    return out


def list_s3_candidates(s3, stage_rel: str) -> dict[int, str]:
    out: dict[int, str] = {}
    prefix = s3._s3_key(f"{stage_rel}/checkpoint")
    paginator = s3.client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=s3.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                name = Path(key).name
                if not name.endswith(".pth"):
                    continue
                step = parse_step(name)
                if step >= 0:
                    out[step] = key.removeprefix(s3.prefix)
    except Exception:
        return {}
    return out


def should_save(step: int, total_steps: int, checkpoint_freq: int) -> bool:
    """Return True if we should save a checkpoint now.

    - checkpoint_freq > 0: save every N steps, at final step, and on preemption.
    - checkpoint_freq <= 0 (e.g. -1): save only at final step and on preemption (no periodic saves).
    """
    if is_preempted():
        return True
    if step == total_steps:
        return True
    if checkpoint_freq > 0 and step % checkpoint_freq == 0:
        return True
    return False


def save_checkpoint(
    stage: StageConfig,
    model: torch.nn.Module,
    state: dict,
    config: ExperimentConfig,
) -> None:
    step = int(state["step"])
    total_steps = int(state["total_steps"])
    stage_dir = stage.res_dir
    out_name = "checkpoint.pth" if step == total_steps else f"checkpoint_step-{step}.pth"
    out_path = stage_dir / out_name

    if is_main_process():
        stage_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"model": get_raw_model(model).state_dict()}
        payload.update(state)
        tmp_path = out_path.with_suffix(".pth.tmp")
        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(out_path))

        # Eagerly upload to S3 before returning so that non-rank-0 ranks
        # on nodes without shared storage (c2/c3 and c1 /workspace) can
        # fetch the checkpoint via S3 in restore_partial_state. The async
        # S3 watcher will also pick this up, but its poll interval (~30s)
        # races with downstream restore calls.
        s3 = get_s3_manager(config)
        stage_rel = stage_rel_dir(stage, config)
        if s3 is not None and stage_rel is not None:
            s3.upload(str(out_path), f"{stage_rel}/{out_name}")

    barrier()

    if is_preempted():
        cancel_forced_exit()

    if is_main_process():
        _prune_local_checkpoints(stage_dir, stage, config, keep=2)


def _prune_local_checkpoints(
    stage_dir: Path,
    stage: StageConfig,
    config: ExperimentConfig,
    keep: int = 2,
) -> None:
    """Remove old local step-checkpoints, keeping the *keep* most recent.

    Only deletes files that have been confirmed uploaded to S3 (via
    head_object size check). Skips entirely if the background S3 watcher
    is not running (meaning uploads aren't guaranteed).

    The S3 ``head_object`` + ``unlink`` pass runs in a background daemon
    thread so the caller (rank 0 inside ``save_checkpoint``) returns
    immediately. Synchronous execution here previously blocked rank 0
    between the post-save ``barrier()`` and the next collective, which —
    when S3 was slow — stacked up enough latency to trigger the 10-min
    NCCL watchdog on every other rank.
    """

    s3 = get_s3_manager(config)
    if s3 is None:
        return

    stage_rel = stage_rel_dir(stage, config)
    if stage_rel is None:
        return

    def _prune_worker() -> None:
        step_files = sorted(
            [fp for fp in stage_dir.glob("checkpoint_step-*.pth") if fp.is_file()],
            key=lambda p: parse_step(p.name),
        )
        if len(step_files) <= keep:
            return
        for fp in step_files[:-keep]:
            s3_key = s3._s3_key(f"{stage_rel}/{fp.name}")
            try:
                head = s3.client.head_object(Bucket=s3.bucket, Key=s3_key)
                if head["ContentLength"] == fp.stat().st_size:
                    fp.unlink()
            except Exception:
                continue

    threading.Thread(target=_prune_worker, daemon=True, name="ckpt-prune").start()


def restore_partial_state(stage: StageConfig, config: ExperimentConfig) -> dict | None:
    """Load the newest checkpoint for a stage.

    Rank 0 decides which checkpoint is newest (local + S3), broadcasts its
    path and — if sourced from S3 — the S3 key so every rank can download
    independently. This prevents non-rank-0 from calling ``torch.load`` on a
    file that only exists on rank 0 (the case on clusters without shared
    storage), which would silently desync collectives later.
    """
    ckpt_path: str | None = None
    s3_rel_key: str | None = None
    stage_dir = stage.res_dir

    if is_main_process():
        local_candidates = list_local_candidates(stage_dir)
        stage_rel = stage_rel_dir(stage, config)
        s3 = get_s3_manager(config)
        s3_candidates: dict[int, str] = {}
        if s3 is not None and stage_rel is not None:
            s3_candidates = list_s3_candidates(s3, stage_rel)

        all_steps = set(local_candidates.keys()) | set(s3_candidates.keys())
        if all_steps:
            newest_step = max(all_steps)
            if newest_step in local_candidates:
                ckpt_path = str(local_candidates[newest_step])
                if newest_step in s3_candidates:
                    s3_rel_key = s3_candidates[newest_step]
            elif newest_step in s3_candidates and s3 is not None:
                rel_key = s3_candidates[newest_step]
                target = stage_dir / Path(rel_key).name
                target.parent.mkdir(parents=True, exist_ok=True)
                if s3.download(rel_key, str(target)):
                    ckpt_path = str(target)
                    s3_rel_key = rel_key

    ckpt_path = broadcast_object(ckpt_path)
    s3_rel_key = broadcast_object(s3_rel_key)

    if ckpt_path is None:
        logger = config.run.logger
        logger.warning(f"restore_partial_state: no checkpoint found at {stage_dir}")
        return None

    # Non-rank-0 ranks on non-shared-storage clusters need the file locally.
    # If the chosen ckpt has an S3 key (i.e., not local-only on a shared FS),
    # each rank downloads it independently. Safe to call on rank 0 too — the
    # file already exists there and s3.download is a no-op / overwrite.
    if not is_main_process() and s3_rel_key is not None and not Path(ckpt_path).exists():
        s3 = get_s3_manager(config)
        if s3 is not None:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            s3.download(s3_rel_key, ckpt_path)

    # Guard: if a rank still doesn't have the file, fail loudly instead of
    # desyncing in torch.load.
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"restore_partial_state: rank missing ckpt at {ckpt_path} "
            f"(s3_rel_key={s3_rel_key}); check cluster shared-storage / S3"
        )

    barrier()
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def restore_partial(
    model: torch.nn.Module,
    stage: StageConfig,
    config: ExperimentConfig,
) -> tuple[torch.nn.Module, dict | None]:
    state = restore_partial_state(stage, config)
    if state and "model" in state:
        get_raw_model(model).load_state_dict(state.pop("model"))
    return model, state


def is_stage_completed(stage: StageConfig) -> bool:
    path = stage.state_path
    if not path.exists():
        return False
    with open(path, "r") as f:
        data = json.load(f)
    return bool(data.get("completed", False))


def get_completed_iterations(stage: StageConfig) -> set[str]:
    path = stage.state_path
    if not path.exists():
        return set()
    with open(path, "r") as f:
        data = json.load(f)
    return set(data.get("completed_iterations", []))


def mark_iteration_completed(
    stage: StageConfig,
    iteration_key: str,
    config: ExperimentConfig | None = None,
) -> None:
    if not is_main_process():
        return
    path = stage.state_path
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {"completed": False, "completed_iterations": []}
    vals = set(data.get("completed_iterations", []))
    vals.add(iteration_key)
    data["completed_iterations"] = sorted(vals)
    data["stage"] = json_safe(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def mark_stage_completed(stage: StageConfig, config: ExperimentConfig | None = None) -> None:
    if not is_main_process():
        return
    path = stage.state_path
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {"completed": False, "completed_iterations": []}
    data["completed"] = True
    data["stage"] = json_safe(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

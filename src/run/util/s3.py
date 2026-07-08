from __future__ import annotations

import datetime
import logging
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from src.run.util.distributed import is_main_process

if TYPE_CHECKING:
    from src.run.util.config import ExperimentConfig, RunConfig

_logger = logging.getLogger(__name__)

_CHECKPOINT_STEP_RE = re.compile(r"^checkpoint_step-(\d+)\.pth$")


class S3MirrorManager:
    def __init__(self, bucket: str, prefix: str, region: str):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if self.prefix:
            self.prefix += "/"
        self.client = boto3.client(
            "s3",
            region_name=region,
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=30,
            ),
        )

    def _s3_key(self, relative_key: str) -> str:
        return f"{self.prefix}{relative_key}"

    def upload(self, local_path: str, relative_key: str) -> bool:
        s3_key = self._s3_key(relative_key)
        for attempt in range(3):
            try:
                self.client.upload_file(local_path, self.bucket, s3_key)
                local_size = os.path.getsize(local_path)
                remote_size = self.client.head_object(Bucket=self.bucket, Key=s3_key)["ContentLength"]
                if local_size == remote_size:
                    return True
            except Exception:
                time.sleep(2**attempt)
        return False

    def download(self, relative_key: str, local_path: str) -> bool:
        s3_key = self._s3_key(relative_key)
        for attempt in range(3):
            try:
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, s3_key, local_path)
                local_size = os.path.getsize(local_path)
                remote_size = self.client.head_object(Bucket=self.bucket, Key=s3_key)["ContentLength"]
                if local_size == remote_size:
                    return True
            except Exception:
                time.sleep(2**attempt)
        return False

    def list_objects(self, prefix: str = "") -> list[dict]:
        """List all objects under self.prefix + prefix.

        Returns a list of dicts with keys 'Key' (relative to self.prefix),
        'LastModified' (datetime), and 'Size' (int).
        """
        full_prefix = self._s3_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        results = []
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    rel_key = obj["Key"].removeprefix(self.prefix)
                    results.append({
                        "Key": rel_key,
                        "LastModified": obj["LastModified"],
                        "Size": obj["Size"],
                    })
        except Exception:
            pass
        return results

    def sync_down(self, local_dir: Path, latest_checkpoints_only: bool = True) -> int:
        """Pull files from S3 to local_dir.

        For each file on S3:
        - If it doesn't exist locally, download it.
        - If it exists locally with a different size, prefer the more recent one
          (compare S3 LastModified vs local mtime).

        When latest_checkpoints_only is True (default), for each stage directory
        only the highest-step checkpoint is downloaded instead of all checkpoints.

        Returns the number of files downloaded.
        """
        remote_objects = self.list_objects()

        if latest_checkpoints_only:
            remote_objects = _filter_latest_checkpoints(remote_objects)

        # First pass: figure out what needs downloading
        to_download = []
        for obj in remote_objects:
            rel_key = obj["Key"]
            local_path = local_dir / rel_key
            s3_mtime = obj["LastModified"]
            if s3_mtime.tzinfo is None:
                s3_mtime = s3_mtime.replace(tzinfo=datetime.timezone.utc)

            if not local_path.exists():
                to_download.append(obj)
            else:
                local_size = local_path.stat().st_size
                if local_size != obj["Size"]:
                    local_mtime = datetime.datetime.fromtimestamp(
                        local_path.stat().st_mtime, tz=datetime.timezone.utc
                    )
                    if s3_mtime > local_mtime:
                        to_download.append(obj)

        if not to_download:
            print(f"S3 sync down: local is up to date ({len(remote_objects)} remote file(s) checked)")
            return 0

        print(f"S3 sync down: {len(to_download)} file(s) on remote not on local — downloading before proceeding")
        for obj in to_download:
            print(f"  -> {obj['Key']}")

        # Second pass: download
        downloaded = 0
        for obj in to_download:
            if self.download(obj["Key"], str(local_dir / obj["Key"])):
                downloaded += 1

        print(f"S3 sync down: downloaded {downloaded}/{len(to_download)} file(s)")
        return downloaded

    def sync_up(self, local_dir: Path) -> int:
        """Push files from local_dir to S3.

        For each local file:
        - If it doesn't exist on S3, upload it.
        - If it exists on S3 with a different size, prefer the more recent one
          (compare local mtime vs S3 LastModified).

        Returns the number of files uploaded.
        """
        remote_objects = {obj["Key"]: obj for obj in self.list_objects()}
        uploaded = 0

        if not local_dir.exists():
            return 0

        for local_path in local_dir.rglob("*"):
            if not local_path.is_file():
                continue
            if local_path.suffix == ".tmp":
                continue

            rel_key = str(local_path.relative_to(local_dir))

            should_upload = False

            if rel_key not in remote_objects:
                should_upload = True
            else:
                remote = remote_objects[rel_key]
                local_size = local_path.stat().st_size
                if local_size != remote["Size"]:
                    local_mtime = datetime.datetime.fromtimestamp(
                        local_path.stat().st_mtime, tz=datetime.timezone.utc
                    )
                    s3_mtime = remote["LastModified"]
                    if s3_mtime.tzinfo is None:
                        s3_mtime = s3_mtime.replace(tzinfo=datetime.timezone.utc)
                    if local_mtime > s3_mtime:
                        should_upload = True

            if should_upload:
                if self.upload(str(local_path), rel_key):
                    uploaded += 1

        return uploaded

    def find_latest(self, stage_dir: str) -> str | None:
        final_key = f"{stage_dir}/checkpoint.pth"
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._s3_key(final_key))
            return final_key
        except Exception:
            pass

        best_step, best_key = -1, None
        prefix = self._s3_key(f"{stage_dir}/checkpoint_step-")
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    m = re.search(r"checkpoint_step-(\d+)\.pth$", key)
                    if m:
                        step = int(m.group(1))
                        if step > best_step:
                            best_step = step
                            best_key = key.removeprefix(self.prefix)
        except Exception:
            return None
        return best_key


def _filter_latest_checkpoints(objects: list[dict]) -> list[dict]:
    """Keep only the latest checkpoint per stage directory.

    For each directory containing checkpoint files, keeps:
    - checkpoint.pth (final) if present, OR
    - the highest-step checkpoint_step-N.pth

    Non-checkpoint files are always kept.
    """
    # Group checkpoint files by their parent directory
    checkpoints_by_dir: dict[str, list[dict]] = {}
    non_checkpoints = []

    for obj in objects:
        key = obj["Key"]
        name = Path(key).name
        if name == "checkpoint.pth" or _CHECKPOINT_STEP_RE.match(name):
            parent = str(Path(key).parent)
            checkpoints_by_dir.setdefault(parent, []).append(obj)
        else:
            non_checkpoints.append(obj)

    # For each directory, pick only the best checkpoint
    for ckpts in checkpoints_by_dir.values():
        best = None
        best_step = -1
        for obj in ckpts:
            name = Path(obj["Key"]).name
            if name == "checkpoint.pth":
                best = obj
                best_step = 10**18  # final always wins
            else:
                m = _CHECKPOINT_STEP_RE.match(name)
                if m:
                    step = int(m.group(1))
                    if step > best_step:
                        best = obj
                        best_step = step
        if best is not None:
            non_checkpoints.append(best)

    return non_checkpoints


# --------------------------------------------------------------------------- #
# S3 Watcher — background thread that monitors res_dir for changes            #
# --------------------------------------------------------------------------- #

class S3Watcher:
    """Background thread that periodically syncs local res_dir changes to S3.

    Tracks file mtimes and sizes. On each poll cycle, uploads files that are
    new or have changed since the last scan. Skips .tmp files.
    """

    def __init__(
        self,
        s3: S3MirrorManager,
        res_dir: Path,
        interval: float = 30.0,
    ):
        self.s3 = s3
        self.res_dir = res_dir
        self.interval = interval
        self._known: dict[str, tuple[float, int]] = {}  # rel_key -> (mtime, size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        # Snapshot current state so we don't re-upload everything on first poll
        self._scan_current()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="s3-watcher"
        )
        self._thread.start()
        _logger.info(f"S3 watcher started: {self.res_dir} -> s3://{self.s3.bucket}/{self.s3.prefix} (every {self.interval}s)")

    def stop(self, timeout: float = 60.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        _logger.info("S3 watcher stopped")

    def flush(self) -> int:
        """Force an immediate sync of changed files. Returns count uploaded."""
        try:
            return self._poll()
        except Exception:
            _logger.warning("S3 watcher: flush failed", exc_info=True)
            return 0

    def _scan_current(self) -> None:
        """Record mtime/size of all current files without uploading."""
        if not self.res_dir.exists():
            return
        for fp in self.res_dir.rglob("*"):
            if not fp.is_file() or fp.suffix == ".tmp":
                continue
            rel_key = str(fp.relative_to(self.res_dir))
            try:
                st = fp.stat()
                self._known[rel_key] = (st.st_mtime, st.st_size)
            except OSError:
                pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                uploaded = self._poll()
                if uploaded:
                    _logger.debug(f"S3 watcher: uploaded {uploaded} file(s)")
            except Exception:
                _logger.debug("S3 watcher: poll error", exc_info=True)

    def _poll(self) -> int:
        if not self.res_dir.exists():
            return 0

        uploaded = 0
        for fp in self.res_dir.rglob("*"):
            if not fp.is_file() or fp.suffix == ".tmp":
                continue

            rel_key = str(fp.relative_to(self.res_dir))
            try:
                st = fp.stat()
            except OSError:
                continue

            current = (st.st_mtime, st.st_size)
            prev = self._known.get(rel_key)

            if prev == current:
                continue

            # File is new or changed — upload it
            try:
                if self.s3.upload(str(fp), rel_key):
                    uploaded += 1
            except Exception:
                _logger.debug(f"S3 watcher: failed to upload {rel_key}", exc_info=True)
                continue

            self._known[rel_key] = current

        return uploaded


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #

# Global watcher instance (at most one per process)
_watcher: S3Watcher | None = None
# Set to False by setup_s3 when credentials are invalid; all S3 ops become no-ops.
_s3_available: bool = True


def _get_run_cfg(config: ExperimentConfig | RunConfig) -> RunConfig:
    return config.run if hasattr(config, "run") else config


@lru_cache(maxsize=16)
def get_s3_manager_from_values(
    bucket: str | None,
    prefix: str | None,
    experiment_id: str | None,
) -> S3MirrorManager | None:
    bucket_val = (bucket or os.environ.get("S3_CHECKPOINT_BUCKET", "")).strip()
    if not bucket_val:
        return None

    prefix_val = (prefix if prefix is not None else os.environ.get("S3_CHECKPOINT_PREFIX", "")).strip("/")
    exp_id = (experiment_id or "").strip("/")
    full_prefix = f"{prefix_val}/{exp_id}" if prefix_val and exp_id else (exp_id or prefix_val)
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1").strip()
    return S3MirrorManager(bucket=bucket_val, prefix=full_prefix, region=region)


def get_s3_manager(config: ExperimentConfig | RunConfig) -> S3MirrorManager | None:
    run_cfg = _get_run_cfg(config)
    return get_s3_manager_from_values(
        bucket=run_cfg.s3_bucket,
        prefix=run_cfg.s3_prefix,
        experiment_id=run_cfg.experiment_id,
    )


def sync_from_s3(config: ExperimentConfig | RunConfig) -> None:
    """Pull existing files from S3 to local res_dir on experiment start.

    Only downloads the latest checkpoint per stage directory (not all
    intermediate checkpoints).  Runs on ALL ranks so every node has the
    checkpoint locally before NCCL collectives begin.
    """
    if not _s3_available:
        return

    s3 = get_s3_manager(config)
    if s3 is None:
        return

    run_cfg = _get_run_cfg(config)
    res_dir = run_cfg.res_dir

    print(f"S3 sync down: s3://{s3.bucket}/{s3.prefix} -> {res_dir}")
    try:
        s3.sync_down(res_dir, latest_checkpoints_only=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("WARNING: S3 sync down failed — continuing without S3 download")


def sync_to_s3(config: ExperimentConfig | RunConfig) -> None:
    """Push local res_dir to S3 on experiment completion."""
    if not is_main_process() or not _s3_available:
        return

    s3 = get_s3_manager(config)
    if s3 is None:
        return

    run_cfg = _get_run_cfg(config)
    res_dir = run_cfg.res_dir

    _logger.info(f"S3 sync up: {res_dir} -> s3://{s3.bucket}/{s3.prefix}")
    try:
        uploaded = s3.sync_up(res_dir)
        if uploaded:
            _logger.info(f"S3 sync up: uploaded {uploaded} file(s)")
    except Exception:
        _logger.warning("S3 sync up failed — continuing without S3 upload", exc_info=True)


def start_watcher(config: ExperimentConfig | RunConfig) -> None:
    """Start the background S3 watcher (main process only)."""
    global _watcher

    if not is_main_process() or not _s3_available:
        return

    s3 = get_s3_manager(config)
    if s3 is None:
        return

    if _watcher is not None:
        return

    run_cfg = _get_run_cfg(config)
    try:
        _watcher = S3Watcher(s3, run_cfg.res_dir)
        _watcher.start()
    except Exception:
        _logger.warning("Failed to start S3 watcher — continuing without it", exc_info=True)
        _watcher = None


def stop_watcher() -> None:
    """Stop the background S3 watcher and flush remaining changes."""
    global _watcher

    if _watcher is None:
        return

    _watcher.flush()
    _watcher.stop()
    _watcher = None


def setup_s3(config: ExperimentConfig | RunConfig) -> None:
    """Prime S3 manager and verify credentials are valid."""
    global _s3_available

    s3 = get_s3_manager(config)
    if s3 is None:
        print("S3 mirroring disabled (no bucket configured)")
        _s3_available = False
        return
    try:
        # list_objects_v2 with MaxKeys=0 works regardless of bucket region,
        # unlike head_bucket which returns 400 when the region is wrong.
        s3.client.list_objects_v2(Bucket=s3.bucket, Prefix=s3.prefix, MaxKeys=0)
        print(f"S3 credentials valid — bucket '{s3.bucket}' accessible")
        _s3_available = True
    except Exception as e:
        print(f"WARNING: S3 credentials invalid or bucket '{s3.bucket}' inaccessible: {e}")
        _s3_available = False

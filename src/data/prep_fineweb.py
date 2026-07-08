#!/usr/bin/env python
"""
prep_fineweb.py — Reliable pipeline for fineweb token bins.

Architecture: orchestrator + worker subprocess isolation.
See prep_code.py for detailed rationale.

Orchestrator streams HuggingFaceFW/fineweb-edu (sample-100BT) in chunks.
Each chunk is saved as an Arrow dataset, then a worker subprocess tokenises
it, writes .bin shards, and exits. The orchestrator uploads shards, updates
progress, and moves on.
"""

# ── Env vars BEFORE any HF / tokenizers imports ──────────────────
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/.cache/huggingface/datasets")

import argparse
import gc
import itertools
import json
import logging as pylogging
import signal
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity(40)

try:
    import hf_transfer  # noqa: F401
    _HF_TRANSFER_OK = True
except ImportError:
    _HF_TRANSFER_OK = False

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Constants ────────────────────────────────────────────────────────
LABEL = "fineweb"
PROGRESS_FILE = "progress.json"
WORKER_RESULT_FILE = "worker_result.json"
TMP_DIR = Path("/workspace/tmp/prep_fineweb")

# ── Graceful shutdown ────────────────────────────────────────────────
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"\n[{time.strftime('%H:%M:%S')}] Received {sig_name} — "
          "will stop after current chunk", flush=True)
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── Logging ──────────────────────────────────────────────────────────
_file_logger: pylogging.Logger | None = None


def _ts() -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


def _log(msg: str) -> None:
    line = f"{_ts()} {msg}"
    print(line, flush=True)
    if _file_logger is not None:
        _file_logger.info(msg)


def _setup_logging(out_dir: Path) -> None:
    global _file_logger
    _file_logger = pylogging.getLogger("prep_fineweb")
    _file_logger.setLevel(pylogging.DEBUG)
    _file_logger.handlers.clear()
    fh = pylogging.FileHandler(out_dir / "run.log", mode="a", encoding="utf-8")
    fh.setFormatter(pylogging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _file_logger.addHandler(fh)
    _log(f"Persistent log → {out_dir / 'run.log'}")


def _log_resources(prefix: str = "") -> None:
    p = prefix + "  " if prefix else "  "
    mem = psutil.virtual_memory()
    _log(f"{p}RAM: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB ({mem.percent}%)")
    for mount in ["/", "/workspace"]:
        try:
            du = psutil.disk_usage(mount)
            _log(f"{p}Disk {mount}: {du.used / 1e9:.1f}GB / {du.total / 1e9:.1f}GB "
                 f"({du.percent}%) — {du.free / 1e9:.1f}GB free")
        except Exception:
            pass


def _fmt_tokens(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


# ── Progress tracking ────────────────────────────────────────────────


def _default_progress() -> dict:
    return {
        "stream_examples_consumed": 0,
        "train_tokens": 0,
        "test_tokens": 0,
        "train_shard_idx": 0,
        "test_shard_idx": 0,
        "train_shard_files": [],
        "test_shard_files": [],
    }


def _save_progress(path: Path, progress: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2, default=str)
    tmp.rename(path)
    _log("Progress saved")


def _load_progress(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                prog = json.load(f)
            _log(f"Loaded progress from {path}")
            _log(f"  Stream: {prog['stream_examples_consumed']:,} examples")
            _log(f"  Train: {_fmt_tokens(prog['train_tokens'])} "
                 f"({prog['train_shard_idx']} shards)")
            return prog
        except Exception as e:
            _log(f"WARNING: Could not load progress ({e}), starting fresh")

    _log("Starting fresh")
    return _default_progress()


# ── memmap write ─────────────────────────────────────────────────────


def memmap_write_streaming(
    fname: Path,
    dataset: Dataset,
    ids_column: str = "ids",
    len_column: str = "len",
    dtype: np.dtype = np.uint16,
    batch_size: int = 20_000,
) -> int:
    total_tokens = int(np.sum(dataset[len_column]))
    if total_tokens == 0:
        np.memmap(fname, dtype=dtype, mode="w+", shape=(1,))
        return 0

    mmap = np.memmap(fname, dtype=dtype, mode="w+", shape=(total_tokens,))
    idx = 0
    n = len(dataset)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_ids = dataset[start:end][ids_column]
        flat = np.concatenate(
            [np.asarray(ids, dtype=dtype) for ids in batch_ids])
        mmap[idx : idx + len(flat)] = flat
        idx += len(flat)

    mmap.flush()
    del mmap
    gc.collect()
    return total_tokens


# ── Upload helper ────────────────────────────────────────────────────


def _upload_file(api: HfApi, local_path: Path, repo_id: str, subfolder: str,
                 hf_token: str) -> None:
    remote_path = f"{subfolder}/{local_path.name}"
    for attempt in range(3):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            _log(f"    Uploaded {local_path.name} → {repo_id}/{remote_path}")
            return
        except Exception as e:
            if attempt < 2:
                _log(f"    Upload retry {attempt+1}/3 for {local_path.name}: {e}")
                time.sleep(5 * (attempt + 1))
            else:
                _log(f"    Upload FAILED for {local_path.name}: {e}")
                raise


# ═══════════════════════════════════════════════════════════════════════
# WORKER
# ═══════════════════════════════════════════════════════════════════════


def worker_main():
    """
    Stateless chunk processor.
    Loads Arrow dataset → tokenises → train/test split → writes .bin shards
    → writes result.json → exits.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-length", type=int, default=-1)
    ap.add_argument("--length-strategy", default="none")
    ap.add_argument("--num-proc", type=int, default=32)
    ap.add_argument("--chunk-num", type=int, default=0)
    ap.add_argument("--train-shard-idx", type=int, required=True)
    ap.add_argument("--test-shard-idx", type=int, required=True)
    args = ap.parse_args(sys.argv[2:])

    out_dir = Path(args.out_dir)

    def wlog(msg):
        print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def wlog_resources():
        mem = psutil.virtual_memory()
        wlog(f"RAM: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB ({mem.percent}%)")

    wlog(f"Starting: num_proc={args.num_proc}")
    wlog_resources()

    # Load chunk
    wlog(f"Loading chunk from {args.chunk_dir} ...")
    ds = Dataset.load_from_disk(args.chunk_dir)
    wlog(f"Loaded {len(ds):,} examples")

    # Load tokenizer
    wlog(f"Loading tokenizer {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    assert len(tokenizer) <= np.iinfo(np.uint16).max, (
        f"tokenizer vocab size {len(tokenizer)} exceeds uint16; bump memmap dtype")
    eos_id = tokenizer.eos_token_id
    do_trunc = args.length_strategy == "truncate" and args.max_length > 0

    def tok_batch(examples):
        all_ids, all_lens = [], []
        for text in examples["text"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(eos_id)
            if do_trunc:
                ids = ids[:args.max_length]
                ids[-1] = eos_id
            all_ids.append(ids)
            all_lens.append(len(ids))
        return {"ids": all_ids, "len": all_lens}

    # Tokenise
    wlog(f"Tokenising {len(ds):,} examples ...")
    wlog_resources()
    t0 = time.time()
    ds = ds.map(
        tok_batch,
        batched=True,
        batch_size=2_000,
        num_proc=args.num_proc,
        remove_columns=["text"],
        desc="tok fineweb",
    )
    wlog(f"Tokenised in {time.time() - t0:.0f}s")
    wlog_resources()

    # Drop long examples
    if args.length_strategy == "drop" and args.max_length > 0:
        before = len(ds)
        ds = ds.filter(lambda ex: ex["len"] <= args.max_length)
        if len(ds) < before:
            wlog(f"Dropped {before - len(ds):,} long examples")

    vocab_size = len(tokenizer)

    result = {
        "train_tokens": 0, "test_tokens": 0,
        "train_shard_files": [], "test_shard_files": [],
        "vocab_size": vocab_size,
        "train_example": "", "test_example": "",
    }

    if len(ds) == 0:
        with open(out_dir / WORKER_RESULT_FILE, "w") as f:
            json.dump(result, f)
        wlog("No examples, exiting")
        return

    # Train / test split (98/2)
    splits = ds.train_test_split(test_size=0.02, seed=42 + args.chunk_num)
    train_ds = splits["train"]
    test_ds  = splits["test"]
    del ds, splits
    gc.collect()

    # Write shards
    for split_name, split_ds, shard_idx in [
        ("train", train_ds, args.train_shard_idx),
        ("test",  test_ds,  args.test_shard_idx),
    ]:
        if len(split_ds) == 0:
            continue

        shard_fname = f"{LABEL}_{split_name}_{shard_idx:03d}.bin"
        shard_path = out_dir / shard_fname

        if shard_path.exists():
            os.remove(shard_path)

        tokens = memmap_write_streaming(shard_path, split_ds)
        result[f"{split_name}_tokens"] = tokens
        result[f"{split_name}_shard_files"].append(shard_fname)

        example_ids = split_ds[-1]["ids"]
        result[f"{split_name}_example"] = tokenizer.decode(
            example_ids, skip_special_tokens=False)

        size_mb = shard_path.stat().st_size / 1e6
        wlog(f"Wrote {shard_fname}: {_fmt_tokens(tokens)} tokens "
             f"({len(split_ds):,} ex, {size_mb:.0f}MB)")

    del train_ds, test_ds
    gc.collect()

    with open(out_dir / WORKER_RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    wlog_resources()
    wlog("Done, exiting")


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════


def _run_worker(cmd: list, log_fn, timeout: int = 1800) -> int:
    """Spawn worker subprocess with watchdog timeout. Returns exit code."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )

    proc_done = threading.Event()
    timed_out = threading.Event()

    def _watchdog():
        if not proc_done.wait(timeout):
            timed_out.set()
            log_fn(f"  TIMEOUT after {timeout}s — killing worker (pid {proc.pid})")
            try:
                proc.kill()
            except Exception:
                pass

    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()

    last_output = time.time()
    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                log_fn(f"  {line}")
                last_output = time.time()
            if proc.poll() is not None:
                break
            if time.time() - last_output > 300:
                log_fn(f"  WARNING: worker silent for "
                       f"{time.time() - last_output:.0f}s")
    except Exception as e:
        log_fn(f"  ERROR reading worker output: {e}")
        try:
            proc.kill()
        except Exception:
            pass

    proc_done.set()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)

    return -9 if timed_out.is_set() else proc.returncode


def orchestrate(
    out_dir: Path,
    num_proc: int,
    max_length: int,
    length_strategy: str,
    tokenizer_name: str,
    upload: bool,
    token_budget: int,
    chunk_size: int,
) -> None:
    wall_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    _setup_logging(out_dir)

    _log("=" * 70)
    _log("prep_fineweb.py — orchestrator starting")
    _log("=" * 70)
    _log(f"  Python: {sys.executable}")
    _log(f"  Script: {Path(__file__).resolve()}")
    _log(f"  out_dir: {out_dir}")
    if _HF_TRANSFER_OK:
        _log("  hf_transfer: active")
    _log(f"  Worker config: num_proc={num_proc}, chunk_size={chunk_size:,}")
    _log(f"  Budget: {_fmt_tokens(token_budget)} train tokens")
    _log_resources()

    # HF setup
    hf_token = os.getenv("HF_TOKEN", None)
    repo_id = "AE-data/modular-pretraining"
    subfolder = "fineweb"
    hf_api = None

    if upload and hf_token:
        hf_api = HfApi(token=hf_token)
        try:
            hf_api.create_repo(
                repo_id=repo_id, token=hf_token,
                exist_ok=True, repo_type="dataset")
            _log(f"  HF repo {repo_id} ready")
        except Exception as e:
            _log(f"  Note: Could not verify repo: {e}")

    # Load progress
    progress_path = out_dir / PROGRESS_FILE
    progress = _load_progress(progress_path)

    # Open stream
    _log("Opening streaming iterator for HuggingFaceFW/fineweb-edu (sample-100BT) ...")
    ds_stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-100BT",
        split="train", streaming=True)

    skip_count = progress["stream_examples_consumed"]
    if skip_count > 0:
        _log(f"Skipping {skip_count:,} already-processed examples ...")
        ds_stream = ds_stream.skip(skip_count)

    iterator = iter(ds_stream)
    chunk_num = 0
    processing_start = time.time()

    # ── Main chunk loop ──────────────────────────────────────────────
    while progress["train_tokens"] < token_budget:
        if _shutdown_requested:
            _log("Shutdown requested, saving progress ...")
            _save_progress(progress_path, progress)
            return

        try:
            du = psutil.disk_usage("/")
            if du.percent > 95:
                _log("CRITICAL: Overlay disk > 95%! Saving and aborting.")
                _save_progress(progress_path, progress)
                return
        except Exception:
            pass

        chunk_num += 1
        _log(f"\n=== Chunk {chunk_num} "
             f"(consumed {progress['stream_examples_consumed']:,} examples) ===")
        _log(f"  {LABEL}: {_fmt_tokens(progress['train_tokens'])}"
             f"/{_fmt_tokens(token_budget)} "
             f"({progress['train_shard_idx']} shards)")
        _log_resources()

        # ── Consume chunk from stream ────────────────────────────────
        t0 = time.time()
        raw_chunk = []
        for ex in itertools.islice(iterator, chunk_size):
            raw_chunk.append({"text": ex["text"]})

        if not raw_chunk:
            _log("Stream exhausted before budget met!")
            break

        chunk_len = len(raw_chunk)
        _log(f"  Materialised {chunk_len:,} examples in {time.time() - t0:.0f}s")

        # ── Save chunk as Arrow dataset ──────────────────────────────
        chunk_dir = TMP_DIR / f"chunk_{chunk_num}"
        ds_chunk = Dataset.from_list(raw_chunk)
        ds_chunk.save_to_disk(str(chunk_dir))
        chunk_mb = sum(
            f.stat().st_size for f in chunk_dir.rglob("*") if f.is_file()
        ) / 1e6
        _log(f"  Saved Arrow dataset: {chunk_mb:.0f}MB")
        del raw_chunk, ds_chunk
        gc.collect()

        # ── Spawn worker subprocess ──────────────────────────────────
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker",
            "--chunk-dir",       str(chunk_dir),
            "--out-dir",         str(out_dir),
            "--tokenizer",       tokenizer_name,
            "--max-length",      str(max_length),
            "--length-strategy", length_strategy,
            "--num-proc",        str(num_proc),
            "--chunk-num",       str(chunk_num),
            "--train-shard-idx", str(progress["train_shard_idx"]),
            "--test-shard-idx",  str(progress["test_shard_idx"]),
        ]

        _log(f"  Spawning worker subprocess (num_proc={num_proc}) ...")
        worker_ok = False

        for attempt in range(3):
            if attempt > 0:
                retry_nproc = max(1, num_proc // (2 ** attempt))
                cmd_retry = list(cmd)
                np_idx = cmd.index("--num-proc") + 1
                cmd_retry[np_idx] = str(retry_nproc)
                _log(f"  Retry {attempt}/3 (num_proc reduced to {retry_nproc}) ...")
                time.sleep(10)
            else:
                cmd_retry = cmd

            returncode = _run_worker(cmd_retry, _log, timeout=1800)
            if returncode == 0:
                worker_ok = True
                break
            else:
                _log(f"  Worker exited with code {returncode}")

        if not worker_ok:
            _log("CRITICAL: Worker failed 3 times. Saving progress and aborting.")
            _save_progress(progress_path, progress)
            shutil.rmtree(chunk_dir, ignore_errors=True)
            return

        # ── Read worker result ───────────────────────────────────────
        result_path = out_dir / WORKER_RESULT_FILE
        try:
            with open(result_path) as f:
                wr = json.load(f)
        except Exception as e:
            _log(f"ERROR: Could not read worker result: {e}")
            _save_progress(progress_path, progress)
            shutil.rmtree(chunk_dir, ignore_errors=True)
            return

        # ── Upload shards + update progress ──────────────────────────
        all_shard_files = wr.get("train_shard_files", []) + wr.get("test_shard_files", [])

        if upload and hf_api and all_shard_files:
            _log(f"  Uploading {len(all_shard_files)} shard(s) ...")
            for sf in all_shard_files:
                _upload_file(hf_api, out_dir / sf, repo_id, subfolder, hf_token)

        progress["stream_examples_consumed"] += chunk_len
        progress["train_tokens"] += wr.get("train_tokens", 0)
        progress["test_tokens"]  += wr.get("test_tokens", 0)
        progress["train_shard_files"].extend(wr.get("train_shard_files", []))
        progress["test_shard_files"].extend(wr.get("test_shard_files", []))
        progress["train_shard_idx"] += len(wr.get("train_shard_files", []))
        progress["test_shard_idx"]  += len(wr.get("test_shard_files", []))

        if wr.get("vocab_size"):
            progress["vocab_size"] = wr["vocab_size"]
        if wr.get("train_example"):
            progress["train_example"] = wr["train_example"]
        if wr.get("test_example"):
            progress["test_example"] = wr["test_example"]

        _save_progress(progress_path, progress)

        # ── Cleanup ──────────────────────────────────────────────────
        shutil.rmtree(chunk_dir, ignore_errors=True)
        result_path.unlink(missing_ok=True)

        # ── Throughput & ETA ─────────────────────────────────────────
        chunk_elapsed = time.time() - t0
        processing_elapsed = time.time() - processing_start
        remaining = max(0, token_budget - progress["train_tokens"])

        if processing_elapsed > 0:
            tok_per_sec = progress["train_tokens"] / processing_elapsed
            eta_sec = remaining / tok_per_sec if tok_per_sec > 0 else 0
            eta_str = (f"{eta_sec/3600:.1f}h" if eta_sec >= 3600
                       else f"{eta_sec/60:.0f}m")
            _log(f"  Chunk {chunk_num} done in {chunk_elapsed:.0f}s  |  "
                 f"throughput: {tok_per_sec/1e6:.1f}M tok/s  |  "
                 f"remaining: {_fmt_tokens(remaining)}  |  ETA: {eta_str}")
        else:
            _log(f"  Chunk {chunk_num} done in {chunk_elapsed:.0f}s")

    # ── Write metadata.json ──────────────────────────────────────────
    _log("\nWriting metadata.json ...")
    meta = {
        LABEL: {
            "train": {
                "total_tokens": progress["train_tokens"],
                "num_shards":   progress["train_shard_idx"],
                "shard_files":  progress["train_shard_files"],
                "example":      progress.get("train_example", ""),
            },
            "test": {
                "total_tokens": progress["test_tokens"],
                "num_shards":   progress["test_shard_idx"],
                "shard_files":  progress["test_shard_files"],
                "example":      progress.get("test_example", ""),
            },
        },
        "all": {
            "total_tokens_train": progress["train_tokens"],
            "total_tokens_test":  progress["test_tokens"],
            "tokenizer": tokenizer_name,
            "vocab_size": progress.get("vocab_size", 0),
            "labels": [LABEL],
            "token_budget": token_budget,
            "total_examples_consumed": progress["stream_examples_consumed"],
        },
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    if upload and hf_api:
        _upload_file(hf_api, meta_path, repo_id, subfolder, hf_token)

    if progress["train_tokens"] >= token_budget:
        progress_path.unlink(missing_ok=True)
        _log("Budget met — progress file removed")

    elapsed = time.time() - wall_start
    _log(f"\nDone! Wall time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    _log(f"  {LABEL}: train={_fmt_tokens(progress['train_tokens'])} "
         f"({progress['train_shard_idx']} shards)  "
         f"test={_fmt_tokens(progress['test_tokens'])} "
         f"({progress['test_shard_idx']} shards)")
    _log_resources()


# ═══════════════════════════════════════════════════════════════════════
# Download mode
# ═══════════════════════════════════════════════════════════════════════


def download_bins(out_dir: Path, limit: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN") or None
    repo_id = "AE-data/modular-pretraining"
    subfolder = "fineweb"

    _log(f"Downloading files from {repo_id}/{subfolder} ...")
    api = HfApi(token=hf_token)

    repo_files = api.list_repo_files(
        repo_id=repo_id, repo_type="dataset", token=hf_token)
    all_files = [
        f for f in repo_files
        if f.startswith(subfolder)
        and (f.endswith(".bin") or f.endswith("metadata.json"))
    ]

    if not all_files:
        _log(f"No files found in {repo_id}/{subfolder}")
        return

    if limit is not None:
        train_bins = [f for f in all_files if "_train_" in f and f.endswith(".bin")][:limit]
        test_bins = [f for f in all_files if "_test_" in f and f.endswith(".bin")][:limit]
        meta_files = [f for f in all_files if f.endswith("metadata.json")]
        all_files = train_bins + test_bins + meta_files

    _log(f"Found {len(all_files)} files")

    def _dl(fp):
        hf_hub_download(
            repo_id=repo_id, filename=fp,
            repo_type="dataset", token=hf_token,
            local_dir=out_dir.parent, local_dir_use_symlinks=False)
        return f"{fp} → {out_dir / Path(fp).name}"

    with ThreadPoolExecutor(max_workers=min(8, len(all_files))) as pool:
        futs = {pool.submit(_dl, fp): fp for fp in all_files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
            _log(f"  {fut.result()}")

    _log("Download complete!")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main()
        sys.exit(0)

    ap = argparse.ArgumentParser(
        description="Prepare fineweb token bins from HuggingFaceFW/fineweb-edu")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: ./fineweb)")
    ap.add_argument("--num_proc", type=int, default=32,
                    help="Workers for tokenisation per subprocess (default 32)")
    ap.add_argument("--max_length", type=int, default=-1)
    ap.add_argument("--length_strategy", default="none",
                    choices=["truncate", "drop", "none"])
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neo-125M")
    ap.add_argument("--download_bins", nargs="?", type=int, const=-1, default=None,
                    help="Download existing bins from HF instead of generating. "
                         "Optionally specify how many bin files to download.")
    ap.add_argument("--upload_bins", action="store_true",
                    help="Upload each shard to HF Hub after writing")
    ap.add_argument("--token_budget", type=int, default=100_000_000_000,
                    help="Train token budget (default 100B)")
    ap.add_argument("--chunk_size", type=int, default=100_000,
                    help="Examples per chunk (default 100k)")
    args = ap.parse_args()

    default_out_dir = Path(__file__).parent / "fineweb"
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.resolve() != default_out_dir.resolve():
        if default_out_dir.is_symlink() or default_out_dir.exists():
            default_out_dir.unlink() if default_out_dir.is_symlink() else shutil.rmtree(default_out_dir)
        default_out_dir.parent.mkdir(parents=True, exist_ok=True)
        default_out_dir.symlink_to(out_dir.resolve())
        print(f"Symlinked {default_out_dir} -> {out_dir.resolve()}")

    if args.download_bins is not None:
        limit = None if args.download_bins == -1 else args.download_bins
        download_bins(out_dir, limit=limit)
        sys.exit(0)

    orchestrate(
        out_dir=out_dir,
        num_proc=args.num_proc,
        max_length=args.max_length,
        length_strategy=args.length_strategy,
        tokenizer_name=args.tokenizer,
        upload=args.upload_bins,
        token_budget=args.token_budget,
        chunk_size=args.chunk_size,
    )

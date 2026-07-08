#!/usr/bin/env python
"""
prep_code.py — Reliable pipeline for code-lisp / code-other token bins.

Architecture
============
Orchestrator (default mode)
  Streams bigcode/the-stack-dedup in chunks. For each chunk:
    1. Materialises chunk to a temp Arrow dataset on /workspace
    2. Spawns a **worker subprocess** that splits by language,
       tokenises, and writes .bin shards
    3. Streams worker stdout in real-time to the log (watchdog kills on timeout)
    4. Reads worker result JSON, uploads shards to HF Hub
    5. Updates progress.json, deletes temp files

Worker (--worker mode)
  Stateless process — called by the orchestrator:
    1. Loads chunk from Arrow dataset
    2. Splits into Lisp (Common Lisp + Emacs Lisp) / other
    3. Tokenises each subset
    4. Writes .bin shard files
    5. Writes result.json with token counts
    6. Exits — OS reclaims all memory

Why subprocess isolation?
  • Hard guarantee against memory leaks / accumulation across chunks
  • Clean failure: subprocess crash → orchestrator retries, no corrupt state
  • Resource usage during tokenisation is logged from *inside* the worker
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
from typing import Any, Dict

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
LABEL_LISP  = "code-lisp"
LABEL_OTHER  = "code-other"
LISP_LANGUAGES = {"Common Lisp", "Emacs Lisp"}  # values in the-stack-dedup's "lang" column
PROGRESS_FILE = "progress.json"
WORKER_RESULT_FILE = "worker_result.json"
TMP_DIR = Path("/workspace/tmp/prep_code")

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
    _file_logger = pylogging.getLogger("prep_code")
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
        "lisp_examples_consumed": 0,
        "labels": {
            label: {
                "train_tokens": 0,
                "test_tokens": 0,
                "train_shard_idx": 0,
                "test_shard_idx": 0,
                "train_shard_files": [],
                "test_shard_files": [],
            }
            for label in [LABEL_LISP, LABEL_OTHER]
        },
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
            prog.setdefault("lisp_examples_consumed", 0)
            _log(f"Loaded progress from {path}")
            _log(f"  Stream position: {prog['stream_examples_consumed']:,} examples")
            for label in [LABEL_LISP, LABEL_OTHER]:
                lp = prog["labels"][label]
                _log(f"  {label}: {_fmt_tokens(lp['train_tokens'])} train, "
                     f"{lp['train_shard_idx']} shards")
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
    batch_size: int = 20_000, #in terms of tokens
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
        # Concatenate entire batch into one array, write in a single memmap slice
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
# WORKER — runs in a separate subprocess, processes one chunk
# ═══════════════════════════════════════════════════════════════════════


def worker_main():
    """
    Stateless chunk processor.
    Loads Arrow dataset → splits by language → tokenises → writes .bin shards
    → writes result.json → exits.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True,
                    help="Path to Arrow dataset directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-length", type=int, default=-1)
    ap.add_argument("--length-strategy", default="none")
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--chunk-num", type=int, default=0)
    ap.add_argument("--shard-indices", required=True,
                    help="JSON dict: {label: {train: idx, test: idx}}")
    ap.add_argument("--active-labels", required=True,
                    help="JSON list of labels that still need tokens")
    args = ap.parse_args(sys.argv[2:])  # skip [script, "--worker"]

    out_dir = Path(args.out_dir)
    shard_indices = json.loads(args.shard_indices)
    active_labels = json.loads(args.active_labels)

    def wlog(msg):
        print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def wlog_resources():
        mem = psutil.virtual_memory()
        wlog(f"RAM: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB ({mem.percent}%)")

    wlog(f"Starting: active_labels={active_labels}, num_proc={args.num_proc}")
    wlog_resources()

    # Load chunk from Arrow dataset
    wlog(f"Loading chunk from {args.chunk_dir} ...")
    ds_chunk = Dataset.load_from_disk(args.chunk_dir)
    wlog(f"Loaded {len(ds_chunk):,} examples")

    # Split by language (vectorised via numpy)
    lang_arr = np.array(ds_chunk["lang"])
    lisp_mask = np.isin(lang_arr, list(LISP_LANGUAGES))
    lisp_idx  = np.where(lisp_mask)[0].tolist()
    other_idx = np.where(~lisp_mask)[0].tolist()
    del lang_arr, lisp_mask

    wlog(f"Language split: Lisp={len(lisp_idx):,}  Other={len(other_idx):,}")
    wlog_resources()

    # Build per-label subsets (remove lang column, keep only content)
    label_subsets: Dict[str, Dataset] = {}
    for label, indices in [(LABEL_LISP, lisp_idx), (LABEL_OTHER, other_idx)]:
        if label in active_labels and indices:
            label_subsets[label] = ds_chunk.select(indices).remove_columns(["lang"])
    del ds_chunk, lisp_idx, other_idx
    gc.collect()

    # Load tokenizer
    wlog(f"Loading tokenizer {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tokenizer.eos_token_id
    do_trunc = args.length_strategy == "truncate" and args.max_length > 0

    def tok_batch(examples):
        all_ids, all_lens = [], []
        for text in examples["content"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(eos_id)
            if do_trunc:
                ids = ids[:args.max_length]
                ids[-1] = eos_id
            all_ids.append(ids)
            all_lens.append(len(ids))
        return {"ids": all_ids, "len": all_lens}

    # Process each label
    result: Dict[str, Any] = {}

    for label in [LABEL_LISP, LABEL_OTHER]:
        if label not in label_subsets:
            result[label] = {
                "train_tokens": 0, "test_tokens": 0,
                "train_shard_files": [], "test_shard_files": [],
            }
            continue

        ds = label_subsets.pop(label)
        wlog(f"Tokenising {label}: {len(ds):,} examples ...")
        wlog_resources()

        t0 = time.time()
        ds = ds.map(
            tok_batch,
            batched=True,
            batch_size=2_000,
            num_proc=args.num_proc,
            remove_columns=["content"],
            desc=f"tok {label}",
        )
        wlog(f"Tokenised {label} in {time.time() - t0:.0f}s")
        wlog_resources()

        # Drop long examples
        if args.length_strategy == "drop" and args.max_length > 0:
            before = len(ds)
            ds = ds.filter(lambda ex: ex["len"] <= args.max_length)
            if len(ds) < before:
                wlog(f"Dropped {before - len(ds):,} long examples from {label}")

        label_result = {
            "train_tokens": 0, "test_tokens": 0,
            "train_shard_files": [], "test_shard_files": [],
        }

        if len(ds) == 0:
            result[label] = label_result
            continue

        # Train / test split (99/1)
        splits = ds.train_test_split(test_size=0.01, seed=42 + args.chunk_num)
        train_ds = splits["train"]
        test_ds  = splits["test"]
        del ds, splits
        gc.collect()

        # Write shards
        si = shard_indices[label]
        for split_name, split_ds, shard_idx in [
            ("train", train_ds, si["train"]),
            ("test",  test_ds,  si["test"]),
        ]:
            if len(split_ds) == 0:
                continue

            shard_fname = f"{label}_{split_name}_{shard_idx:03d}.bin"
            shard_path = out_dir / shard_fname

            if shard_path.exists():
                os.remove(shard_path)

            tokens = memmap_write_streaming(shard_path, split_ds)
            label_result[f"{split_name}_tokens"] = tokens
            label_result[f"{split_name}_shard_files"].append(shard_fname)

            size_mb = shard_path.stat().st_size / 1e6
            wlog(f"Wrote {shard_fname}: {_fmt_tokens(tokens)} tokens "
                 f"({len(split_ds):,} ex, {size_mb:.0f}MB)")

        del train_ds, test_ds
        gc.collect()
        result[label] = label_result

    # Write result JSON
    with open(out_dir / WORKER_RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    wlog_resources()
    wlog("Done, exiting")


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — streams data, spawns workers, tracks progress
# ═══════════════════════════════════════════════════════════════════════


def _run_worker(cmd: list, log_fn, timeout: int = 1800) -> int:
    """
    Spawn a worker subprocess, stream its stdout line-by-line to log_fn.

    Uses a watchdog thread to enforce the timeout — even if the worker
    hangs without producing any output, the watchdog will kill it after
    `timeout` seconds.  Returns the process exit code (-9 on timeout).
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Watchdog: kills the process if it exceeds the timeout, regardless of
    # whether readline is blocked waiting for output.
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

    proc_done.set()  # tell watchdog to stop

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)

    if timed_out.is_set():
        return -9
    return proc.returncode


def _open_lisp_streams():
    """Load each lisp language directly from the-stack-dedup using data_dir.

    Returns a list of iterators, one per lisp language.  This is MUCH faster
    than streaming the full alphabetically-sorted dataset when only lisp data
    is needed, because it goes straight to the relevant parquet files.
    """
    lisp_iters = []
    for lang in sorted(LISP_LANGUAGES):
        dirname = lang.lower().replace(" ", "-")
        _log(f"  Loading {lang} (data/{dirname}) ...")
        try:
            ds = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir=f"data/{dirname}",
                split="train",
                streaming=True,
            )
            lisp_iters.append(iter(ds))
            _log(f"    {lang} stream ready")
        except Exception as e:
            _log(f"    WARNING: Could not load {lang}: {e}")
    return lisp_iters


def orchestrate(
    out_dir: Path,
    num_proc: int,
    max_length: int,
    length_strategy: str,
    tokenizer_name: str,
    upload: bool,
    lisp_tokens: int,
    other_tokens: int,
    chunk_size: int,
) -> None:
    """Stream the dataset, spawn worker subprocesses per chunk, track progress."""

    wall_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up stale temp data from previous (crashed) runs
    for stale in sorted(TMP_DIR.glob("chunk_*")):
        shutil.rmtree(stale, ignore_errors=True)

    _setup_logging(out_dir)

    _log("=" * 70)
    _log("prep_code.py — orchestrator starting")
    _log("=" * 70)
    _log(f"  Python: {sys.executable}")
    _log(f"  Script: {Path(__file__).resolve()}")
    _log(f"  out_dir: {out_dir}")
    _log(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    _log(f"  HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', 'not set')}")
    if _HF_TRANSFER_OK:
        _log("  hf_transfer: active")
    _log(f"  Worker config: num_proc={num_proc}, chunk_size={chunk_size:,}")
    _log(f"  Budgets: {LABEL_LISP}={_fmt_tokens(lisp_tokens)}, "
         f"{LABEL_OTHER}={_fmt_tokens(other_tokens)}")
    _log_resources()

    # HF setup
    hf_token = os.getenv("HF_TOKEN") or None
    repo_id = "AE-data/modular-pretraining"
    subfolder = "code"
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

    budgets = {LABEL_LISP: lisp_tokens, LABEL_OTHER: other_tokens}

    # ── Choose stream strategy ────────────────────────────────────────
    # If code-other is already done (from a previous run), streaming the
    # full alphabetically-sorted dataset and skipping 31M+ examples is
    # extremely slow.  Instead, load ONLY the lisp language subsets
    # directly -- goes straight to the relevant parquet files on HF Hub.
    labels_data = progress["labels"]
    lisp_at_budget = labels_data[LABEL_LISP]["train_tokens"] >= budgets[LABEL_LISP]
    other_at_budget = labels_data[LABEL_OTHER]["train_tokens"] >= budgets[LABEL_OTHER]

    targeted_lisp_mode = False

    if lisp_at_budget and other_at_budget:
        _log("Both budgets already met -- nothing to do!")
        iterator = iter([])
    elif other_at_budget and not lisp_at_budget:
        _log("=" * 50)
        _log("code-other already at budget -- TARGETED LISP MODE")
        _log("=" * 50)
        _log("Loading lisp language subsets directly (skips streaming "
             "through millions of irrelevant examples) ...")
        lisp_iters = _open_lisp_streams()
        if not lisp_iters:
            _log("ERROR: Could not load any lisp language subset -- aborting")
            return
        iterator = itertools.chain(*lisp_iters)
        skip_count = progress["lisp_examples_consumed"]
        if skip_count > 0:
            _log(f"Skipping {skip_count:,} already-processed lisp examples ...")
            iterator = itertools.islice(iterator, skip_count, None)
        targeted_lisp_mode = True
        _log("Targeted lisp loading ready")
    else:
        _log("Opening streaming iterator for bigcode/the-stack-dedup ...")
        ds_stream = load_dataset(
            "bigcode/the-stack-dedup", split="train", streaming=True)

        skip_count = progress["stream_examples_consumed"]
        if skip_count > 0:
            _log(f"Skipping {skip_count:,} already-processed examples ...")
            ds_stream = ds_stream.skip(skip_count)

        iterator = iter(ds_stream)

    chunk_num = 0
    processing_start = time.time()
    session_tokens_start = sum(
        labels_data[l]["train_tokens"] for l in [LABEL_LISP, LABEL_OTHER])

    # ── Main chunk loop ──────────────────────────────────────────────
    while True:
        # Check if both budgets are met
        labels_data = progress["labels"]
        lisp_done = labels_data[LABEL_LISP]["train_tokens"] >= budgets[LABEL_LISP]
        ot_done = labels_data[LABEL_OTHER]["train_tokens"]  >= budgets[LABEL_OTHER]
        if lisp_done and ot_done:
            _log("Both token budgets met!")
            break

        if _shutdown_requested:
            _log("Shutdown requested, saving progress ...")
            _save_progress(progress_path, progress)
            return

        # Check disk
        try:
            du = psutil.disk_usage("/")
            if du.percent > 95:
                _log("CRITICAL: Overlay disk > 95%! Saving and aborting.")
                _save_progress(progress_path, progress)
                return
        except Exception:
            pass

        # Which labels still need tokens?
        active_labels = [
            l for l in [LABEL_LISP, LABEL_OTHER]
            if labels_data[l]["train_tokens"] < budgets[l]
        ]

        # When only lisp is active, discard non-lisp examples immediately
        # to avoid expensive Arrow serialisation + worker spawn for nothing.
        if set(active_labels) == {LABEL_LISP}:
            keep_languages = LISP_LANGUAGES
        else:
            keep_languages = None  # keep everything

        chunk_num += 1
        _log(f"\n=== Chunk {chunk_num} ===")
        for l in [LABEL_LISP, LABEL_OTHER]:
            lp = labels_data[l]
            status = "DONE" if lp["train_tokens"] >= budgets[l] else "active"
            _log(f"  {l}: {_fmt_tokens(lp['train_tokens'])}/{_fmt_tokens(budgets[l])} "
                 f"({lp['train_shard_idx']} shards) [{status}]")
        _log_resources()

        # ── Consume chunk from stream ────────────────────────────────
        t0 = time.time()
        raw_chunk = []
        lang_counts: Dict[str, int] = {}
        consumed = 0

        for ex in itertools.islice(iterator, chunk_size):
            consumed += 1
            lang = ex["lang"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if keep_languages is None or lang in keep_languages:
                raw_chunk.append({"content": ex["content"], "lang": lang})

        if consumed == 0:
            if targeted_lisp_mode:
                _log("All lisp language streams exhausted")
            else:
                _log("Stream exhausted before budgets met!")
            break

        cursor_key = "lisp_examples_consumed" if targeted_lisp_mode else "stream_examples_consumed"
        mat_time = time.time() - t0

        # Show languages in this chunk
        top_langs = sorted(lang_counts.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{l}={c:,}" for l, c in top_langs)
        _log(f"  Consumed {consumed:,} examples, kept {len(raw_chunk):,} "
             f"in {mat_time:.0f}s")
        _log(f"  Languages (top 5): {top_str}")

        # ── Fast-skip: no relevant data in this chunk ────────────────
        if not raw_chunk:
            _log("  No data for active labels -- skipping chunk")
            progress[cursor_key] += consumed
            _save_progress(progress_path, progress)
            del lang_counts
            gc.collect()
            continue

        # ── Save chunk as Arrow dataset ──────────────────────────────
        chunk_dir = TMP_DIR / f"chunk_{chunk_num}"
        ds_chunk = Dataset.from_list(raw_chunk)
        ds_chunk.save_to_disk(str(chunk_dir))
        chunk_mb = sum(
            f.stat().st_size for f in chunk_dir.rglob("*") if f.is_file()
        ) / 1e6
        _log(f"  Saved Arrow dataset: {chunk_dir} ({chunk_mb:.0f}MB)")
        del raw_chunk, lang_counts, ds_chunk
        gc.collect()

        # ── Build shard indices for the worker ───────────────────────
        shard_indices = {}
        for label in [LABEL_LISP, LABEL_OTHER]:
            lp = labels_data[label]
            shard_indices[label] = {
                "train": lp["train_shard_idx"],
                "test":  lp["test_shard_idx"],
            }

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
            "--shard-indices",   json.dumps(shard_indices),
            "--active-labels",   json.dumps(active_labels),
        ]

        _log(f"  Spawning worker subprocess (num_proc={num_proc}) ...")
        worker_ok = False

        for attempt in range(3):
            if attempt > 0:
                # On retry, halve num_proc to reduce memory pressure
                retry_nproc = max(1, num_proc // (2 ** attempt))
                cmd_retry = [
                    c if c != str(num_proc) or cmd[cmd.index(c) - 1] != "--num-proc"
                    else str(retry_nproc)
                    for c in cmd
                ]
                # Simpler: just rebuild the --num-proc arg
                np_idx = cmd.index("--num-proc") + 1
                cmd_retry = list(cmd)
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
        progress[cursor_key] += consumed
        all_shard_files = []
        for label in [LABEL_LISP, LABEL_OTHER]:
            lr = wr.get(label, {})
            lp = labels_data[label]

            train_files = lr.get("train_shard_files", [])
            test_files  = lr.get("test_shard_files", [])

            lp["train_tokens"] += lr.get("train_tokens", 0)
            lp["test_tokens"]  += lr.get("test_tokens", 0)
            lp["train_shard_files"].extend(train_files)
            lp["test_shard_files"].extend(test_files)
            lp["train_shard_idx"] += len(train_files)
            lp["test_shard_idx"]  += len(test_files)

            all_shard_files.extend(train_files)
            all_shard_files.extend(test_files)

        if upload and hf_api and all_shard_files:
            _log(f"  Uploading {len(all_shard_files)} shard(s) ...")
            for sf in all_shard_files:
                _upload_file(hf_api, out_dir / sf, repo_id, subfolder, hf_token)

        _save_progress(progress_path, progress)

        # ── Cleanup ──────────────────────────────────────────────────
        shutil.rmtree(chunk_dir, ignore_errors=True)
        result_path.unlink(missing_ok=True)

        # ── Throughput & ETA ─────────────────────────────────────────
        chunk_elapsed = time.time() - t0
        processing_elapsed = time.time() - processing_start
        total_train_tok = sum(
            labels_data[l]["train_tokens"] for l in [LABEL_LISP, LABEL_OTHER])
        remaining_tok = sum(
            max(0, budgets[l] - labels_data[l]["train_tokens"])
            for l in [LABEL_LISP, LABEL_OTHER])

        # Use tokens produced THIS session for accurate throughput
        session_tokens = total_train_tok - session_tokens_start
        if processing_elapsed > 0 and session_tokens > 0:
            tok_per_sec = session_tokens / processing_elapsed
            eta_sec = remaining_tok / tok_per_sec if tok_per_sec > 0 else 0
            eta_str = (f"{eta_sec/3600:.1f}h" if eta_sec >= 3600
                       else f"{eta_sec/60:.0f}m")
            _log(f"  Chunk {chunk_num} done in {chunk_elapsed:.0f}s  |  "
                 f"throughput: {tok_per_sec/1e6:.1f}M tok/s  |  "
                 f"remaining: {_fmt_tokens(remaining_tok)}  |  "
                 f"ETA: {eta_str}")
        else:
            _log(f"  Chunk {chunk_num} done in {chunk_elapsed:.0f}s")

    # ── Write metadata.json ──────────────────────────────────────────
    _log("\nWriting metadata.json ...")
    meta: Dict[str, Any] = {}
    labels_data = progress["labels"]

    for label in [LABEL_LISP, LABEL_OTHER]:
        lp = labels_data[label]
        meta[label] = {
            "train": {
                "total_tokens": lp["train_tokens"],
                "num_shards":   lp["train_shard_idx"],
                "shard_files":  lp["train_shard_files"],
            },
            "test": {
                "total_tokens": lp["test_tokens"],
                "num_shards":   lp["test_shard_idx"],
                "shard_files":  lp["test_shard_files"],
            },
        }

    meta["all"] = {
        "total_tokens_train": sum(
            labels_data[l]["train_tokens"] for l in labels_data),
        "total_tokens_test": sum(
            labels_data[l]["test_tokens"] for l in labels_data),
        "tokenizer": tokenizer_name,
        "labels": [LABEL_LISP, LABEL_OTHER],
        "lisp_token_budget": lisp_tokens,
        "other_token_budget": other_tokens,
        "total_examples_consumed": progress["stream_examples_consumed"],
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    if upload and hf_api:
        _upload_file(hf_api, meta_path, repo_id, subfolder, hf_token)

    # Remove progress file if fully done
    all_done = all(
        labels_data[l]["train_tokens"] >= budgets[l]
        for l in budgets
    )
    if all_done:
        progress_path.unlink(missing_ok=True)
        _log("All budgets met — progress file removed")

    elapsed = time.time() - wall_start
    _log(f"\nDone! Wall time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    for label in [LABEL_LISP, LABEL_OTHER]:
        lp = labels_data[label]
        _log(f"  {label}: train={_fmt_tokens(lp['train_tokens'])} "
             f"({lp['train_shard_idx']} shards)  "
             f"test={_fmt_tokens(lp['test_tokens'])} "
             f"({lp['test_shard_idx']} shards)")
    _log_resources()


# ═══════════════════════════════════════════════════════════════════════
# Download mode
# ═══════════════════════════════════════════════════════════════════════


def download_bins(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN", None)
    repo_id = "AE-data/modular-pretraining"
    subfolder = "code"

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
    # Fast-path for worker mode
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main()
        sys.exit(0)

    ap = argparse.ArgumentParser(
        description="Prepare code-lisp / code-other token bins "
                    "from bigcode/the-stack-dedup")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: ./code)")
    ap.add_argument("--num_proc", type=int, default=32,
                    help="Workers for tokenisation per subprocess (default 32)")
    ap.add_argument("--max_length", type=int, default=-1)
    ap.add_argument("--length_strategy", default="none",
                    choices=["truncate", "drop", "none"])
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neo-125M")
    ap.add_argument("--download_bins", action="store_true",
                    help="Download existing bins from HF instead of generating")
    ap.add_argument("--upload_bins", action="store_true",
                    help="Upload each shard to HF Hub after writing")
    ap.add_argument("--lisp_tokens", type=int, default=1_000_000_000,
                    help="Train token budget for code-lisp (default 1B)")
    ap.add_argument("--other_tokens", type=int, default=24_000_000_000,
                    help="Train token budget for code-other (default 24B)")
    ap.add_argument("--chunk_size", type=int, default=100_000,
                    help="Examples per chunk (default 100k)")
    args = ap.parse_args()

    default_out_dir = Path(__file__).parent / "code"
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.resolve() != default_out_dir.resolve():
        if default_out_dir.is_symlink() or default_out_dir.exists():
            default_out_dir.unlink() if default_out_dir.is_symlink() else shutil.rmtree(default_out_dir)
        default_out_dir.parent.mkdir(parents=True, exist_ok=True)
        default_out_dir.symlink_to(out_dir.resolve())
        print(f"Symlinked {default_out_dir} -> {out_dir.resolve()}")

    if args.download_bins:
        download_bins(out_dir)
        sys.exit(0)

    orchestrate(
        out_dir=out_dir,
        num_proc=args.num_proc,
        max_length=args.max_length,
        length_strategy=args.length_strategy,
        tokenizer_name=args.tokenizer,
        upload=args.upload_bins,
        lisp_tokens=args.lisp_tokens,
        other_tokens=args.other_tokens,
        chunk_size=args.chunk_size,
    )

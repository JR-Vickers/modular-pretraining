#!/usr/bin/env python
"""
Prepare token-level binary shards and write a metadata.json file.

Performance optimisations vs. baseline
---------------------------------------
1. hf_transfer   – Rust-accelerated multi-threaded HF Hub downloads
2. Batched tok    – one tokenizer.encode call per batch (fewer Python↔Rust trips)
3. Bucket split   – single O(N) pass instead of L×N filter scans
4. Streaming mmap – writes tokens in batches → peak RAM ∝ batch_size, not dataset
5. Cache cleanup  – Arrow intermediates + optional full HF cache nuke
6. Timestamps     – every log line timestamped; never silent > 30 s
"""

# ── Set perf env vars BEFORE any HF / tokenizers imports ─────────────
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")   # Rust downloads
os.environ["TOKENIZERS_PARALLELISM"] = "false"            # avoid fork deadlocks

import argparse
import gc
import json
import shutil
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity(40)

# Verify hf_transfer is available (the env var on line 10 enables it;
# this just gives the user a clear message at import time).
try:
    import hf_transfer  # noqa: F401
    _HF_TRANSFER_OK = True
except ImportError:
    _HF_TRANSFER_OK = False

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


# ─────────────────────── helpers ─────────────────────────────────────


def _ts() -> str:
    """Timestamp prefix for log lines."""
    return time.strftime("[%H:%M:%S]")


def _log(msg: str) -> None:
    """Print with timestamp and immediate flush."""
    print(f"{_ts()} {msg}", flush=True)


def memmap_write_streaming(
    fname: Path,
    dataset: Dataset,
    ids_column: str = "ids",
    len_column: str = "len",
    dtype: np.dtype = np.uint16,
    batch_size: int = 5_000,
) -> int:
    """
    Stream-write token ids to a memmap file.

    Only *batch_size* examples' ids are resident in Python at a time,
    keeping peak RAM proportional to batch_size rather than dataset size.

    Returns:
        Total number of tokens written.
    """
    # len column is small ints → cheap to load in full
    total_tokens = int(np.sum(dataset[len_column]))
    if total_tokens == 0:
        np.memmap(fname, dtype=dtype, mode="w+", shape=(1,))
        return 0

    mmap = np.memmap(fname, dtype=dtype, mode="w+", shape=(total_tokens,))
    idx = 0
    n = len(dataset)

    with tqdm(total=n, desc=f"  write {fname.name}", unit="ex", mininterval=10) as pbar:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = dataset[start:end][ids_column]   # only this slice in RAM
            for ids in batch_ids:
                mmap[idx : idx + len(ids)] = ids
                idx += len(ids)
            pbar.update(end - start)

    mmap.flush()
    del mmap
    gc.collect()
    return total_tokens


# ─────────────────────── prep ────────────────────────────────────────


def prep(
    num_proc: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    column: str,
    length_strategy: Literal["truncate", "drop", "none"],
    sample_pct: float = 1.0,
) -> Dict[str, Dict[str, Dataset]]:

    _log("Loading dataset AE-data/dual-use-papers ...")
    dset_name = "AE-data/dual-use-papers"
    ds = load_dataset(dset_name, split="train")
    _log(f"Loaded {len(ds):,} examples")

    # Sample dataset if requested
    if sample_pct < 1.0:
        sample_size = int(len(ds) * sample_pct)
        ds = ds.select(range(sample_size))
        _log(f"Sampled {sample_pct*100:.1f}%: {sample_size:,} examples")

    # Keep only the column "column" plus "text"
    ds = ds.select_columns([column, "text"])

    # ── train / test split ───────────────────────────────────────────
    _log("Splitting train/test (90/10) ...")
    splits = ds.train_test_split(test_size=0.1, seed=42)
    train, test = splits["train"], splits["test"]
    _log(f"Train: {len(train):,}  Test: {len(test):,}")

    # Combine split-label addition + category normalisation into ONE map
    # (saves 2 extra .map() calls and their Arrow cache files)
    def _prep_train(ex):
        return {"split": "train", column: "papers-" + str(ex[column]).lower().replace(" ", "-")}

    def _prep_test(ex):
        return {"split": "test", column: "papers-" + str(ex[column]).lower().replace(" ", "-")}

    train = train.map(_prep_train, num_proc=num_proc, desc="prep train")
    test = test.map(_prep_test, num_proc=num_proc, desc="prep test")

    ds = concatenate_datasets([train, test])
    del train, test, splits
    gc.collect()

    _log(f"Columns: {ds.column_names}")

    # ─────────────────────────────────────────────────────────────────
    # 1. tokenisation (batched → far fewer Python↔Rust round-trips)
    # ─────────────────────────────────────────────────────────────────
    labels = sorted(ds.unique(column))
    _log(f"Found {len(labels)} unique '{column}' values: {labels}")

    eos_id = tokenizer.eos_token_id
    do_trunc = length_strategy == "truncate" and max_length > 0

    def tok_batch(examples: Dict[str, list]) -> Dict[str, list]:
        """Tokenise a whole batch at once (called by Dataset.map w/ batched=True)."""
        all_ids: List[list] = []
        all_lens: List[int] = []
        for text in examples["text"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(eos_id)
            if do_trunc:
                ids = ids[:max_length]
                ids[-1] = eos_id
            all_ids.append(ids)
            all_lens.append(len(ids))
        return {"ids": all_ids, "len": all_lens}

    _log(f"Tokenising ({num_proc} workers, batched=True) ...")
    ds = ds.map(
        tok_batch,
        batched=True,
        batch_size=2_000,
        num_proc=num_proc,
        remove_columns=["text"],   # drop text column early → less RAM & cache
        desc="tokenize",
    )
    gc.collect()

    # Drop long examples if needed
    if length_strategy == "drop" and max_length > 0:
        before = len(ds)
        ds = ds.filter(
            lambda ex: ex["len"] <= max_length,
            num_proc=num_proc,
            desc="drop long",
        )
        _log(f"Dropped {before - len(ds):,} examples > {max_length} tokens")

    # ─────────────────────────────────────────────────────────────────
    # 2. vectorised bucket split  (numpy, no Python loop)
    # ─────────────────────────────────────────────────────────────────
    _log("Building (label, split) index arrays (vectorised) ...")
    col_arr = np.array(ds[column])        # 1-D string array
    split_arr = np.array(ds["split"])     # "train" / "test"
    all_indices = np.arange(len(ds))

    data = OrderedDict()
    for label in labels:
        label_mask = col_arr == label
        tr_idx = all_indices[label_mask & (split_arr == "train")].tolist()
        te_idx = all_indices[label_mask & (split_arr == "test")].tolist()
        data[label] = {
            "train": ds.select(tr_idx),
            "test":  ds.select(te_idx),
        }
        _log(f"  {label}: train={len(tr_idx):,}  test={len(te_idx):,}")

    del col_arr, split_arr, all_indices
    gc.collect()

    # Clean intermediate Arrow cache files
    try:
        n_cleaned = ds.cleanup_cache_files()
        if n_cleaned:
            _log(f"Cleaned {n_cleaned} intermediate cache files")
    except Exception:
        pass

    return data


# ─────────────────────── write ───────────────────────────────────────


def write(
    datasets: Dict[str, Dict[str, Dataset]],
    column: str,
    out_dir: Path,
    max_length: int,
    tokenizer_name: str,
    tokenizer: AutoTokenizer,
    length_strategy: Literal["truncate", "drop", "none"],
) -> None:
    """Write datasets to binary files and collect metadata."""

    meta: dict[str, Any] = {}
    total_tokens_train = 0
    total_tokens_test = 0
    labels = sorted(list(datasets.keys()))

    for label, splits_data in datasets.items():

        _log(f"Writing label={label}")

        meta[label] = {
            "train": {},
            "test": {},
        }

        for split in ["train", "test"]:

            subset = splits_data[split]
            out_path = out_dir / f"{label}_{split}.bin"
            if out_path.exists():
                os.remove(out_path)

            # Streaming write (low RAM)
            total_tokens = memmap_write_streaming(out_path, subset)

            # ---------- per-split statistics ----------
            example_text = tokenizer.decode(subset[-1]["ids"], skip_special_tokens=False)

            meta[label][split] = {
                "total_tokens": total_tokens,
                "example": example_text,
            }

            if split == "train":
                total_tokens_train += total_tokens
            else:
                total_tokens_test += total_tokens

            _log(f"  {label}/{split}: {total_tokens:,} tokens -> {out_path.name}")

    # ---------- global statistics ----------
    meta["all"] = {
        "total_tokens_train": total_tokens_train,
        "total_tokens_test": total_tokens_test,
        "tokenizer": tokenizer_name,
        "vocab_size": len(tokenizer),
        "max_length": max_length,
        "column": column,
        "labels": labels,
        "length_strategy": length_strategy,
    }

    # ---------------------------------------------------- #
    # dump metadata.json                                   #
    # ---------------------------------------------------- #
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    _log("metadata.json written")


# --------------------------------------------------------------------------- #
# cache cleanup                                                               #
# --------------------------------------------------------------------------- #


def _cleanup_hf_cache() -> None:
    """Remove HuggingFace datasets Arrow cache to reclaim disk space."""
    cache_dir = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            Path.home() / ".cache" / "huggingface" / "datasets",
        )
    )
    if cache_dir.exists():
        sz = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        _log(f"Removing HF datasets cache ({sz / 1e9:.2f} GB) at {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
    _log("Cache cleaned")


# --------------------------------------------------------------------------- #
# main preparation sequence                                                   #
# --------------------------------------------------------------------------- #


def _delete_remote_bins(repo_id: str, subfolder: str, hf_token: str | None) -> None:
    """Delete all .bin and metadata.json files under *subfolder* in the remote repo."""
    api = HfApi(token=hf_token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
    targets = [
        f for f in repo_files
        if f.startswith(subfolder + "/") and (f.endswith(".bin") or f.endswith("metadata.json"))
    ]
    if not targets:
        _log(f"No remote files to delete in {repo_id}/{subfolder}")
        return
    from huggingface_hub import CommitOperationDelete
    ops = [CommitOperationDelete(path_in_repo=f) for f in targets]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Delete stale bins in {subfolder}/",
        token=hf_token,
    )
    _log(f"Deleted {len(targets)} remote files from {repo_id}/{subfolder}: {targets}")


def run(
        out_dir: Path | None,
        num_proc: int,
        column: str,
        max_length: int,
        length_strategy: str,
        tokenizer_name: str,
        download_bins: bool,
        upload_bins: bool,
        sample_pct: float = 1.0,
        clean_cache: bool = False,
        delete_uploaded_bins: bool = False,
    ) -> None:

    wall_start = time.time()

    if _HF_TRANSFER_OK:
        _log("hf_transfer enabled - Rust multi-threaded downloads active for all HF Hub calls")
    else:
        _log("hf_transfer not installed (pip install hf_transfer for ~5-10x speedup on all HF downloads)")

    default_out_dir = Path(__file__).parent / "papers"
    if out_dir is None:
        out_dir = default_out_dir
    else:
        out_dir = Path(out_dir)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.resolve() != default_out_dir.resolve():
        if default_out_dir.is_symlink() or default_out_dir.exists():
            default_out_dir.unlink() if default_out_dir.is_symlink() else shutil.rmtree(default_out_dir)
        default_out_dir.parent.mkdir(parents=True, exist_ok=True)
        default_out_dir.symlink_to(out_dir.resolve())
        _log(f"Symlinked {default_out_dir} -> {out_dir.resolve()}")

    _log(f"out_dir: {out_dir}")

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    repo_id = "AE-data/modular-pretraining"
    subfolder = "papers"

    # ── Delete remote bins if requested (before anything else) ──────
    if delete_uploaded_bins:
        _log(f"Deleting remote .bin files from {repo_id}/{subfolder} ...")
        _delete_remote_bins(repo_id, subfolder, hf_token)

    # ── Download bins if requested ───────────────────────────────────
    if download_bins:
        _log(f"Downloading .bin files from {repo_id}/{subfolder}...")
        api = HfApi(token=hf_token)

        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
            all_files = [
                f for f in repo_files
                if f.startswith(subfolder) and (f.endswith(".bin") or f.endswith("metadata.json"))
            ]

            if not all_files:
                _log(f"No .bin or metadata.json files found in {repo_id}/{subfolder}")
            else:
                # Download files concurrently via ThreadPoolExecutor
                def _download_one(file_path: str) -> str:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        repo_type="dataset",
                        token=hf_token,
                        local_dir=out_dir.parent,
                        local_dir_use_symlinks=False,
                    )
                    return f"{file_path} -> {out_dir / Path(file_path).name}"

                max_workers = min(8, len(all_files))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_download_one, fp): fp for fp in all_files}
                    for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                        _log(f"  {fut.result()}")

                _log("Download complete!")
        except Exception as e:
            _log(f"Error downloading files: {e}")
            raise

        return

    # ── Load tokenizer (ensure Rust fast backend) ────────────────────
    _log(f"Loading tokenizer {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if getattr(tokenizer, "is_fast", False):
        _log("Fast (Rust) tokenizer backend active")
    else:
        _log("WARNING: slow Python tokenizer in use - consider a model with a fast tokenizer")

    # ── Prep ─────────────────────────────────────────────────────────
    data = prep(
        num_proc=num_proc,
        tokenizer=tokenizer,
        column=column,
        max_length=max_length,
        length_strategy=length_strategy,
        sample_pct=sample_pct,
    )

    # ── Write datasets and metadata ──────────────────────────────────
    write(
        datasets=data,
        column=column,
        out_dir=out_dir,
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        length_strategy=length_strategy,
    )

    elapsed = time.time() - wall_start
    _log(f"Done - binary shards + metadata.json written to {out_dir}  (wall {elapsed:.0f}s)")

    # ── Optional cache cleanup ───────────────────────────────────────
    if clean_cache:
        _cleanup_hf_cache()

    # ── Upload bins if requested ─────────────────────────────────────
    if upload_bins:
        _log(f"Uploading .bin files to {repo_id}/{subfolder}...")
        api = HfApi(token=hf_token)

        # Ensure repo exists (will not error if it already exists)
        try:
            api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, repo_type="dataset")
            _log(f"Dataset repository {repo_id} ready")
        except Exception as e:
            _log(f"Note: Could not create/verify repo (it may already exist): {e}")

        # Find all .bin files and metadata.json in out_dir
        bin_files = list(out_dir.glob("*.bin"))
        metadata_file = out_dir / "metadata.json"

        files_to_upload = bin_files.copy()
        if metadata_file.exists():
            files_to_upload.append(metadata_file)

        if not files_to_upload:
            _log(f"No .bin or metadata.json files found in {out_dir} to upload")
        else:
            for file_path in tqdm(files_to_upload, desc="Uploading files"):
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"{subfolder}/{file_path.name}",
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token,
                    )
                    _log(f"  Uploaded {file_path.name} to {repo_id}/{subfolder}")
                except Exception as e:
                    _log(f"Error uploading {file_path.name}: {e}")
                    raise

            _log("Upload complete!")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    ap = argparse.ArgumentParser("Prepare papers dataset")
    ap.add_argument("--out_dir", default=None, help="directory to write .bin files")
    ap.add_argument("--num_proc", type=int, default=64,
                    help="parallel workers for dataset.map (default 64; set ≤ core count)")
    ap.add_argument("--column", type=str, default="category")
    ap.add_argument("--max_length", type=int, default=-1)
    ap.add_argument("--length_strategy", type=str, default="none", choices=["truncate", "drop", "none"])
    ap.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neo-125M")
    ap.add_argument("--download_bins", action="store_true")
    ap.add_argument("--upload_bins", action="store_true")
    ap.add_argument("--sample", type=float, default=1.0, help="Fraction of data to use (0.0-1.0), e.g., 0.01 for 1%%")
    ap.add_argument("--clean_cache", action="store_true",
                    help="remove HF datasets cache after the run to reclaim disk space")
    ap.add_argument("--delete_uploaded_bins", action="store_true",
                    help="delete all remote .bin/metadata.json in papers/ before starting")
    args = ap.parse_args()

    run(
        out_dir=args.out_dir,
        num_proc=args.num_proc,
        column=args.column,
        max_length=args.max_length,
        length_strategy=args.length_strategy,
        tokenizer_name=args.tokenizer,
        download_bins=args.download_bins,
        upload_bins=args.upload_bins,
        sample_pct=args.sample,
        clean_cache=args.clean_cache,
        delete_uploaded_bins=args.delete_uploaded_bins,
    )

#!/usr/bin/env python
"""
Prepare token-level binary shards and write a metadata.json file.
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal
from collections import OrderedDict

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging

logging.set_verbosity(40)

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

def memmap_write(
    fname: Path,
    arr: List[List[int]],
    dtype: np.dtype = np.uint16,
) -> None:
    """
    Write array data to a memory-mapped file.

    Args:
        fname: Path to output file
        arr: List of arrays to write
        dtype: NumPy data type for the memory-mapped array
    """
    
    total = sum(len(a) for a in arr)
    mmap = np.memmap(fname, dtype=dtype, mode="w+", shape=(total,))
    idx = 0
    for a in tqdm(arr, desc="writing", total=len(arr)):
        mmap[idx : idx + len(a)] = a
        idx += len(a)
    mmap.flush()


def prep(
    num_proc: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    column: str,
    length_strategy: Literal["truncate", "drop", "none"],
    sample_pct: float = 1.0,
) -> Dict[str, Dataset]:

    dset_name = "SimpleStories/SimpleStories"
    ds = load_dataset(dset_name, split="train")

    # Sample dataset if requested
    if sample_pct < 1.0:
        sample_size = int(len(ds) * sample_pct)
        ds = ds.select(range(sample_size))
        print(f"Sampling {sample_pct*100}% of data: {sample_size} examples")

    #keep only the column "column" plus "story"
    ds = ds.select_columns([column, "story"])

    splits = ds.train_test_split(test_size=0.1, seed=42) #NOTE test size was 0.01 in OG version
    train, test = splits["train"], splits["test"]

    train = train.map(lambda ex: {"split": "train"}, num_proc=num_proc)
    test = test.map(lambda ex: {"split": "test"}, num_proc=num_proc)

    ds = concatenate_datasets([train, test])

    # For the col "column", replace all values in that col with the value in that col replaced with dashes
    ds = ds.map(lambda ex: {column: str(ex[column]).lower().replace(' ', '-')}, num_proc=num_proc)

    print("Dataset columns:", ds.column_names)

    # --------------------------------------------------------- #
    # 1. tokenisation                                           #
    # --------------------------------------------------------- #

    # Build label list
    labels = sorted(ds.unique(column))
    print(f"Found {len(labels)} unique {column} values")
    print(labels)

    def tok_fn(ex: Dict[str, Any]) -> Dict[str, Any]:

        ids = tokenizer.encode(ex["story"], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)

        if length_strategy == "truncate" and max_length > 0:
            ids = ids[:max_length]
            ids[-1] = tokenizer.eos_token_id

        return {"ids": ids, "len": len(ids)}

    ds = ds.map(tok_fn, num_proc=num_proc)

    # If dropping is enabled, remove stories longer than max_length
    if length_strategy == "drop" and max_length > 0:
        ds = ds.filter(lambda ex: ex["len"] <= max_length, num_proc=num_proc)

    # --------------------------------------------------------- #
    # 2. value mapping                                          #
    # --------------------------------------------------------- #

    # value -> value_id (makes subsetting much faster)
    ds = ds.map(lambda ex: {f"{column}_id": labels.index(ex[column])}, num_proc=num_proc)

    data = OrderedDict()
    for label in tqdm(labels, desc=f"Splitting by {column}"):
        value_id = labels.index(label)
        subset = ds.filter(lambda ex, value_id=value_id: ex[f"{column}_id"] == value_id, num_proc=num_proc)
        train = subset.filter(lambda ex: ex["split"] == "train", num_proc=num_proc)
        test = subset.filter(lambda ex: ex["split"] == "test", num_proc=num_proc)
        data[label] = {
            "train": train,
            "test": test,
        }

    return data


def write(
    datasets: Dict[str, Dataset],
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

    for label, splits in datasets.items():

        print("label", label)
        print("splits", splits)

        meta[label] = {
            "train": {},
            "test": {},
        }

        for split in ["train", "test"]:

            subset = splits[split]
            out_path = out_dir / f"{label}_{split}.bin"
            if out_path.exists():
                os.remove(out_path)

            # write tokens
            memmap_write(
                out_path,
                subset["ids"],
                np.uint16,
            )

            # ---------- per‑split statistics ----------
            total_tokens = int(np.sum(subset["len"]))
            example_text = tokenizer.decode(subset[-1]["ids"], skip_special_tokens=False)

            meta[label][split] = {
                "total_tokens": total_tokens,
                "example": example_text,
            }

            if split == "train":
                total_tokens_train += total_tokens
            else:
                total_tokens_test += total_tokens

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


# --------------------------------------------------------------------------- #
# main preparation sequence                                                   #
# --------------------------------------------------------------------------- #


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
    ) -> None:

    default_out_dir = Path(__file__).parent / "stories"
    if out_dir is None:
        out_dir = default_out_dir

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.resolve() != default_out_dir.resolve():
        
        if default_out_dir.is_symlink() or default_out_dir.exists():
            default_out_dir.unlink() if default_out_dir.is_symlink() else shutil.rmtree(default_out_dir)
        default_out_dir.parent.mkdir(parents=True, exist_ok=True)
        default_out_dir.symlink_to(out_dir.resolve())
        print(f"Symlinked {default_out_dir} -> {out_dir.resolve()}")

    print("out_dir:", out_dir)

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN", None)
    repo_id = "erol-AE/GR-MoE"
    subfolder = "stories"

    # Download bins if requested
    if download_bins:
        print(f"Downloading .bin files from {repo_id}/{subfolder}...")
        api = HfApi(token=hf_token)
        
        # List all files in the subfolder
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
            bin_files = [f for f in repo_files if f.startswith(subfolder) and f.endswith('.bin')]
            
            # Also download metadata.json
            metadata_files = [f for f in repo_files if f.startswith(subfolder) and f.endswith('metadata.json')]
            
            all_files = bin_files + metadata_files
            
            if not all_files:
                print(f"No .bin or metadata.json files found in {repo_id}/{subfolder}")
            else:
                for file_path in tqdm(all_files, desc="Downloading files"):
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        repo_type="dataset",
                        token=hf_token,
                        local_dir=out_dir.parent,
                        local_dir_use_symlinks=False,
                    )
                    print(f"Downloaded {file_path} to {out_dir / Path(file_path).name}")
                    
                print("Download complete!")
        except Exception as e:
            print(f"Error downloading files: {e}")
            raise
        
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data = prep(
        num_proc=num_proc,
        tokenizer=tokenizer,
        column=column,
        max_length=max_length,       
        length_strategy=length_strategy,
        sample_pct=sample_pct,
    )

    # Write datasets and metadata
    write(
        datasets=data,
        column=column,
        out_dir=out_dir,
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        length_strategy=length_strategy,
    )

    print("Done - binary shards + metadata.json written to", out_dir)

    # Upload bins if requested
    if upload_bins:
        print(f"Uploading .bin files to {repo_id}/{subfolder}...")
        api = HfApi(token=hf_token)
        
        # Ensure repo exists (will not error if it already exists)
        try:
            api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, repo_type="dataset")
            print(f"Dataset repository {repo_id} ready")
        except Exception as e:
            print(f"Note: Could not create/verify repo (it may already exist): {e}")
        
        # Find all .bin files and metadata.json in out_dir
        bin_files = list(out_dir.glob("*.bin"))
        metadata_file = out_dir / "metadata.json"
        
        files_to_upload = bin_files.copy()
        if metadata_file.exists():
            files_to_upload.append(metadata_file)
        
        if not files_to_upload:
            print(f"No .bin or metadata.json files found in {out_dir} to upload")
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
                    print(f"Uploaded {file_path.name} to {repo_id}/{subfolder}")
                except Exception as e:
                    print(f"Error uploading {file_path.name}: {e}")
                    raise
            
            print("Upload complete!")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    ap = argparse.ArgumentParser("Prepare simple stories")
    ap.add_argument("--out_dir", default=None, help="directory to write .bin files")
    ap.add_argument("--num_proc", type=int, default=20)
    ap.add_argument("--column", type=str, default="topic")
    ap.add_argument("--max_length", type=int, default=-1)
    ap.add_argument("--length_strategy", type=str, default="none", choices=["truncate", "drop", "none"])
    ap.add_argument("--tokenizer", type=str, default="SimpleStories/SimpleStories-1.25M")
    ap.add_argument("--download_bins", action="store_true")
    ap.add_argument("--upload_bins", action="store_true")
    ap.add_argument("--sample", type=float, default=1.0, help="Fraction of data to use (0.0-1.0), e.g., 0.01 for 1%%")
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
    )
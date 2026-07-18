"""Download and validate the public SimpleStories token shards used by Phase 1."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


STORIES_REPO_ID = "erol-AE/GR-MoE"


def expected_stories_files(metadata: dict) -> dict[str, int]:
    """Return expected filename -> byte size from tracked metadata."""
    expected = {}
    for label in metadata["all"]["labels"]:
        for split in ("train", "test"):
            expected[f"{label}_{split}.bin"] = 2 * metadata[label][split]["total_tokens"]
    return expected


def download_missing_stories(data_dir: Path) -> list[Path]:
    """Download only absent public story shards, never any other dataset path."""
    from huggingface_hub import hf_hub_download

    metadata = json.loads((data_dir / "metadata.json").read_text())
    downloaded = []
    for filename in expected_stories_files(metadata):
        destination = data_dir / filename
        if destination.exists():
            continue
        print(f"Downloading {filename}", flush=True)
        cached = hf_hub_download(
            repo_id=STORIES_REPO_ID,
            repo_type="dataset",
            filename=f"stories/{filename}",
        )
        temporary = destination.with_suffix(".bin.tmp")
        shutil.copyfile(cached, temporary)
        temporary.replace(destination)
        downloaded.append(destination)
    return downloaded


def validate_stories_data(data_dir: Path) -> dict[str, int]:
    """Validate the 48 train/test pairs against the tracked token totals."""
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing tracked metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    labels = metadata["all"]["labels"]
    if len(labels) != 48 or len(set(labels)) != 48:
        raise ValueError(f"Expected 48 unique story labels, found {len(set(labels))}")
    if metadata["all"].get("vocab_size") != 4096:
        raise ValueError("Stories metadata must record vocabulary size 4096")

    expected = expected_stories_files(metadata)
    missing = [name for name in expected if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} stories shards (first: {missing[0]})"
        )

    for filename, expected_bytes in expected.items():
        actual_bytes = (data_dir / filename).stat().st_size
        if actual_bytes <= 0:
            raise ValueError(f"Empty stories shard: {filename}")
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Token count mismatch for {filename}: "
                f"expected {expected_bytes // 2}, found {actual_bytes // 2}"
            )

    train_total = sum(metadata[label]["train"]["total_tokens"] for label in labels)
    test_total = sum(metadata[label]["test"]["total_tokens"] for label in labels)
    if train_total != metadata["all"]["total_tokens_train"]:
        raise ValueError("Per-label training totals do not match metadata all-total")
    if test_total != metadata["all"]["total_tokens_test"]:
        raise ValueError("Per-label test totals do not match metadata all-total")
    return {
        "labels": len(labels),
        "files": len(expected),
        "vocab_size": metadata["all"]["vocab_size"],
        "train_tokens": train_total,
        "test_tokens": test_total,
    }

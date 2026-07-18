from __future__ import annotations

import os
import re
from collections import defaultdict
import random
import torch
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Union

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from src.run.util.distributed import get_rank
from src.run.util.tools import set_seeds


class TokenDataset(Dataset):
    """PyTorch Dataset for memory-mapped token files."""

    def __init__(self, filename: str, T: int):
        self.filename = filename
        self.T = T

        # Load tokens as memory map
        self.tokens = np.memmap(filename, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.tokens)

        # The number of sequences that can be formed
        self.num_sequences = (self.num_tokens - 1) // self.T

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:

        # Get sequence of T tokens if _labeled to not repeat label tokens at the end
        if "_labeled" in self.filename:
            buf = self.tokens[idx * self.T : (idx + 1) * self.T]
            
        # Get sequence of T+1 tokens otherwise
        else:
            buf = self.tokens[idx * self.T : (idx + 1) * self.T + 1]

        # Convert to tensor
        return torch.from_numpy(buf.astype(np.int64))


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, B: int, T: int):
        self.B = B
        self.T = T
    
    @abstractmethod
    def reset(self, epoch: int) -> None:
        """Reset the data loader."""
        pass
    
    @abstractmethod
    def next_batch(self) -> tuple[torch.Tensor, str | None]:
        """Get the next batch of data."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of batches."""
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}(len={len(self)})"


class SingleDataLoader(DataLoader):

    def __init__(
        self,
        filename: str,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        label: str | None = None,
        seed: int = 42,
        drop_last: bool = True,
        start: int = 0,
        end: int = -1,
    ):
        self.B = B
        self.T = T
        self.device = device
        self.filename = filename
        self.label = label
        self.seed = seed

        base_dataset = TokenDataset(filename, T)
        
        # Apply partition if start/end are specified.
        # NOTE: start/end are interpreted as per-rank sequence indices.
        if start != 0 or end != -1:

            dataset_size = len(base_dataset)

            # Calculate total number of sequences per rank.
            if drop_last:
                total_sequences = dataset_size // num_processes
            else:
                total_sequences = (dataset_size + num_processes - 1) // num_processes

            # Normalize negative sequence indices.
            if start < 0:
                start += total_sequences

            if end < 0:
                end += total_sequences + 1

            assert 0 <= start <= total_sequences, f"start out of range 0, {total_sequences}"
            assert 0 <= end <= total_sequences, f"end out of range 0, {total_sequences}"
            assert start <= end, f"start ({start}) must be <= end ({end})"
            assert end - start >= B, (
                f"partition too small for one batch: start={start}, end={end}, B={B}"
            )

            # Convert per-rank sequence indices to global dataset indices.
            start_seq = start * num_processes
            end_seq = end * num_processes
            
            # Define indices selector
            indices = list(range(start_seq, end_seq))
            self.dataset = Subset(base_dataset, indices)

        else:
            self.dataset = base_dataset

        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=num_processes,
            rank=process_rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )

        self.dataloader = TorchDataLoader(
            self.dataset,
            batch_size=B,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=drop_last,
        )

        self.iterator = iter(self.dataloader)

        self.last_epoch = 0

    def reset(self, epoch: int = 0) -> None:
        """Resets the data iterator."""
        self.sampler.set_epoch(epoch)
        self.iterator = iter(self.dataloader)
        self.last_epoch = epoch

    def next_batch(self) -> tuple[torch.Tensor, str | None]:
        """Gets the next batch, automatically resetting the iterator if it's exhausted."""
        try:
            batch = next(self.iterator)

        except StopIteration:
            self.reset(self.last_epoch + 1)
            batch = next(self.iterator)

        if self.device is not None:
            batch = batch.to(self.device, non_blocking=self.dataloader.pin_memory)

        return batch, self.label

    def __len__(self) -> int:
        return len(self.dataloader)

    def __repr__(self) -> str:
        src = Path(self.filename).name
        return f"SingleDataLoader({src}, len={len(self)})"

    #implement the partition method
    #returns a new instance of SingleDataLoader
    #but such that the underlying dataset is only the subset of the data from start to end
    def partition(self, start: int, end: int) -> DataLoader:
        """Create a new SingleDataLoader with a subset of batches from start to end.
        
        Args:
            start: Start index (inclusive) relative to the original dataset
            end: End index (exclusive) relative to the original dataset
        """

        # Convert batch indices to per-rank sequence indices.
        seq_start = start * self.B
        seq_end = end * self.B
        return self.partition_sequences(seq_start, seq_end)

    def partition_sequences(self, start: int, end: int) -> DataLoader:
        """Create a new SingleDataLoader with a subset of sequences per rank.

        Args:
            start: Start sequence index (inclusive), per rank
            end: End sequence index (exclusive), per rank
        """

        # Create new instance with partition parameters
        return SingleDataLoader(
            filename=self.filename,
            B=self.B,
            T=self.T,
            process_rank=self.sampler.rank,
            num_processes=self.sampler.num_replicas,
            num_workers=self.dataloader.num_workers,
            device=self.device,
            pin_memory=self.dataloader.pin_memory,
            persistent_workers=self.dataloader.persistent_workers,
            label=self.label,
            seed=self.sampler.seed,
            drop_last=self.dataloader.drop_last,
            start=start,
            end=end,
        )

class InterleavedDataLoader(DataLoader):
    """Combine multiple loaders with either sequential or interleaved iteration.

    strategy="sequential":  exhaust each sub-loader in order before moving
                            to the next (deterministic, reproducible).
    strategy="interleaved": randomly sample from sub-loaders each step,
                            weighted by remaining batch counts (uses a
                            torch RNG for DDP-safe determinism).
    """

    def __init__(
        self,
        loaders: Sequence[DataLoader],
        weighted: bool = True,
    ) -> None:

        assert len(loaders) > 0, "No loaders provided"

        base_B, base_T = loaders[0].B, loaders[0].T
        for ld in loaders:
            assert ld.B == base_B and ld.T == base_T, "All loaders must have the same B and T values."

        super().__init__(base_B, base_T)

        self.loaders: List[DataLoader] = list(loaders)
        self.seed = self.loaders[0].seed

        random.seed(self.seed)
        random.shuffle(self.loaders)
        self.weighted = weighted

        self.remaining: List[int] = []
        self.last_epoch = 0

        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(self.seed)

        self.reset()

    def __repr__(self) -> str:
        return f"InterleavedDataLoader(len={len(self)})"

    def reset(self, epoch: int = 0) -> None:
        for ld in self.loaders:
            ld.reset(epoch)
        self.remaining = [len(ld) for ld in self.loaders]
        self.last_epoch = epoch
        self.rng.manual_seed(self.seed + epoch)

    def next_batch(self) -> tuple[torch.Tensor, str | None]:

        if sum(self.remaining) == 0:
            self.reset(self.last_epoch + 1)

        if self.weighted:
            probs = torch.tensor(self.remaining, dtype=torch.float32)
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1, generator=self.rng).item()
        else:
            idx = torch.randint(0, len(self.loaders), (1,), generator=self.rng).item()

        self.remaining[idx] -= 1
        return self.loaders[idx].next_batch()

    def __len__(self) -> int:
        return sum(len(ld) for ld in self.loaders)

    def partition(self, start: int, end: int) -> DataLoader:
        if start < 0:
            start += len(self)
        start = min(max(0, start), len(self))

        if end < 0:
            end += len(self) + 1
        end = min(max(0, end), len(self))

        start_prc = start / len(self)
        end_prc = end / len(self)
        new_loaders = []
        for loader in self.loaders:
            temp_start = int(len(loader) * start_prc)
            temp_end = int(len(loader) * end_prc)
            new_loaders.append(loader.partition(temp_start, temp_end))

        return InterleavedDataLoader(new_loaders, weighted=self.weighted)

# --------------------------------------------------------------------------- #
# Helper functions for auto-detecting categories and shards                   #
# --------------------------------------------------------------------------- #


def parse_bin_filename(filename: str) -> Optional[tuple[str, str, Optional[int]]]:
    """
    Parse a .bin filename into (category, split, shard_idx).
    
    Supports patterns:
    - category_train.bin -> ("category", "train", None)
    - category_test.bin -> ("category", "test", None)
    - category_train_00.bin -> ("category", "train", 0)
    - category_test_03.bin -> ("category", "test", 3)
    
    Args:
        filename: Filename to parse (e.g., "biology_train_00.bin")
    
    Returns:
        Tuple of (category, split, shard_idx) or None if pattern doesn't match
    """
    # Pattern: {category}_{split}[_{shard}].bin
    # Matches: biology_train.bin, biology_train_00.bin, biology_test.bin, etc.
    pattern = r'^(.+?)_(train|test)(?:_(\d+))?\.bin$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    category = match.group(1)
    split = match.group(2)
    shard_str = match.group(3)
    shard_idx = int(shard_str) if shard_str is not None else None
    
    return category, split, shard_idx


def get_bin_file_token_count(bin_paths: list[Path]) -> int:
    """Calculate total token count available from bin files."""
    total_tokens = 0
    for bin_path in bin_paths:
        file_size = os.path.getsize(bin_path)
        num_tokens = file_size // 2  # uint16 = 2 bytes per token
        total_tokens += num_tokens  
    return total_tokens


def get_labels_token_count(
    data_dirs: list[Path],
    labels: list[str],
) -> int:
    """Calculate total token count for labels across data dirs."""
    categories = {}
    for data_dir in data_dirs:
        categories.update(auto_detect_categories(data_dir))

    all_bin_paths = []
    for label in labels:
        if label in categories:
            all_bin_paths.extend(categories[label]["train"])
        else:
            raise ValueError(f"Label {label} not found in data directories")

    return get_bin_file_token_count(all_bin_paths)


def auto_detect_categories(data_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """
    Auto-detect all categories and their shards from .bin files in a directory.
    
    Args:
        data_dir: Directory containing .bin files
    
    Returns:
        Dict of {category: {"train": [shard_paths], "test": [shard_paths]}}
        Shard lists are sorted by shard index.
    
    Example:
        {
            "fineweb": {
                "train": [Path("fineweb_train_00.bin"), Path("fineweb_train_01.bin")],
                "test": [Path("fineweb_test.bin")]
            },
            "biology": {
                "train": [Path("biology_train.bin")],
                "test": [Path("biology_test.bin")]
            }
        }
    """
    categories = defaultdict(lambda: {"train": [], "test": []})
    
    # Find all .bin files
    bin_files = list(data_dir.glob("*.bin"))
    
    for bin_path in bin_files:
        parsed = parse_bin_filename(bin_path.name)
        if parsed is None:
            continue
        
        category, split, shard_idx = parsed
        
        # Store as (shard_idx, path) for sorting
        categories[category][split].append((shard_idx if shard_idx is not None else 0, bin_path))
    
    # Sort by shard index and extract just paths
    result = {}
    for category, splits in categories.items():
        result[category] = {
            "train": [path for _, path in sorted(splits["train"])],
            "test": [path for _, path in sorted(splits["test"])],
        }
    
    return result    


def make_loaders(
    data_dirs: list[Path],
    aux_labels: list[str],
    core_labels: list[str],
    label_token_counts: dict[str, int],
    num_processes: int,
    B: int, #micro batch size
    T: int, #sequence length
    seed: int,
    device: torch.device,
    pin_memory: bool = True,
    max_num_test: int = -1, #max test tokens per label
    upsample_labels: set[str] = frozenset(), #train labels allowed to repeat past avail
) -> dict[str, DataLoader]:
    """
    Args:
        data_dirs: List of directories containing .bin files
        aux_labels: List of auxiliary labels to create loaders for
        core_labels: List of core labels to create loaders for
        label_token_counts: Dictionary of label token counts
        num_processes: Number of GPUs
        B: Micro batch size
        T: Sequence length
        seed: Random seed
        device: Device to load data to
        max_num_test: Max test tokens
        upsample_labels: train-split labels whose token_limit may exceed their
            available tokens; the shortfall is filled by repeating shards. Any
            label not listed keeps the downsample-only behaviour (use available
            tokens once). Empty by default, so existing callers are unaffected.

    Returns:
        loaders_dict: {label: {"train": loader, "test": loader}, ...}
    """

    process_rank = get_rank()

    categories = {}
    for data_dir in data_dirs:
        categories.update(auto_detect_categories(data_dir))

    all_labels = sorted(categories.keys())

    assert all(x in all_labels for x in aux_labels), f"aux_labels {set(aux_labels) - set(all_labels)} not found in all_labels"
    assert all(x in all_labels for x in core_labels), f"core_labels {set(core_labels) - set(all_labels)} not found in all_labels"

    loaders = {}

    for label in all_labels:

        loaders[label] = {}

        for split in ["train", "test"]:

            shard_paths = categories[label][split]
            
            set_seeds(seed)
            random.shuffle(shard_paths)
            
            assert len(shard_paths) > 0, f"No {split} data found for label {label}"

            token_limit = int(label_token_counts[label])
            if max_num_test > 0 and split == "test":
                token_limit = min(token_limit, int(max_num_test))
            
            shard_loaders = []
            total_tokens = 0
            for shard_path in shard_paths:
                
                loader = SingleDataLoader(
                    filename=str(shard_path),
                    B=B,
                    T=T,
                    process_rank=process_rank,
                    num_processes=num_processes,
                    label=label if label in aux_labels else "core",
                    seed=seed,
                    device=device,
                    pin_memory=pin_memory,
                    drop_last=True,
                )

                # num micro batches * len of micro batch * num gpus * ctx len
                shard_tokens = len(loader) * B * num_processes * T

                if total_tokens + shard_tokens > token_limit:
                    gap = token_limit - total_tokens
                    tokens_per_iter = B * num_processes * T
                    if gap < tokens_per_iter:
                        break
                    end_seq = gap // (num_processes * T) #tokens -> sequences
                    loader = loader.partition_sequences(0, end_seq)
                    shard_loaders.append(loader)
                    break

                total_tokens += shard_tokens
                shard_loaders.append(loader)

            # Upsample: if this train label is allowed to repeat and one pass
            # over its shards fell short of token_limit, cycle the shards
            # (re-instantiating loaders over the same files) until we reach it.
            # Labels not in upsample_labels skip this entirely -> unchanged
            # downsample-only behaviour. We recompute the loaded token count
            # from shard_loaders rather than trusting total_tokens, because the
            # first pass does not increment total_tokens on its partition-break.
            def _loaded_tokens(lst):
                return sum(len(ld) * B * num_processes * T for ld in lst)

            if label in upsample_labels and split == "train":
                tokens_per_iter = B * num_processes * T
                cur = _loaded_tokens(shard_loaders)
                while cur < token_limit:
                    pass_start = cur
                    stop = False
                    for shard_path in shard_paths:
                        loader = SingleDataLoader(
                            filename=str(shard_path),
                            B=B,
                            T=T,
                            process_rank=process_rank,
                            num_processes=num_processes,
                            label=label if label in aux_labels else "core",
                            seed=seed,
                            device=device,
                            pin_memory=pin_memory,
                            drop_last=True,
                        )
                        shard_tokens = len(loader) * B * num_processes * T
                        if shard_tokens == 0:
                            continue
                        if cur + shard_tokens > token_limit:
                            gap = token_limit - cur
                            if gap < tokens_per_iter:
                                stop = True
                                break
                            end_seq = gap // (num_processes * T) #tokens -> sequences
                            loader = loader.partition_sequences(0, end_seq)
                            shard_loaders.append(loader)
                            cur += end_seq * num_processes * T
                            stop = True
                            break
                        shard_loaders.append(loader)
                        cur += shard_tokens
                    if stop or cur <= pass_start:  # filled, or no progress
                        break

            if len(shard_loaders) == 0:
                raise ValueError(
                    f"No shard loader created for label '{label}' split '{split}' "
                    f"with token_limit={token_limit}"
                )
            if len(shard_loaders) > 1:
                loaders[label][split] = InterleavedDataLoader(shard_loaders)
            else:
                loaders[label][split] = shard_loaders[0]

    #make output "core" and "all" loaders
    for agg_name, agg_labels in [("core", core_labels), ("aux", aux_labels), ("all", core_labels + aux_labels)]:
        loaders[agg_name] = {}
        for split in ["train", "test"]:
            agg_loaders = [loaders[label][split] for label in agg_labels if split in loaders[label]]
            if len(agg_loaders) > 1:
                loaders[agg_name][split] = InterleavedDataLoader(agg_loaders)
            elif len(agg_loaders) == 1:
                loaders[agg_name][split] = agg_loaders[0]
    
    return loaders

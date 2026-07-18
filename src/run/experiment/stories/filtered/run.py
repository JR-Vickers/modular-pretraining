"""Run the Phase 2 dense control with the primary stories topic held out.

MacBook Pro (MPS)::

    python -m src.run.experiment.stories.filtered.run

Pass ``--experiment-id`` with an existing result directory name to resume it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data.stories_utils import download_missing_stories, validate_stories_data
from src.run.experiment.config import GetStoriesConfig
from src.run.main import run
from src.run.train.base import FilteringConfig


FULL_TRAIN_TOKENS = 547_853_673
PRIMARY_AUX_LABEL = "a-deadline-or-time-limit"
PRIMARY_TRAIN_TOKENS = 11_625_008
FILTERED_TRAIN_TOKENS = FULL_TRAIN_TOKENS - PRIMARY_TRAIN_TOKENS
REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = REPO_ROOT / "src/data/stories"


def make_filtered_config(args: argparse.Namespace):
    """Build the Phase 2 control that excludes only the primary topic."""
    config = GetStoriesConfig()
    config.data.core.method = "total"
    config.data.core.limit = 1.0
    config.data.aux.method = "total"
    config.data.aux.limit = 1.0

    aux_labels = sorted(config.data.aux.labels)
    if PRIMARY_AUX_LABEL not in aux_labels:
        raise ValueError(f"Primary auxiliary label is missing: {PRIMARY_AUX_LABEL}")
    retained = ["core", *[label for label in aux_labels if label != PRIMARY_AUX_LABEL]]

    config.run.res_root = REPO_ROOT / f"results/stories_phase2/seed_{args.seed}"
    config.run.seed = args.seed
    config.run.device = args.device
    config.run.dtype = args.dtype
    config.run.compile = args.compile
    config.run.cleanup_distributed = False
    config.run.s3_bucket = args.s3_bucket
    config.run.s3_prefix = args.s3_prefix
    config.run.target_effective_batch_size = -1
    config.run.micro_batch_size = 16
    config.run.accumulation_steps = 8
    config.run.nominal_token_budget = FILTERED_TRAIN_TOKENS
    config.run.model_shape = "paper"
    config.run.model_shape_note = "26M dense paper configuration; primary topic held out"
    if args.experiment_id is not None:
        config.run.experiment_id = args.experiment_id

    config.stages = [
        FilteringConfig(
            num_train_evals=0,
            num_checkpoints=args.num_checkpoints,
            do_elicit=False,
            lr=5e-3,
            acc_mode="heterogeneous",
            retain_targets=[retained],
        )
    ]
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="mps")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False,
        help="Compile the model (disabled by default for the validated MPS path)",
    )
    parser.add_argument("--num-checkpoints", type=int, default=2)
    parser.add_argument("--experiment-id", "--experiment_id", default=None)
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.s3_bucket is None) != (args.s3_prefix is None):
        raise ValueError("--s3-bucket and --s3-prefix must be provided together")
    if args.num_checkpoints < -1 or args.num_checkpoints == 0:
        raise ValueError("--num-checkpoints must be -1 or a positive integer")
    if args.download_only:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("--download-only must be run outside torchrun")
        downloaded = download_missing_stories(DATA_DIR)
        print(f"Downloaded {len(downloaded)} missing stories shards")

    summary = validate_stories_data(DATA_DIR)
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    primary_tokens = metadata[PRIMARY_AUX_LABEL]["train"]["total_tokens"]
    if summary["train_tokens"] != FULL_TRAIN_TOKENS or primary_tokens != PRIMARY_TRAIN_TOKENS:
        raise ValueError("Stories metadata does not match the Phase 2 filtered-run budget")
    print(json.dumps(summary, indent=2))
    print(f"Held-out topic: {PRIMARY_AUX_LABEL} ({PRIMARY_TRAIN_TOKENS:,} tokens)")
    if args.download_only:
        return
    run(make_filtered_config(args))


if __name__ == "__main__":
    main()

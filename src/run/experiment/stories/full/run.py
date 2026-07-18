"""Run the Phase 2 GRAM experiment on the complete SimpleStories corpus.

MacBook Pro (MPS)::

    python -m src.run.experiment.stories.full.run

Pass ``--experiment-id`` with an existing result directory name to resume it.
Download the public stories shards with ``--download-only`` if they are not already
present.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data.stories_utils import download_missing_stories, validate_stories_data
from src.model.config import RoutedModelConfig
from src.run.experiment.config import GetStoriesConfig
from src.run.main import run
from src.run.train.routed import UnorderedConfig


FULL_TRAIN_TOKENS = 547_853_673
PRIMARY_AUX_LABEL = "a-deadline-or-time-limit"
REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = REPO_ROOT / "src/data/stories"


def evaluation_profiles(aux_labels: list[str]) -> list[list[str]]:
    """Return the paper profiles followed by the Phase 3 comparison profiles."""
    aux_labels = sorted(aux_labels)
    if PRIMARY_AUX_LABEL not in aux_labels:
        raise ValueError(f"Primary auxiliary label is missing: {PRIMARY_AUX_LABEL}")

    paper_profiles = [["core"]] + [["core", label] for label in aux_labels]
    all_experts = ["core", *aux_labels]
    primary_ablated = [label for label in all_experts if label != PRIMARY_AUX_LABEL]
    return paper_profiles + [all_experts, primary_ablated]


def make_full_config(args: argparse.Namespace):
    """Build the dedicated seed/config for the required Phase 2 GRAM run."""
    config = GetStoriesConfig()

    # ``total=1.0`` consumes each side's complete corpus. Keeping core and
    # auxiliary budgets separate preserves the upstream routing distribution.
    config.data.core.method = "total"
    config.data.core.limit = 1.0
    config.data.aux.method = "total"
    config.data.aux.limit = 1.0

    config.run.res_root = REPO_ROOT / f"results/stories_phase2/seed_{args.seed}"
    config.run.seed = args.seed
    config.run.device = args.device
    config.run.dtype = args.dtype
    config.run.compile = args.compile
    config.run.cleanup_distributed = False
    config.run.s3_bucket = args.s3_bucket
    config.run.s3_prefix = args.s3_prefix
    # Match the eager FP32 MPS configuration validated by the Phase 1 smoke run:
    # 16 sequences per micro-batch x 8 accumulation steps = effective batch 128.
    config.run.target_effective_batch_size = -1
    config.run.micro_batch_size = 16
    config.run.accumulation_steps = 8
    config.run.nominal_token_budget = FULL_TRAIN_TOKENS
    config.run.model_shape = "paper"
    config.run.model_shape_note = "26M dense core / 32.57M GRAM paper configuration"
    if args.experiment_id is not None:
        config.run.experiment_id = args.experiment_id

    config.stages = [
        UnorderedConfig(
            model=RoutedModelConfig.from_base(
                config.model,
                arch="moe",
                core_param_prc=1.0,
                aux_param_prc=0.1,
            ),
            robust_prc=0.5,
            aux_route_prc=0.3,
            num_train_evals=0,
            num_checkpoints=args.num_checkpoints,
            do_elicit=False,
            lr=5e-3,
            equal_compute=True,
            acc_mode="heterogeneous",
            retain_targets=evaluation_profiles(config.data.aux.labels),
        )
    ]
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="mps")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model (disabled by default for the validated MPS path)",
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=2,
        help="Number of evenly spaced intermediate checkpoints; signals also trigger a save",
    )
    parser.add_argument(
        "--experiment-id",
        "--experiment_id",
        default=None,
        help="Existing timestamp/directory name to resume; a new timestamp is used by default",
    )
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Optional S3 prefix; use together with --s3-bucket",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and validate stories shards, then exit (single process only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.s3_bucket is None) != (args.s3_prefix is None):
        raise ValueError("--s3-bucket and --s3-prefix must be provided together")
    if args.num_checkpoints < -1 or args.num_checkpoints == 0:
        raise ValueError("--num-checkpoints must be -1 (disabled) or a positive integer")

    if args.download_only:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("--download-only must be run outside torchrun")
        downloaded = download_missing_stories(DATA_DIR)
        print(f"Downloaded {len(downloaded)} missing stories shards")

    data_summary = validate_stories_data(DATA_DIR)
    if data_summary["train_tokens"] != FULL_TRAIN_TOKENS:
        raise ValueError(
            f"Expected {FULL_TRAIN_TOKENS:,} training tokens, "
            f"found {data_summary['train_tokens']:,}"
        )
    print(json.dumps(data_summary, indent=2))
    if args.download_only:
        return

    run(make_full_config(args))


if __name__ == "__main__":
    main()

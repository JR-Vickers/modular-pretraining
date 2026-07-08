"""
Experiment templates and model sizing utilities.

Two experiment templates are provided as factory functions that return
pre-configured ``ExperimentConfig`` instances
"""

from __future__ import annotations

import json
from pathlib import Path

from src.run.util.config import (
    ExperimentConfig,
    DataConfig,
    DataLabelConfig,
    ModelConfig,
    RunConfig,
)
from src.model.utils import find_base_params


# --------------------------------------------------------------------------- #
# Experiment templates                                                         #
# --------------------------------------------------------------------------- #

root_dir = Path("src").absolute()

def GetStoriesConfig(num_aux: int = 4) -> ExperimentConfig:
    """Small-scale TinyStories experiment for fast iteration and debugging."""

    data_dir = root_dir / "data/stories"
    metadata = json.load(open(data_dir / "metadata.json"))
    all_labels = sorted(metadata["all"]["labels"])
    aux_labels = all_labels[:num_aux]

    config = ExperimentConfig(
        model=ModelConfig(
            ctx_len=256,
            vocab_size=4096,
            num_layers=8,
            num_heads=8,
            num_key_value=2,
            attn_bias=True,
            mlp_dim=512 * 4,
            eos_token_id=1,
        ),
        data=DataConfig(
            dirs=[data_dir],
            aux=DataLabelConfig(labels=aux_labels),
        ),
        run=RunConfig(
            warmup_prc=0.1,
            decay_prc=0.1,
            micro_batch_size=128, 
            target_effective_batch_size=128,
            accumulation_steps=1),
    )

    return config


def GetRealisticConfig(model_size: int = 100e6, align: int = 64) -> ExperimentConfig:
    """Production-scale experiment with FineWeb, code, and academic papers."""

    model_params = find_base_params(model_size, V=50304, align=align)
    # old_root_dir = Path("/workspace/gradient-routing/experiments/ICML-Codebase/src")

    config = ExperimentConfig(
        model=ModelConfig(**model_params, ctx_len=1024, eos_token_id=50256),
        data=DataConfig(
            dirs=[
                root_dir / "data/fineweb",
                root_dir / "data/code",
                root_dir / "data/papers",
            ],
            core=DataLabelConfig(
                labels=["fineweb", "code-other", "papers-other"],
                method="optimal",
                limit=1.0,
                dist={
                    "fineweb": 0.84,
                    "code-other": 0.15,
                    "papers-other": 0.01,
                },
            ),
            aux=DataLabelConfig(
                labels=["code-lisp", "papers-biology", "papers-nuclear", "papers-cyber"],
                method="optimal",
                limit=0.01,
                dist={
                    "code-lisp": 0.25,
                    "papers-biology": 0.25,
                    "papers-nuclear": 0.25,
                    "papers-cyber": 0.25,
                },
            ),
        ),
        #ICML
        # data=DataConfig(
        #     dirs=[
        #         old_root_dir / "data/fineweb",
        #         old_root_dir / "data/bigcode",
        #         old_root_dir / "data/arxiv",
        #     ],
        #     core=DataLabelConfig(
        #         labels=["fineweb"],
        #         method="optimal",
        #         limit=1.0,
        #         dist={
        #             "fineweb": 1.0,
        #         },
        #     ),
        #     aux=DataLabelConfig(
        #         labels=["bigcode", "biology", "nuclear", "cyber"],
        #         method="optimal",
        #         limit=0.05,
        #         dist={
        #             "bigcode": 0.25,
        #             "biology": 0.25,
        #             "nuclear": 0.25,
        #             "cyber": 0.25,
        #         },
        #     ),
        # ),
        run=RunConfig(),
    )

    return config
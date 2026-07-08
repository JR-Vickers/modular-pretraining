import json
from pathlib import Path
import torch

from src.model.config import RoutedModelConfig
from src.run.train.base import BaselineConfig, FilteringConfig
from src.run.train.coreftaux import CoreftauxConfig
from src.run.train.maxent import MaxentConfig
from src.run.train.routed import (
    OrderedConfig,
    UnorderedConfig,
)
from src.run.train.demix import DemixConfig
from src.run.experiment.config import GetStoriesConfig
from src.run.main import run

torch.cuda.empty_cache()

root_dir = Path("src").absolute()

for seed in [1, 2, 3]:

    config = GetStoriesConfig()
    LR = 5e-3
    BS = 128

    prefix = f"stories/seed_{seed}"
    res_root = root_dir.parent / "results" / prefix
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = prefix
    config.run.res_root = res_root

    config.run.target_effective_batch_size = BS
    config.run.seed = seed
    config.run.log_level = "DEBUG"
    config.run.compile = True
    config.run.find_unused_parameters = True
    config.run.cleanup_distributed = False

    base_model = config.model

    config.stages = [
        BaselineConfig(
            num_train_evals=200,
            do_elicit=False,
            lr=LR,
            acc_mode="heterogeneous",
        ),
        UnorderedConfig(
            model=RoutedModelConfig.from_base(base_model, arch="moe", core_param_prc=1.0, aux_param_prc=0.1),
            robust_prc=0.5,
            aux_route_prc=0.3,
            num_train_evals=0,
            do_elicit=True,
            lr=LR,
            equal_compute=True,
            acc_mode="heterogeneous",
        ),
        OrderedConfig(
            model=RoutedModelConfig.from_base(base_model, arch="lora", core_param_prc=1.0, aux_param_prc=0.1),
            core_aux_ratio=2.0,
            num_train_evals=0,
            do_elicit=True,
            lr=LR,
            equal_compute=True,
            acc_mode="heterogeneous",
        ),
        DemixConfig(
            model=RoutedModelConfig.from_base(base_model, arch="demix"),
            num_train_evals=0,
            do_elicit=True,
            lr=LR,
            acc_mode="heterogeneous",
        ),
        FilteringConfig(
            num_train_evals=0,
            do_elicit=True,
            lr=LR,
            acc_mode="heterogeneous",
        ),
        CoreftauxConfig(
            core_aux_ratio=2.0,
            num_train_evals=0,
            do_elicit=True,
            lr=LR,
            acc_mode="heterogeneous",
        )
    ]

    run(config)
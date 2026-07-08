# export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.arch.run

import json
from pathlib import Path
import torch
import torch._dynamo as dynamo

dynamo.config.optimize_ddp = "python_reducer"

from src.model.config import RoutedModelConfig
from src.run.train.base import BaselineConfig
from src.run.train.routed import (
    OrderedConfig,
    UnorderedConfig,
)
from src.run.experiment.config import GetRealisticConfig
from src.run.main import run
from src.run.experiment.common import make_param_str

torch.cuda.empty_cache()

root_dir = Path("src").absolute()

for n_params in [100e6]:
    for seed in [1, 2, 3]:

        config = GetRealisticConfig(n_params)
        power_laws_path =  root_dir.parent / "analysis" / "optimize" / "base" / "power_laws.json"
        power_laws = json.loads(power_laws_path.read_text())
        LR = power_laws["lr"]["coef"] * (n_params ** power_laws["lr"]["exp"])
        BS = round(power_laws["bs"]["coef"] * (n_params ** power_laws["bs"]["exp"]))

        prefix = f"arch/{make_param_str(n_params)}/seed_{seed}"
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
            OrderedConfig(
                model=RoutedModelConfig.from_base(base_model, arch="lora", core_param_prc=1.0, aux_param_prc=0.1),
                core_aux_ratio=1.0,
                num_train_evals=0,
                do_elicit=True,
                lr=LR,
                equal_compute=True,
                eval_arbsub=False,
                acc_mode="heterogeneous",
            ),
            OrderedConfig(
                model=RoutedModelConfig.from_base(base_model, arch="moe", core_param_prc=1.0, aux_param_prc=0.1),
                core_aux_ratio=1.0,
                num_train_evals=0,
                do_elicit=True,
                lr=LR,
                equal_compute=True,
                eval_arbsub=False,
                acc_mode="heterogeneous",
            ),
            UnorderedConfig(
                model=RoutedModelConfig.from_base(base_model, arch="lora", core_param_prc=1.0, aux_param_prc=0.1),
                robust_prc=0.2,
                aux_route_prc=0.5,
                num_train_evals=0,
                do_elicit=True,
                lr=LR,
                equal_compute=True,
                eval_arbsub=False,
                acc_mode="heterogeneous",
            ),
            UnorderedConfig(
                model=RoutedModelConfig.from_base(base_model, arch="moe", core_param_prc=1.0, aux_param_prc=0.1),
                robust_prc=0.2,
                aux_route_prc=0.5,
                num_train_evals=0,
                do_elicit=True,
                lr=LR,
                equal_compute=True,
                eval_arbsub=False,
                acc_mode="heterogeneous",
            ),
        ]

        run(config)
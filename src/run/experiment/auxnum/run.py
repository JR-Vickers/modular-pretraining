import argparse
from pathlib import Path
import torch

from src.model.config import RoutedModelConfig
from src.run.train.base import BaselineConfig
from src.run.train.routed import UnorderedConfig
from src.run.experiment.config import GetStoriesConfig
from src.run.main import run

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                    help="Which seeds to run (default: 1 2 3).")
args = parser.parse_args()

root_dir = Path("src").absolute()

for seed in args.seeds:

    for num_aux in [4, 8, 12, 16, 20]:

        config = GetStoriesConfig(num_aux=num_aux)

        config.data.core.method = "dataset"
        config.data.core.limit = 0.8
        config.data.aux.method = "dataset"
        config.data.aux.limit = 0.2

        LR = 5e-3
        BS = 128

        prefix = f"auxnum/v3/num_aux_{num_aux}/seed_{seed}"
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
                model=RoutedModelConfig.from_base(
                    base_model, 
                    arch="moe", 
                    core_param_prc=1.0, 
                    aux_param_prc=0.1),
                robust_prc=0.5,
                aux_route_prc=0.2,
                do_elicit=True,
                lr=LR,
                equal_compute=True,
                acc_mode="heterogeneous",
            )
        ]

        run(config)
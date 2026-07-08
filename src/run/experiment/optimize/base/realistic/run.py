import argparse
import re
from pathlib import Path

from src.run.util.config import ExperimentConfig
from src.run.train.base import BaselineConfig
from src.run.util.distributed import cleanup_distributed
from src.run.experiment.config import GetRealisticConfig
from src.run.experiment.common import ROOT_DIR, parse_model_size, make_param_str
from src.run.main import run as run_single

def make_config(
    model_size: int,
    lr: float,
    effective_batch_size: int,
    seed: int,
    res_root: Path,
) -> ExperimentConfig:

    config = GetRealisticConfig(model_size)
    stage_config = BaselineConfig(lr=lr, num_train_evals=0, do_elicit=False)
    config.stages = [stage_config]
    config.run.target_effective_batch_size = effective_batch_size
    config.run.seed = seed
    config.run.cleanup_distributed = False
    config.run.res_root = res_root
    config.run.log_level = "DEBUG"
    config.run.compile = True
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = f"optimize/base/realistic/{make_param_str(model_size)}/seed_{seed}"
    
    return config 

def run(
    n_params: int,
    seed: int,
    lr: float,
    eff_bs: int,
) -> None:

    param_str = make_param_str(n_params)
    print(f"[optimize.base.realistic] N={n_params} ({param_str}) lr={lr:.3e} eff_bs={eff_bs}")
    res_root = ROOT_DIR / "optimize" / "base" / "realistic" / f"{param_str}" / f"seed_{seed}"
    config = make_config(n_params, lr, eff_bs, seed, res_root)
    try:
        run_single(config)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        help="Model size (e.g. 50M, 2B).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eff_bs", type=int, default=None)
    args = parser.parse_args()
    run(parse_model_size(args.model_size.upper()), args.seed, args.lr, args.eff_bs)
import argparse
from pathlib import Path
from typing import Literal

from src.run.util.config import ExperimentConfig
from src.run.train.base import BaselineConfig
from src.run.experiment.config import GetRealisticConfig
from src.run.experiment.common import ROOT_DIR, make_param_str, get_bs, get_lr
from src.run.main import run as run_single


def make_config(
    model_size: int,
    lr: float,
    effective_batch_size: int,
    seed: int,
    res_root: Path,
    cleanup_distributed: bool = True,
    acc_mode: Literal["uniform", "heterogeneous"] = "uniform",
) -> ExperimentConfig:

    align = 64 if model_size < 5e9 else 32
    config = GetRealisticConfig(model_size, align=align)
    stage_config = BaselineConfig(
        lr=lr,
        num_checkpoints=100 if model_size > 800_000_000 else -1,
        num_train_evals=100,
        do_elicit=False,
        acc_mode=acc_mode,
        label_prc=1.0, 
    )
    config.stages = [stage_config]
    config.run.target_effective_batch_size = effective_batch_size
    config.run.seed = seed
    config.run.cleanup_distributed = cleanup_distributed
    config.run.res_root = res_root
    config.run.log_level = "DEBUG"
    config.run.compile = True
    config.run.find_unused_parameters = False
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = f"scaling/realistic/base/{make_param_str(model_size)}/seed_{seed}"

    # config.run.accumulation_steps = 6
    # config.run.micro_batch_size = 2
    # config.run.num_gpus = 32
    # config.run.effective_batch_size = 6 * 2 * 32
    # config.run.target_effective_batch_size = -1

    config.run.accumulation_steps = 6 * 4
    config.run.micro_batch_size = 2
    config.run.num_gpus = 8
    config.run.effective_batch_size = 6 * 4 * 2 * 8
    config.run.target_effective_batch_size = -1
    
    return config


def run(
    n_params: int,
    seed: int,
    experiment_id: str | None = None,
    cleanup_distributed: bool = True,
    acc_mode: Literal["uniform", "heterogeneous"] = "uniform",
) -> None:

    param_str = make_param_str(n_params)
    lr = get_lr(n_params)
    eff_bs = get_bs(n_params)
    print(f"[scaling.realistic.base] N={n_params} ({param_str}) lr={lr:.3e} eff_bs={eff_bs}")
    res_root = ROOT_DIR / "scaling" / "realistic" / "base" / param_str / f"seed_{seed}"
    config = make_config(n_params, lr, eff_bs, seed, res_root, cleanup_distributed, acc_mode)
    if experiment_id is not None:
        config.run.experiment_id = experiment_id
    run_single(config)


if __name__ == "__main__":

    run(200e6, 5, experiment_id="acc_mode_heterogeneous_via_old_uniform_script_with_all_label_8gpu", cleanup_distributed=False, acc_mode="heterogeneous")


    # for acc_mode in ["uniform", "heterogeneous"]:
    #     run(
    #         n_params=200e6,
    #         seed=5,
    #         experiment_id=f"acc_mode_{acc_mode}",
    #         cleanup_distributed=False,
    #         acc_mode=acc_mode,
    #     )
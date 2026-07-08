import argparse
from pathlib import Path
from typing import Literal

from src.run.util.config import ExperimentConfig
from src.run.experiment.config import GetRealisticConfig
from src.run.train.routed import OrderedConfig
from src.model.config import RoutedModelConfig
from src.run.experiment.common import ROOT_DIR, make_param_str, get_bs, get_lr
from src.run.main import run as run_single

CORE_PARAM_PRC = 1.0
AUX_PARAM_PRC = 0.1
# AUX_FACTOR = {"code-lisp":2.0, "papers-biology":1.0, "papers-nuclear":1.0, "papers-cyber":1.0}
AUX_FACTOR = 1.0
CORE_AUX_RATIO = 1.0

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
    stage_config = OrderedConfig(
        model=RoutedModelConfig.from_base(
            config.model, 
            arch="lora", 
            core_param_prc=CORE_PARAM_PRC,
            aux_param_prc=AUX_PARAM_PRC),
        aux_factor=AUX_FACTOR,
        core_aux_ratio=CORE_AUX_RATIO,
        lr=lr,
        num_checkpoints=100 if model_size > 800_000_000 else -1,
        equal_compute=True,
        num_train_evals=0,
        do_elicit=True,
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
    config.run.find_unused_parameters = True
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = f"scaling/realistic/lora/{make_param_str(model_size)}/seed_{seed}"

    config.run.accumulation_steps = 6
    config.run.micro_batch_size = 2
    config.run.target_effective_batch_size = -1
    config.run.num_gpus = 32
    config.run.effective_batch_size = 32 * 6 * 2
    
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
    print(f"[scaling.realistic.lora] N={n_params} ({param_str}) lr={lr:.3e} eff_bs={eff_bs}")
    res_root = ROOT_DIR / "scaling" / "realistic" / "lora" / param_str / f"seed_{seed}"
    config = make_config(n_params, lr, eff_bs, seed, res_root, cleanup_distributed, acc_mode)
    if experiment_id is not None:
        config.run.experiment_id = experiment_id
    run_single(config)


if __name__ == "__main__":


    for acc_mode in ["uniform", "heterogeneous"]:
        run(
            n_params=200e6,
            seed=5,
            experiment_id=f"acc_mode_{acc_mode}_factor_1.0",
            cleanup_distributed=False,
            acc_mode=acc_mode,
        )
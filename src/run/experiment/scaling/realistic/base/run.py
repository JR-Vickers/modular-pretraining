import argparse
from pathlib import Path

from src.run.util.config import ExperimentConfig
from src.run.train.base import BaselineConfig
from src.run.experiment.config import GetRealisticConfig
from src.run.experiment.common import ROOT_DIR, parse_model_size, make_param_str, get_bs, get_lr
from src.run.main import run as run_single


def make_config(
    model_size: int,
    lr: float,
    effective_batch_size: int,
    seed: int,
    res_root: Path,
    cleanup_distributed: bool = True,
) -> ExperimentConfig:

    align = 64 if model_size < 5e9 else 32
    config = GetRealisticConfig(model_size, align=align)
    stage_config = BaselineConfig(
        lr=lr,
        num_checkpoints=100 if model_size > 800_000_000 else -1,
        num_train_evals=100,
        do_elicit=True,
        acc_mode="heterogeneous",
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
    
    return config


def run(
    n_params: int,
    seed: int,
    experiment_id: str | None = None,
    cleanup_distributed: bool = True,
) -> None:

    param_str = make_param_str(n_params)
    lr = get_lr(n_params)
    eff_bs = get_bs(n_params)
    print(f"[scaling.realistic.base] N={n_params} ({param_str}) lr={lr:.3e} eff_bs={eff_bs}")
    res_root = ROOT_DIR / "scaling" / "realistic" / "base" / param_str / f"seed_{seed}"
    config = make_config(n_params, lr, eff_bs, seed, res_root, cleanup_distributed)
    if experiment_id is not None:
        config.run.experiment_id = experiment_id
    run_single(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        help="Model size (e.g. 50M, 2B).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--experiment_id", type=str, default=None,
        help="Optional experiment ID (for resuming). Defaults to timestamp.")

    args = parser.parse_args()
    run(
        parse_model_size(args.model_size.upper()), 
        args.seed, 
        args.experiment_id,
    )
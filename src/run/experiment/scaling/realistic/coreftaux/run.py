import argparse
from pathlib import Path

from src.run.util.config import ExperimentConfig
from src.run.train.coreftaux import CoreftauxConfig
from src.run.experiment.config import GetRealisticConfig
from src.run.experiment.common import ROOT_DIR, parse_model_size, make_param_str, get_bs, get_lr
from src.run.main import run as run_single

AUX_FACTOR = 1.0
CORE_AUX_RATIO = 4.0
FT_LR_FACTOR = 0.2

def make_config(
    model_size: int,
    lr: float,
    effective_batch_size: int,
    seed: int,
    res_root: Path,
    cleanup_distributed: bool,
    s3_prefix: str,
) -> ExperimentConfig:

    align = 64 if model_size < 5e9 else 32
    config = GetRealisticConfig(model_size, align=align)
    stage_config = CoreftauxConfig(
        aux_factor=AUX_FACTOR,
        core_aux_ratio=CORE_AUX_RATIO,
        lr=lr,
        num_checkpoints=100 if model_size > 800_000_000 else -1,
        num_train_evals=0,
        do_elicit=True,
        acc_mode="heterogeneous" if model_size >= 5e9 else "uniform",
        label_prc=1.0,
        ft_lr_factor=FT_LR_FACTOR,
    )
    config.stages = [stage_config]
    config.run.target_effective_batch_size = effective_batch_size
    config.run.seed = seed
    config.run.cleanup_distributed = cleanup_distributed
    config.run.res_root = res_root
    config.run.log_level = "DEBUG"
    config.run.compile = True
    config.run.s3_bucket = "ae-gradient-routing-results"
    config.run.s3_prefix = s3_prefix
    config.run.find_unused_parameters = False

    return config


def run(
    n_params: int,
    seed: int,
    cleanup_distributed: bool = True,
    experiment_id: str | None = None,
) -> None:

    param_str = make_param_str(n_params)
    lr = get_lr(n_params)
    eff_bs = get_bs(n_params)
    print(f"[scaling.realistic.coreftaux] N={n_params} ({param_str}) lr={lr:.3e} eff_bs={eff_bs} aux_factor={AUX_FACTOR} core_aux_ratio={CORE_AUX_RATIO} ft_lr_factor={FT_LR_FACTOR}")

    base_res = ROOT_DIR / "scaling" / "realistic" / "coreftaux" / param_str / f"seed_{seed}"
    base_s3 = f"scaling/realistic/coreftaux/{param_str}/seed_{seed}"
    res_root = base_res
    s3_prefix = base_s3

    config = make_config(
        n_params, lr, eff_bs, seed, res_root, cleanup_distributed,
        s3_prefix=s3_prefix,
    )
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
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Resume an existing run by reusing its timestamp dir. If not set, a fresh timestamp is used.",
    )

    args = parser.parse_args()
    run(
        parse_model_size(args.model_size.upper()),
        args.seed,
        core_aux_ratio=args.core_aux_ratio,
        experiment_id=args.experiment_id,
    )

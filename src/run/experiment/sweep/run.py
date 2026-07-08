from pathlib import Path
import torch, json

from src.run.train.routed import UnorderedConfig
from src.model.config import RoutedModelConfig
from src.run.experiment.config import GetRealisticConfig
from src.run.experiment.common import make_param_str
from src.run.main import run
from src.run.train.base import BaselineConfig
from src.run.train.routed import OrderedConfig, UnorderedConfig

torch.cuda.empty_cache()

#GRAM HPARAMS
ROBUST_PRC = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
AUX_ROUTE_PRCS = [0.0, 0.25, 0.5, 0.75, 1.0]

#FT-LORA HPARAMS
RATIOS = [1/4, 1/2, 1/1, 2/1, 4/1]

root_dir = Path("src").absolute()

for n_params in [200e6]:
    for seed in [1, 2, 3]:

        config = GetRealisticConfig(n_params)
        power_laws_path =  root_dir.parent / "analysis" / "optimize" / "base" / "power_laws.json"
        power_laws = json.loads(power_laws_path.read_text())
        LR = power_laws["lr"]["coef"] * (n_params ** power_laws["lr"]["exp"])
        BS = round(power_laws["bs"]["coef"] * (n_params ** power_laws["bs"]["exp"]))

        prefix = f"sweep/realistic/{make_param_str(n_params)}/seed_{seed}"
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
            )
        ]

        for robust_prc in ROBUST_PRC:
            new_config = UnorderedConfig(
                model=RoutedModelConfig.from_base(
                    base_model, 
                    arch="moe", 
                    core_param_prc=1.0, 
                    aux_param_prc=0.1),
                robust_prc=robust_prc,
                aux_route_prc=0.0,
                do_elicit=True,
                lr=LR,
                equal_compute=True,
                acc_mode="heterogeneous",
            )
            config.stages.append(new_config)

        for aux_route_prc in AUX_ROUTE_PRCS:
            new_config = UnorderedConfig(
                model=RoutedModelConfig.from_base(
                    base_model, 
                    arch="moe", 
                    core_param_prc=1.0,
                    aux_param_prc=0.1),
                robust_prc=0.5,
                aux_route_prc=aux_route_prc,
                do_elicit=True,
                lr=LR,
                equal_compute=True,
                acc_mode="heterogeneous",
            )
            config.stages.append(new_config)

        for ratio in RATIOS:
            new_config = OrderedConfig(
                model=RoutedModelConfig.from_base(
                    base_model, 
                    arch="lora", 
                    core_param_prc=1.0,
                    aux_param_prc=0.1),
                core_aux_ratio=ratio,
                aux_factor= 2 / (ratio + 1), #enforces constant FT phase length of aux_len * 2
                do_elicit=True,
                lr=LR,
                equal_compute=True,
                acc_mode="heterogeneous",
            )
            config.stages.append(new_config)

        run(config)
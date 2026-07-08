# export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.scaling.realistic.maxent.run_all
from src.run.experiment.scaling.realistic.maxent.run import run


def run_all():

    seed2id = {
        1: "20260422084509038162",
        2: "20260504072224060283",
        3: "20260504213914888636",
    }

    for seed in [1, 2, 3]:
        for n_params in [800e6]:
            exp_id = seed2id[seed]
            run(
                n_params=n_params,
                seed=seed,
                baseline_experiment_id=exp_id,
                cleanup_distributed=False,
            )


if __name__ == "__main__":
    run_all()

# export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.scaling.realistic.coreftaux.run_all
from src.run.experiment.scaling.realistic.coreftaux.run import run

def run_all():

    for seed in [1, 2, 3]:
        for n_params in [800e6]:
            run(n_params, seed, cleanup_distributed=False)


if __name__ == "__main__":
    run_all()
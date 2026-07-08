# export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.scaling.realistic.base.run_all
from src.run.experiment.scaling.realistic.base.run import run

def run_all():
    for seed in [1, 2, 3]:
        for n_params in [50e6, 100e6, 200e6, 400e6, 800e6, 2000e6]:
            if n_params != 2000e6 and seed != 1: continue
            run(n_params, seed, None, cleanup_distributed=False)

if __name__ == "__main__":
    run_all()
# export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.scaling.realistic.lora.run_all
from src.run.experiment.scaling.realistic.lora.run import run

def run_all():
    
    for seed in [1, 2, 3]:
        for n_params in [50e6, 100e6, 200e6, 400e6, 800e6]:
            run(n_params, seed, None, cleanup_distributed=False)

if __name__ == "__main__":
    run(800e6, 3, None, cleanup_distributed=False)
    # run_all()
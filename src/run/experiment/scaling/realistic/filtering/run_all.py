# export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.experiment.scaling.realistic.filtering.run_all
from src.run.experiment.scaling.realistic.filtering.run import run

# retain_targets = [
#     ["core", "papers-biology"],
#     ["core", "code-lisp"],
#     ["core", "papers-cyber"],
#     ["core", "papers-nuclear"],
#     ["core"],
# ]

def run_all():

    # Only the two runs left after the 2026-05-18 slurmctld crash.
    # papers-nuclear got to ~22000/23498 (94%) before the crash; core never started.
    # No intermediate checkpoints were saved (num_checkpoints=-1 at 800M), so both
    # need a clean restart.
    run(800e6, 2, None, [["core", "papers-nuclear"]], cleanup_distributed=False)
    run(800e6, 2, None, [["core"]],                   cleanup_distributed=False)

    # --- already completed; preserved here as a record of the original sweep ---
    # run(800e6, 3, None, [["core", "papers-cyber"]],   cleanup_distributed=False)
    # run(800e6, 3, None, [["core", "papers-nuclear"]], cleanup_distributed=False)
    # run(800e6, 3, None, [["core"]],                   cleanup_distributed=False)
    #
    # for n_params in [50e6, 100e6, 200e6, 400e6, 800e6]:
    #     if n_params != 800e6:
    #         run(n_params, 2, None, [["core", "papers-biology"]], cleanup_distributed=False)
    #     else:
    #         for targets in retain_targets:
    #             run(n_params, 2, None, [targets], cleanup_distributed=False)


if __name__ == "__main__":
    run_all()
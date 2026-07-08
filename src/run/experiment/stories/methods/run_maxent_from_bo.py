"""
Run MaxEnt unlearning for a SimpleStories seed, using the per-retain-target
hyperparameters found by the maxent_stories Bayesian optimisation.

The seed is read from the MAXENT_SEED env var (default 1). For that seed it:
  * discovers the saved stories baseline (latest complete run dir) and hardlinks
    it in so the baseline stage is skipped + loaded (not retrained);
  * reads the BO-optimal (lr, alpha_retain) for each of the 5 retain targets
    straight from results/optimize/maxent_stories/seed_N/<key>/bo_summary.json
    (no hand-transcription);
  * runs one maxent stage per retain target with its own optimal pair.

Unlearned checkpoints are saved per retain target (run_unlearning persists them)
under results/stories_maxent_seedN/seed_N/maxent_NN/<retain_key>/checkpoint.pth

Single GPU:
    MAXENT_SEED=2 uv run torchrun --standalone --nproc_per_node=1 \
        -m src.run.experiment.stories.methods.run_maxent_from_bo
"""
import json
import os
import shutil
from pathlib import Path

import torch

from src.run.train.base import BaselineConfig
from src.run.train.maxent import MaxentConfig
from src.run.experiment.config import GetStoriesConfig
from src.run.util.tools import get_timestamp
from src.run.main import run

torch.cuda.empty_cache()

SEED = int(os.environ.get("MAXENT_SEED", "1"))
BASELINE_LR = 5e-3            # stories baseline lr (acc_mode=heterogeneous)
ACC_MODE = "heterogeneous"
STEPS = 2000                  # maxent steps per stage (matches the BO)

# Stories aux labels = sorted(all_labels)[:4]; retain targets = core-only + each.
RETAIN_TARGETS = [
    ["core"],
    ["core", "a-deadline-or-time-limit"],
    ["core", "alien-encounters"],
    ["core", "bygone-eras"],
    ["core", "cultural-traditions"],
]


def _retain_key(rt):
    return "_".join(rt)


root_dir = Path("src").absolute()
repo_root = root_dir.parent

# --- discover the saved baseline for this seed (latest complete run dir) ------
seed_results = repo_root / "results" / "stories" / f"seed_{SEED}"
cands = [
    d for d in seed_results.iterdir()
    if d.is_dir()
    and (d / "stats.jsonl").exists()
    and (d / "baseline" / "checkpoint.pth").exists()
    and (d / "baseline" / "losses.pkl").exists()
]
assert cands, f"no complete stories baseline under {seed_results}"
BASELINE_SRC = max(cands, key=lambda d: d.name) / "baseline"

# --- read BO-optimal hyperparameters per retain target from disk --------------
bo_root = repo_root / "results" / "optimize" / "maxent_stories" / f"seed_{SEED}"
best = {}
for rt in RETAIN_TARGETS:
    summ = bo_root / _retain_key(rt) / "bo_summary.json"
    assert summ.exists(), f"missing BO summary {summ}"
    best[tuple(rt)] = json.loads(summ.read_text())["best_params"]

config = GetStoriesConfig()
# Write under the normal stories results tree, as its own timestamped run dir
# (results/stories/seed_N/<ts>/), like any other stories experiment.
config.run.res_root = repo_root / "results" / "stories" / f"seed_{SEED}"
config.run.experiment_id = get_timestamp()
config.run.seed = SEED
config.run.log_level = "DEBUG"
config.run.compile = True
config.run.find_unused_parameters = True
config.run.cleanup_distributed = True
config.run.s3_bucket = None
config.run.s3_prefix = None

# --- pre-stage the saved baseline (hardlink) so it is skipped + loaded --------
if os.environ.get("RANK", "0") == "0":
    bdst = config.run.res_root / config.run.experiment_id / "baseline"
    bdst.mkdir(parents=True, exist_ok=True)
    ckpt_dst = bdst / "checkpoint.pth"
    if not ckpt_dst.exists():
        os.link(BASELINE_SRC / "checkpoint.pth", ckpt_dst)
    for name in ("stage.json", "losses.pkl"):
        if not (bdst / name).exists():
            shutil.copy(BASELINE_SRC / name, bdst / name)

baseline_stage = BaselineConfig(
    lr=BASELINE_LR,
    acc_mode=ACC_MODE,
    num_train_evals=0,
    do_elicit=False,
)

maxent_stages = [
    MaxentConfig(
        lr=best[tuple(rt)]["lr"],
        alpha_retain=best[tuple(rt)]["alpha_retain"],
        steps=STEPS,
        acc_mode=ACC_MODE,
        num_train_evals=0,
        do_eval=True,
        do_elicit=True,
        retain_targets=[rt],
    )
    for rt in RETAIN_TARGETS
]

config.stages = [baseline_stage] + maxent_stages

print(f"[run_maxent_from_bo] seed={SEED} baseline_src={BASELINE_SRC.parent.name} "
      f"params={ {k[-1] if len(k)>1 else 'core': (round(v['lr'],6), round(v['alpha_retain'],1)) for k,v in best.items()} }")

run(config)

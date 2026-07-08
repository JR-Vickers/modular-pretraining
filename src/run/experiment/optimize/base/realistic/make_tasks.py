#!/usr/bin/env python3
"""Generate a tasks.csv with the full grid (Cartesian product) of LR × BS per model size.

Usage:
    python -m src.run.experiment.optimize.base.realistic.make_tasks
    python -m src.run.experiment.optimize.base.realistic.make_tasks -o /path/to/tasks.csv
"""

import argparse
import csv
from itertools import product
from pathlib import Path

GRIDS: dict[int, dict] = {
    50:  {"lr": [4e-4, 8e-4, 1.6e-3], "bs": [32, 64, 128, 256]},
    100: {"lr": [4e-4, 8e-4, 1.6e-3], "bs": [32, 64, 128, 256]},
    200: {"lr": [2e-4, 4e-4, 8e-4, 1.6e-3], "bs": [64, 128, 256, 512]},
    400: {"lr": [2e-4, 4e-4, 8e-4], "bs": [128, 256, 512, 1024]},
}

MODEL_SIZES = [50, 100, 200, 400]
SEEDs = [1, 2, 3]

COLUMNS = ["model_size", "seed", "trial_num", "lr", "batch_size", "status"]


def generate_tasks() -> list[dict]:
    rows: list[dict] = []
    for model_size in MODEL_SIZES:
        for seed in SEEDs:
            grid = GRIDS[model_size]
            trial_num = 0
            for lr, bs in product(grid["lr"], grid["bs"]):
                trial_num += 1
                rows.append({
                    "model_size": model_size,
                    "seed": seed,
                    "trial_num": trial_num,
                    "lr": lr,
                    "batch_size": bs,
                    "status": "pending",
                })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate base grid task CSV")
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path(__file__).resolve().parents[6] / "results" / "optimize" / "base" / "realistic" / "tasks.csv",
    )
    args = parser.parse_args()

    rows = generate_tasks()
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tasks to {args.output}")

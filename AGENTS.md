# Repository Guidelines

Begin each session by reading `docs/PLAN.md` in order to gain context on the project.

## Project Structure & Module Organization

- `src/model/` contains the dense Transformer and GRAM, LoRA, and DEMix variants.
- `src/data/` contains dataset-preparation scripts and metadata; generated tokenized shards are external.
- `src/run/` contains the staged training/evaluation pipeline, method runners, experiment entry points, and distributed utilities.
- `analysis/` contains aggregation, plotting, and LaTeX-generation scripts for paper figures.
- `results/` stores lightweight metrics and run metadata; `docs/` stores project notes and plans.

Keep new experiments under `src/run/experiment/<name>/` and put their analysis under `analysis/<name>/`.

## Build, Test, and Development Commands

Begin each session by activating the virtual environment:

```bash
source .venv/bin/activate
```

Run a distributed experiment with `OMP_NUM_THREADS=16 torchrun --nproc_per_node=8 -m src.run.experiment.partial.run`, or launch the general pipeline with `torchrun --nproc_per_node=8 -m src.run.main`. Use `python -m ...` for analysis modules, for example `uv run python -m analysis.partial.plot_full`. Training requires suitable GPUs, external datasets/checkpoints, and a root `.env` containing `HF_TOKEN`.

## Coding Style & Naming Conventions

Use Python 3 type hints, four-space indentation, descriptive `snake_case` names for functions and variables, and `PascalCase` for classes. Follow the surrounding module’s import and CLI patterns; keep executable scripts guarded by `if __name__ == "__main__":`. Prefer `pathlib.Path` and explicit argparse options. No repository-wide formatter or linter is configured, so keep diffs focused and match nearby style.

## Testing Guidelines

No automated tests or coverage threshold are currently checked in. For changes, run the affected module with a small/local dataset where possible, then perform a syntax/import smoke check such as `python -m compileall src analysis`. For plotting changes, verify the expected `.png`, `.tex`, or summary output is regenerated without modifying unrelated result files.

## Commit & Pull Request Guidelines

Existing commits use short, imperative summaries (for example, `Delete PDF`); continue that convention and keep each commit focused. Pull requests should explain the experiment or analysis change, identify affected paths and commands, link any relevant issue or paper figure, and include representative plots or metric comparisons when outputs change. Do not commit secrets, checkpoints, tokenized data, or other large model artifacts; `.env`, `*.pth`, `*.bin`, `*.pt`, and `*.safetensors` are ignored by design.

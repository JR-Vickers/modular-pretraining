# Modular Pretraining Enables Access Control

**Project page: [modularpretraining.com](https://modularpretraining.com)**

Code, analysis, and results for the paper **“Modular Pretraining Enables Access Control”**, which introduces **GRAM** (Gradient-Routed Auxiliary Modules): a pre-training method that produces many capability profiles from a *single* training run.

> AI developers face a dual-use dilemma: a capability that helps one user can harm another. A gold standard for access control is to serve separately trained, data-filtered models with different capability profiles — but training and deploying many models is prohibitively expensive. GRAM augments the MLP blocks of a dense Transformer with small auxiliary modules and uses **gradient routing** to localize each dual-use capability into its own module. Ablating a module at inference removes its capability, approximating a model trained on filtered data, and the ablated model resists recovery under adversarial finetuning far better than post-hoc unlearning. Training cost is independent of the number of supported capability profiles — a **5× reduction over data filtering** in our 5-profile setting — and a Chinchilla-optimal scaling analysis from **50M to 5B parameters** shows GRAM closely tracks data filtering, with the gap on *removed* capabilities widening with scale.

## Method in one paragraph

Each GRAM layer contains one always-active **core MLP** (the same size as the baseline's MLP) and `N−1` small **auxiliary modules**, one per dual-use domain. During training, gradient routing controls which parameters are updated based on the data label of the batch — auxiliary batches always update their module and update the core with probability `p_as` (*auxiliary spread*); core batches always update the core and, with probability `p_cr` (*core robustness*), also activate a random auxiliary module so core performance stays robust to module ablation. There is **no learned router and no per-token routing**. At inference, a capability profile `S` is served by activating the core plus the modules `i ∈ S` and ablating the rest.

## Repository layout

```
src/                          Training + evaluation code
  data/                       Dataset preparation (→ tokenized .bin shards, uploaded to HF)
    prep_fineweb.py           Core web text  (HuggingFaceFW/fineweb-edu, sample-100BT)
    prep_code.py              Core + Lisp aux code (The Stack)
    prep_papers.py            arXiv / Europe PMC / OSTI papers (dual-use domains)
    prep_stories.py           Simple Stories (synthetic)
  model/                      Model architectures
    base.py                   Dense Transformer (baseline / filtering)
    moe.py                    GRAM auxiliary-module model (gradient-routed, no router)
    lora.py                   LoRA adapters (FT-LoRA, GRAM-LoRA)
    demix.py                  DEMix domain experts (comparison)
  run/
    main.py                   Core pipeline: runs a sequence of stages, auto-resume, preemption-safe
    train/                    Per-method training runners
      base.py                 baseline + filtering
      routed.py               GRAM (moe / lora / demix)
      coreftaux.py            branched finetuning (FT-Full, FT-LoRA)
      maxent.py, rmu.py, ascent.py   post-hoc unlearning
      finetune.py             adversarial elicitation (elicited-forget metric)
    experiment/               One subpackage per paper experiment (defines stages, launches main.run)
    util/                     DDP, dataloaders, checkpointing, S3, preemption
analysis/                     Plotting + aggregation → the paper's TikZ/PNG figures
results/                      Per-run metrics (losses, configs, eval outputs); checkpoints excluded
```

## Experiments → paper figures

Each experiment is a subpackage under `src/run/experiment/`; its metrics land in `results/<name>/` and are plotted by `analysis/<name>/`.

| Experiment | Paper | Description | Model size |
|---|---|---|---|
| `stories` | Fig. 2, Table 1 | Simple Stories: GRAM approximates 5 data-filtered models | 26M |
| `auxnum` | Fig. 3 | Scaling the number of auxiliary categories (4 → 20) | 26M |
| `metric/realistic` | Fig. 4 | Realistic dual-use setting (virology, cyber, nuclear, code) | 800M |
| `arbsub` | Fig. 5 | Arbitrary capability subsets / composability | 800M |
| `partial` | Fig. 6 | Partial labeling (only 50% of data labeled) | 400M |
| `scaling/realistic` | Fig. 7 | Compute-optimal scaling of GRAM vs. filtering vs. FT-LoRA | 50M–5B |
| `arch` | Appendix (architecture ablation) | MLP vs. LoRA module architecture | 100M |
| `sweep` | Appendix (hyperparameters) | `p_as`, `p_cr`, and FT-LoRA core:aux ratio sweeps | 26M |
| `titration` | Appendix (titration) | Smoothly scaling a module's forward weight | — |
| `optimize` | Appendix (LR/BS scaling) | Learning-rate / batch-size power-law fits | 50M–400M |
| `accumulation` | Appendix (accumulation) | Uniform vs. heterogeneous gradient accumulation | 200M |

**Methods compared:** Baseline (dense), Filtering (gold standard), GRAM, FT-LoRA, FT-Full, MaxEnt (post-hoc unlearning), and DEMix.

## Compute Ratio metric

Performance is reported as **compute ratio**: the fraction of baseline training compute needed to reach a given validation loss. For a model `M` and dataset `D_i`, `CR(M, D_i) = L_i^{-1}(loss(M, D_i)) / L_i^{-1}(loss(baseline, D_i))`, where `L_i` is the baseline's fitted power-law learning curve. `CR = 1` matches the baseline; higher is better on retained data, lower is better on forgotten data. *Elicited forget* re-measures forget CR after adversarial finetuning on a small forget sample.

## Setup

There is no packaged manifest; install the dependencies into a fresh environment (uv recommended):

```bash
uv venv && source .venv/bin/activate
uv pip install torch transformers datasets huggingface_hub numpy scipy \
               matplotlib tqdm python-dotenv     # analysis also uses matplot2tikz
```

Create a `.env` at the **repository root** (git-ignored) with a HuggingFace token for data access:

```bash
echo "HF_TOKEN=hf_..." > .env
```

Multi-GPU training uses `torchrun` with DistributedDataParallel (8× GPUs by default).

## Running

The core pipeline runs a list of stages (train → evaluate → elicit) defined by each experiment:

```bash
# Run a full experiment (e.g. the partial-labeling experiment)
export OMP_NUM_THREADS=16
torchrun --nproc_per_node=8 -m src.run.experiment.partial.run

# Or drive the pipeline directly
torchrun --nproc_per_node=8 -m src.run.main
```

Training is **preemption-safe**: on `SIGTERM` (e.g. Slurm preemption) a checkpoint is saved and the run auto-resumes from the latest stage/checkpoint on requeue.

## Reproducing figures

The `analysis/` subpackages read the per-run metrics in `results/` and emit the paper's figures. Aggregation is two-stage (mean within seed, then mean + t-based 90% CI across `N=3` seeds):

```bash
uv run python -m analysis.partial.plot_full     # partial-labeling figure
uv run python -m analysis.realistic.plot         # realistic dual-use figure
uv run python -m analysis.scaling.plot           # scaling figure
# ... one subpackage per experiment
```

## Data

Tokenized datasets are hosted on the HuggingFace Hub:

- [`AE-data/modular-pretraining`](https://huggingface.co/datasets/AE-data/modular-pretraining) — core + auxiliary tokenized shards.
- [`AE-data/dual-use-papers`](https://huggingface.co/datasets/AE-data/dual-use-papers) — the arXiv / Europe PMC / OSTI dual-use papers corpus (see the paper appendix for collection details).

All source documents were retrieved from open, no-login-required endpoints.

## Notes on this repository

- **Checkpoints and tokenized data are not included.** `*.pth`, `*.bin`, and `*.safetensors` are git-ignored; `results/` contains only lightweight metrics (losses, configs, evaluation outputs) needed to regenerate figures.
- Paths in some scripts reflect the original cluster layout and may need adapting to your environment.

## Citation

```bibtex
@inproceedings{roland2026modular,
  title     = {Modular Pretraining Enables Access Control},
  author    = {Roland, Ethan and Cubuktepe, Murat and Martinez, Erick and
               Servaes, Stijn and Pepper, Keenan and Vaiana, Mike and
               Schwerz de Lucena, Diogo and Rosenblatt, Judd and
               Foote, Addie and Anil, Cem and Cloud, Alex},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

Work by [AE Studio](https://ae.studio), with collaborators at Anthropic. Paper, figures, and updates at [modularpretraining.com](https://modularpretraining.com).

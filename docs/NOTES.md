# Phase 0 reconnaissance: SimpleStories quantization study

Phase 0 is documentation-only: no training was run, no dataset was downloaded, and no
runtime code was changed. These notes describe the repository at the inspected commit and
define the corrected reduced study that Phase 1 should implement.

## Repository provenance

| Item | Value |
|---|---|
| Inspected commit | `d3d8dd4face93300cdab5c7cd8745c62a17ac106` (`main`) |
| Fork remote | `origin`: `https://github.com/JR-Vickers/modular-pretraining` |
| Upstream remote | `upstream`: `https://github.com/agencyenterprise/modular-pretraining.git` |
| Initial worktree state | Clean before this notes file was populated |

The provenance values above came from `git rev-parse HEAD`, `git remote -v`, and
`git status --short`. The source locations cited below are paths at that commit.

## Exact stories configuration

The production launcher is
[`src/run/experiment/stories/methods/run.py`](../src/run/experiment/stories/methods/run.py),
and its template is
[`GetStoriesConfig`](../src/run/experiment/config.py). The tokenizer and corpus totals are
recorded in [`src/data/stories/metadata.json`](../src/data/stories/metadata.json).

### Models and data

| Property | Dense baseline | GRAM |
|---|---:|---:|
| Total parameters | 26,257,920 | 32,571,904 |
| Layers | 8 | 8 |
| Embedding width | 512 | 512 |
| Core MLP width | 2,048 | 2,048 |
| Attention heads / KV heads | 8 / 2 | 8 / 2 |
| Context length | 256 | 256 |
| Vocabulary size | 4,096 | 4,096 |
| MLP experts per layer | 1 | 1 core + 4 auxiliary |

GRAM is constructed with `core_param_prc=1.0` and `aux_param_prc=0.1`. The implementation
aligns expert widths to multiples of 64, so the exact auxiliary width is 192 rather than
204.8: 9.375% of the 2,048-wide core. Each auxiliary owns 1,578,496 MLP parameters across
the eight layers (197,312 per layer, 9.397% of a core MLP's per-layer parameters). Thus
“10% auxiliary experts” is the configuration intent; 192 and 1,578,496 are the realized
shape and count. The GRAM core parameter group is exactly 26,257,920 parameters, equal to
the full dense baseline, and the four auxiliaries add 6,313,984 parameters.

`GetStoriesConfig()` sorts the 48 metadata labels and assigns the first four as auxiliary:

- `a-deadline-or-time-limit`
- `alien-encounters`
- `bygone-eras`
- `cultural-traditions`

The other 44 topics are aggregated under the training/evaluation label `core`. The
nominal train corpus contains **547,853,673 tokens**: 501,664,802 core tokens and
46,188,871 auxiliary tokens. This is the exact metadata budget, not merely the roughly
520M-token estimate in `PLAN.md`. Loaders use only complete sequences and complete
distributed micro-batches (`drop_last=True`); per-label limits and accumulation groups are
also truncated to valid batch boundaries. Actual processed tokens therefore depend on
world size, micro-batch size, label rounding, and routing's equal-compute adjustment and
can be lower than the nominal metadata sum.

### Optimization

| Setting | Value |
|---|---|
| Upstream launcher seeds | 1, 2, 3 |
| Seed for this reduced study | **1 only** |
| Learning rate | `5e-3` |
| Target effective batch size | 128 sequences |
| Optimizer | AdamW, betas `(0.9, 0.95)`, fused implementation |
| Epochs | 1 |
| LR schedule | 10% linear warmup, 80% constant, 10% linear decay |
| Compilation | `torch.compile(..., dynamic=True)` enabled |
| Routed compute normalization | `equal_compute=True` |

The setup code recalculates micro-batch size and accumulation steps from the target batch
size, parameter count, and number of processes; the literal `micro_batch_size=128` in the
template is not guaranteed to survive setup. Dense training uses one optimizer. GRAM uses
one optimizer per label/parameter group, with the common learning-rate schedule applied to
all groups.

## GRAM routing and ablation semantics

The relevant implementation is
[`do_routed_unordered`](../src/run/train/routed.py),
[`MoETransformer.get_params`](../src/model/moe.py),
[`get_exp_mask`](../src/run/util/tools.py), and
[`eval_loss`](../src/run/eval.py).

- `robust_prc=0.5` is the paper's `p_cr`: the fraction of core examples in the routed
  fine-tuning pool that are paired with an auxiliary expert and update both core and that
  expert. This trains the core to remain useful with auxiliary modules present or absent.
- `aux_route_prc=0.3` is `p_as`: the fraction of auxiliary examples whose gradients also
  update core. Every auxiliary batch runs core plus its matching auxiliary in the forward
  pass; the other 70% update only that auxiliary's optimizer.
- The `core` optimizer owns embeddings, attention, both block norms, the final norm,
  unembedding, and the core MLP in every layer. Each auxiliary optimizer owns only its
  corresponding auxiliary MLPs across layers.
- Evaluation turns the selected expert-label list into a per-expert boolean mask and uses
  the same mask for forward and backward arguments. In inference there is no backward
  pass, but the mask still selects the forward computation.
- In [`MoE.forward`](../src/model/moe.py), a zero mask entry skips that expert entirely.
  Ablating an auxiliary therefore removes its output from every Transformer layer. It
  neither edits nor zeros checkpoint weights. Consequently, ablate-then-quantize and
  quantize-then-ablate should commute for an inactive auxiliary: its weights are never read
  by the forward pass. Phase 3 should verify this once rather than treating the two orders
  as independent experiments.

## What the upstream stories launcher actually runs

For each of seeds 1, 2, and 3, the launcher builds one timestamped run beneath
`results/stories/seed_<N>/` and executes these stages in order (the dispatcher guarantees
that baseline is first):

| Directory | Stage | Behavior |
|---|---|---|
| `baseline/` | Dense baseline | Train on all labels; 200 in-training validation evaluations; no elicitation |
| `routed_01/` | GRAM | Unordered MoE routing, then five capability-profile evaluations and elicitation |
| `routed_02/` | FT-LoRA | Ordered LoRA routing/fine-tuning, evaluation, and elicitation |
| `routed_03/` | DEMix | DEMix training, single-expert evaluations, and elicitation |
| `filtering/` | Filtering | Train a fresh dense model for each default retain target, then evaluate and elicit |
| `coreftaux/` | FT-Full | Train core first, then branch into full-model auxiliary fine-tunes |

All three routed configs share the stage name `routed`; `get_stage_dirs()` deterministically
deduplicates them as `routed_01`, `routed_02`, and `routed_03`. MaxEnt is supported and
analyzed elsewhere but is not in this launcher.

### Results, checkpoints, and resumption

The timestamped run root contains:

- `config.json`: the fully resolved experiment, model, data, runtime, and stage config.
- `training.log`: rank-aware setup, parameter counts, batch composition, losses, schedule,
  restore, and evaluation messages.
- `stats.jsonl`: final evaluation and elicitation records, including stage config, data
  label, active expert labels/retained set, and loss.
- One stage directory per row above. Every stage has `stage.json`, which stores the stage
  config, a stage-level `completed` flag, and `completed_iterations` for multi-model stages.
- Training directories contain `checkpoint.pth` at the final step and `losses.pkl` with
  per-label train/validation histories. If periodic saving is enabled, or preemption
  occurs, `checkpoint_step-<N>.pth` is also written; the stories stages default to no
  periodic checkpoints, so normally only final and preemption checkpoints exist.
- Filtering and FT-Full have nested per-retain/per-phase directories with their own
  checkpoints and losses. Routed/filtering elicitation creates `elicit/` directories below
  the relevant retained-profile directory; the current `do_finetune()` logs its selected
  best loss to the run-root `stats.jsonl` and does not persist an elicited checkpoint.

Checkpoint payloads include model weights, optimizer state(s), current/total step, and loss
history. Writes use a temporary file followed by an atomic replace. On startup, rank 0
compares local and S3 candidates and restores the highest step (a final `checkpoint.pth`
always wins); other ranks receive or download the selected checkpoint. Completed stages
are skipped from `stage.json`, while filtering, FT-Full, and unlearning-style stages also
skip completed iterations.

`SIGTERM` and `SIGINT` set a preemption flag. At the next save opportunity the loop writes
a step checkpoint and exits cleanly; a grace timer forces exit if a process remains stuck.
The next invocation restores the latest checkpoint. The launcher hard-codes S3 mirroring
to bucket `ae-gradient-routing-results` with prefix `stories/seed_<N>`: it syncs down before
stage inspection, runs a background uploader, uploads checkpoints eagerly, and performs a
final full sync.

The stories compiler
[`analysis/stories/compile.py`](../analysis/stories/compile.py) reads `stats.jsonl` but also
requires each seed's `baseline/losses.pkl` to build the validation-loss reference curve used
for compute-ratio calculations. A GRAM or filtering result alone is therefore insufficient
for the existing paper analysis.

## Corrected reduced study matrix

This project uses seed 1 and only three training families: dense baseline, GRAM, and two
dense filtered controls. LoRA, DEMix, FT-Full, MaxEnt, elicitation/adversarial fine-tuning,
and seeds 2–3 are not required. The deterministic primary topic is
`a-deadline-or-time-limit`.

| Training family / run | Training data or routing | Required evaluation profiles | Role |
|---|---|---|---|
| Dense baseline | All 48 topics | Per-label losses (no expert mask) | Full-data reference |
| GRAM | All 48 topics with `p_cr=0.5`, `p_as=0.3` | Paper profiles plus quantization profiles below | Quantization subject |
| Filtered: paper control | Retain `core + a-deadline-or-time-limit`; filter the other three auxiliary topics | Per-label losses | The official Fig. 2-style comparison for the primary profile |
| Filtered: leave-one-out control | Retain `core + alien-encounters + bygone-eras + cultural-traditions`; filter only `a-deadline-or-time-limit` | Per-label losses | Gold-standard control for the quantization leave-one-out comparison |

GRAM must be evaluated under two distinct protocols:

1. **Paper protocol (five profiles):** `core` alone, then `core` plus exactly one of each
   of the four auxiliary topics. These are the profiles created by the stock
   `get_retain_targets()` path and used by the stories figure analysis.
2. **Quantization protocol (two profiles):** all five experts active, and all experts active
   except `a-deadline-or-time-limit`. This is an added leave-one-out comparison, not part of
   the paper's original five-profile protocol. The second filtered control and this GRAM
   ablation have the same retained topic set.

The stock filtering defaults only create `core` and `core + one auxiliary` retain targets.
The leave-one-out filtered control will therefore need an explicit stage-level
`retain_targets` override in Phase 1/2. Similarly, the stock GRAM stage needs either
`eval_arbsub=True` (which adds all leave-one-out masks plus all-on) or a narrower explicit
evaluation path; the reduced study only requires the two quantization profiles stated
above.

## Phase 1 portability inventory

The aim is a small device/runtime patch, not a rewrite. Items marked **blocker** prevent the
Mac MPS/CPU smoke path or select the wrong backend. Items marked **conditional/no-op** are
CUDA-specific but need changing only if they actually fail or materially harm the target
path.

| Subsystem | Assumption in the current code | Phase 1 assessment |
|---|---|---|
| Device setup | `setup()` asserts `torch.cuda.is_available()`, unconditionally selects `torch.device("cuda")`, and sets TF32/cuDNN/SDP flags. | **Blocker:** replace the assertion/device selection with CUDA/MPS/CPU selection. Backend tuning flags are harmless CUDA-only settings once guarded. |
| Model dtype | `make_model()` and `copy_model()` always move models to BF16; elicitation also moves models through CPU BF16. | **Blocker/compatibility risk:** choose a supported dtype per device; FP32 fallback is acceptable for MPS/CPU smoke runs. |
| Optimizer | Dense, routed, and other training loops construct `AdamW(..., fused=True)`. | **Blocker risk:** fused AdamW is CUDA-oriented; gate it or use the unfused implementation off CUDA. Only dense and routed loops are in the reduced scope. |
| CUDA diagnostics | Routed training calls `torch.cuda.memory_summary()` before training and during evaluation logging. | **Blocker:** guard diagnostics by device type. |
| CUDA cache/seeding | Launchers and loops call `torch.cuda.empty_cache()`, `manual_seed()`, and `manual_seed_all()`. | **Conditional/no-op:** normally harmless without an active CUDA device; guard only where the installed PyTorch build requires it. |
| Compilation | `RunConfig.compile=True`, the stories launcher reasserts it, and model construction calls dynamic `torch.compile`. | **Compatibility/performance risk:** default off for the first MPS/CPU smoke run; enable only after eager correctness is established. |
| Distributed setup | A `torchrun` launch is detected from `RANK`/`WORLD_SIZE`, then forces NCCL, calls `torch.cuda.set_device`, and initializes with a CUDA device ID. | **Blocker for torchrun on Mac:** use a plain single-process launch for Phase 1, or add a non-NCCL backend only if multi-process support is actually needed. |
| DDP collectives | DDP uses CUDA `device_ids`/`output_device`; barriers pass `torch.cuda.current_device()`. | **Blocker only in distributed mode:** the existing single-process branches avoid these calls. Do not broaden the first port to multi-device MPS. |
| NCCL cleanup | Import sets `NCCL_SHM_DISABLE=1`; setup/exit removes `/dev/shm/nccl-*`. | **Conditional/no-op on macOS single process:** Linux/NCCL-specific cleanup should be skipped off NCCL, but it is not the core device port. |
| Loader transfer | DataLoader defaults to pinned memory and transfers batches with `non_blocking=True`. | **Compatibility/performance risk:** disable pinning off CUDA if warnings/errors occur; nonblocking transfer is otherwise semantically safe. |
| Batch construction | Effective batch, micro-batch, accumulation, `DistributedSampler`, and token truncation depend on world size/GPU count. | **Reproducibility risk:** record resolved `config.json` values and realized steps/tokens for the one-device run; keep target effective batch 128. |
| Working directory | `Path("src").absolute()` in the stories launcher and experiment template assumes the process starts at repository root. | **Blocker when launched elsewhere:** either enforce/document repo-root CWD or derive paths from `__file__`; the latter is more portable and localized. |
| S3 | The launcher hard-codes bucket `ae-gradient-routing-results`; boto3/AWS credentials and `AWS_DEFAULT_REGION` (default `us-east-1`) are assumed. Sync-down, eager checkpoint upload, background mirroring, and final sync all use it. | **Environment risk, not required locally:** make S3 opt-in/disabled for the smoke run. Invalid credentials are caught and disable later S3 operations, but importing/constructing the client and contacting AE's bucket are unnecessary. |
| Hugging Face | Setup calls `AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")`; tokenized shards are external. This requires a cached tokenizer or network access and may use `HF_TOKEN`. | **Data/setup dependency:** Phase 1 must arrange the tokenizer and permitted stories shards; authentication is optional if the artifacts are public but may still be supplied through the root `.env`. Do not fetch the dual-use dataset. |
| Stories generation | `generate.py`, `generate_gram.py`, `generate_gram_seedsearch.py`, and `generate_maxent.py` each assert CUDA and select a CUDA device independently. | **Out-of-scope blocker:** generation is not needed for the training/evaluation smoke gate. Port these only if later quantization analysis uses them. |

The minimum Phase 1 execution path is therefore: select MPS/CPU, select a supported dtype,
disable/gate fused AdamW and CUDA memory diagnostics, run eager single-process from the
repository root, disable S3, and preserve the resolved batch configuration. CUDA cache
calls, TF32/cuDNN settings, NCCL cleanup, generation scripts, and broader DDP portability
should not expand the patch unless the chosen path reaches them.

## Phase 0 gate

The exact configuration, routing semantics, launcher stages, result/resume behavior,
corrected two-protocol matrix, analysis dependency, and stories-path portability risks are
now recorded. Phase 0 is complete; the next allowed work is the small Phase 1 port and smoke
setup, still without the dual-use dataset.

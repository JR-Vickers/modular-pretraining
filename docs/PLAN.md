# PLAN.md — GRAM Quantization Robustness Study

## Objective

Test whether GRAM's structural knowledge isolation survives post-training quantization.

GRAM (Gradient-Routed Auxiliary Modules, Roland et al., ICML 2026, arXiv:2607.08077) is a
pretraining method from AE Studio / Anthropic that isolates domain knowledge into small
auxiliary MLP modules which can be ablated at inference to remove a capability. The paper
shows ablation resists recovery under adversarial finetuning far better than post-hoc
unlearning. **Nobody has tested the quantization attack surface.** Prior work (Zhang et al.,
"catastrophic failure of quantized LLM unlearning," ICLR 2025 era) showed 4-bit quantization
largely *restores* knowledge removed by behavioral unlearning methods.

**Hypothesis:** Structural removal (parameters physically absent from the forward pass)
should be robust to precision loss in a way weight-perturbation unlearning is not.
Quantizing a GRAM model with a module ablated should NOT restore the forgotten capability.

Either outcome is publishable as a short LessWrong post:
- Robust → new empirical selling point for GRAM vs. unlearning.
- Not robust → real deployment caveat for the method.

**Deadline context:** Final writeup must be publishable by ~Aug 10, 2026 (feeds an
application due Aug 17). Bias every decision toward shipping.

## Source materials

- Official code: `https://github.com/agencyenterprise/modular-pretraining` (paper repo;
  training + eval + elicitation code included; **checkpoints NOT included**)
- Tokenized data on HuggingFace Hub: `AE-data/modular-pretraining` (core + aux shards);
  `AE-data/dual-use-papers` (NOT needed — realistic experiment out of scope)
- Paper: arXiv 2607.08077; project page modularpretraining.com
- Requires `HF_TOKEN` in `.env` at repo root (git-ignored) for data access

### GRAM mechanics the agent must understand before touching code

- Each GRAM layer = one always-active **core MLP** (baseline-sized) + N−1 small
  **auxiliary modules**, one per domain. No learned router, no per-token routing.
- Training: gradient routing by batch data label. Aux batches always update their module,
  update core with prob `p_as`. Core batches always update core; with prob `p_cr` also
  activate a random aux module (keeps core robust to ablation).
- Inference: capability profile = core + chosen subset of aux modules; rest ablated
  (binary mask on module outputs).
- Relevant source files: `src/model/moe.py` (GRAM), `src/model/base.py` (dense baseline),
  `src/run/experiment/stories/` (our target experiment), `src/run/train/finetune.py`
  (adversarial elicitation — stretch goal only).

## Scope

**IN:** The `stories` experiment only — 26M-param model on Simple Stories synthetic data,
where GRAM approximates 5 data-filtered models (paper Fig. 2 / Table 1). Then quantization
experiments on the resulting checkpoint.

**OUT (do not attempt):**
- The 800M "realistic" dual-use experiment (virology/cyber/nuclear). Too much compute.
- Training all 5 filtered comparison models. Train GRAM + baseline + at most 2 filtered
  models.
- Multi-seed replication (paper uses 3 seeds). One seed; state this honestly in writeup.
- Any MLX port. Run THEIR code, minimally modified. Replication credibility depends on it.

## Hardware strategy

- **Local (MacBook Pro M4, 64GB):** environment port, path fixes, shape checks, smoke runs
  (~5–20M tokens) to validate loss curves and that module ablation measurably changes
  forget-topic loss. Memory is a non-issue at 26M params. Full local training is feasible
  (~overnight per run on MPS) but is the fallback, not the plan.
- **Rented GPU (Prime Intellect, 1× 4090 or A100):** full training runs. A 26M
  Chinchilla-optimal run (~520M tokens, ~8e16 FLOPs) is roughly an afternoon; the full
  run matrix costs single-digit dollars at spot prices.

## Phases

### Phase 0 — Recon (no training)

1. Clone repo; read `README.md`, `src/run/experiment/stories/`, `src/model/moe.py`,
   `src/run/main.py`, checkpointing/dataloader utils.
2. Produce a short `NOTES.md`: exact stories-experiment stages, config values (model dims,
   N modules, `p_as`, `p_cr`, token budget, batch/LR schedule), what metrics get written to
   `results/`, and how ablation masks are set at eval time.
3. Identify every CUDA/DDP/cluster-path assumption. Repo defaults to
   `torchrun --nproc_per_node=8` DDP; README warns paths reflect their cluster layout.

**Gate:** NOTES.md complete and the run matrix below confirmed/corrected against actual
configs before any code changes.

### Phase 1 — Port & smoke test (local)

1. Minimal single-device patch: run without DDP (or `--nproc_per_node=1`), device
   selection `cuda` → `cuda|mps|cpu`, fix `pin_memory`/bf16 assumptions
   (fp32 fallback on MPS is acceptable at this scale), fix hardcoded paths.
   Keep the diff SMALL and isolated — record it; the writeup must state exactly what
   was changed from upstream (cite commit hash).
2. Download stories shards from `AE-data/modular-pretraining`.
3. Smoke run: short GRAM training (~5–20M tokens). Acceptance: loss decreases; ablating an
   aux module increases loss on its topic more than on other topics (direction, not
   magnitude).
4. If a genuine upstream bug is found, note it (candidate GitHub issue/PR — flag for human).

**Gate:** smoke run passes acceptance; diff vs upstream is documented.

### Phase 2 — Full training runs (rented GPU preferred)

Run matrix, priority order:

| # | Run | Purpose | Required? |
|---|-----|---------|-----------|
| 1 | GRAM, full stories config | The subject of all quantization work | YES |
| 2 | Dense baseline | Reference for retain performance | YES |
| 3 | Filtered model (1 topic held out) | Gold-standard comparison for removal | Strongly preferred |
| 4 | Second filtered model | Extra comparison point | Only if time allows |

Evaluate per paper protocol: per-topic losses with each aux module ablated vs. active.
Confirm qualitative reproduction: ablation ≈ filtering for forget topic; core/retain
performance holds. Match direction and rough magnitude, not their CIs (1 seed vs their 3).

**Gate:** GRAM checkpoint reproduces the qualitative Fig. 2 effect. If it does not, STOP
and debug/report — the quantization study is meaningless on a broken replication.

### Phase 3 — Quantization experiments (the actual contribution)

Implement **hand-rolled fake-quant in pure PyTorch** (works on MPS and CUDA; no
bitsandbytes — CUDA-only; no GPTQ/AWQ — won't ingest a custom arch). Per-tensor or
per-channel symmetric quantization of Linear weights to a k-bit grid, then dequantize;
inference proceeds in full precision. Parameterize by (a) bit width, (b) which parameter
groups are quantized: {core MLPs, aux modules, attention, embeddings} independently
selectable. Unit-test roundtrip error against known tensors first.

Precisions: fp32 reference, int8, int6, int4. (int3/int2 optional if curves are flat.)

Measurements, priority order:

1. **Leakage under ablation (headline).** For each precision: quantize all weights,
   ablate module *i*, measure forget-topic loss/eval vs. the fp32-ablated model and vs.
   the filtered model. Question: does forget performance stay at "removed" levels, or
   drift back toward the unablated model as precision drops?
2. **Differential degradation.** Quantize core-only vs. aux-only vs. all, at each
   precision. Do small aux modules degrade faster than the core? Report retain-topic and
   forget-topic deltas separately.
3. **Order effects.** Ablate-then-quantize vs. quantize-then-ablate. (With ablation as a
   binary output mask these may be trivially identical — verify from the code in Phase 0;
   if identical, say so in one line and move on.)
4. **STRETCH ONLY — elicited forget.** Adversarial finetuning recovery
   (`src/run/train/finetune.py`) on int4 vs fp32 GRAM. Skip unless Phases 1–3 are done
   with ≥5 days to deadline.

Also record for the writeup: unablated-model behavior under quantization (sanity that
general degradation is graceful), and quantization error stats per parameter group.

### Phase 4 — Analysis & writeup

1. Plots: forget-topic and retain-topic performance vs. bit width, one line per condition
   (GRAM-ablated, GRAM-unablated, filtered, baseline). One clean headline figure.
2. Draft LessWrong post, structured as: hypothesis → decision rules (stated BEFORE
   results) → setup (their code, commit hash, exact diff, 1 seed, toy scale) → results →
   limitations (synthetic data; 26M; may not transfer to 800M realistic setting; single
   seed) → open question: does the effect interact with scale?
3. Publish repo (fork with diffs + quant code + analysis notebooks), reproducible with a
   single documented command sequence.

**Decision rules (commit to these up front, report against them):**
- "Leakage" = quantized-ablated forget performance recovers ≥X% of the gap between
  fp32-ablated and fp32-unablated (set X after seeing fp32 gap size in Phase 2; candidate
  X = 20%).
- "Robust" = recovery < X% at int4.
- Anything between: report the curve, no binary claim.

## Constraints & style

- Their code, minimally patched. Every deviation logged. Cite the upstream commit hash.
- Honest scope statements beat inflated claims: 1 seed, subset of comparison models,
  synthetic data — all stated plainly.
- Timebox: Phase 0–1 ≤ 2 days, Phase 2 ≤ 2 days, Phase 3 ≤ 3 days, Phase 4 ≤ 2 days.
  If any phase blows its box, cut from the bottom of the priority lists, never from
  honesty or from the headline measurement.
- Do not train on or download `AE-data/dual-use-papers`.
- All experiment configs, seeds, and outputs versioned; runs resumable (their pipeline is
  preemption-safe — keep that property).

## Deliverables

1. Forked repo: minimal port diff, quantization module + tests, run configs, analysis
   notebooks, headline figure.
2. `RESULTS.md` with the decision-rule verdicts.
3. LessWrong post draft: "Does GRAM's knowledge isolation survive quantization?"
4. (If found) upstream bug report/PR against `agencyenterprise/modular-pretraining`.

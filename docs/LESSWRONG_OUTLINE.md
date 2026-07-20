# Does GRAM's Knowledge Isolation Survive Quantization?

> Outcome-neutral working outline. Select exactly one interpretation branch only after the
> complete 290-record full-test report passes validation. CPU smoke results are pipeline
> checks and must never supply claims, numbers, or a verdict.

## One-paragraph summary

- State the practical question: can ordinary post-training quantization undo a capability
  removal that is implemented by disabling GRAM auxiliary modules?
- Give the complete-full-test verdict in one sentence, followed immediately by scope:
  one seed, SimpleStories, a 26M dense core / 32.57M GRAM model, and fake weight-only
  quantization.
- Distinguish evidence about robustness to a weight-grid perturbation from claims about
  safety, semantic knowledge deletion, or deployed integer inference.

## Motivation and competing hypotheses

- Behavioral unlearning can be fragile under quantization because the intervention is
  encoded as a delicate change to weights; rounding may partially undo that change.
- GRAM's intervention is structural: a topic-specific auxiliary module is removed from the
  forward pass by a binary mask. This suggests robustness because quantization cannot turn
  a masked output back on.
- Competing mechanism: quantizing the shared core and remaining modules could change their
  behavior enough to recover the held-out topic or collapse the separation between all-on
  and ablated profiles, even though the removed module stays off.
- Competing null: low precision may simply worsen every topic. A lower isolation gap is not
  recovery if the whole model is degrading, hence the separate recovery, erosion, and
  retained-utility measurements.

## Decision rules fixed before the result

- Define deadline-topic all-on and deadline-off losses at FP32 as
  \(L_{on,32}\) and \(L_{off,32}\), with isolation gap
  \(G_{32}=L_{off,32}-L_{on,32}\).
- At precision \(k\), define \(G_k=L_{off,k}-L_{on,k}\), signed absolute
  recovery \(A_k=(L_{off,32}-L_{off,k})/G_{32}\), and signed isolation erosion
  \(E_k=1-G_k/G_{32}\). Do not clip negative values.
- The authoritative condition is all-weight, per-output-channel int4:
  - capability recovery if \(A_4\ge20\%\);
  - isolation erosion without recovery if \(E_4\ge20\%\) and \(A_4<20\%\);
  - robust if both are below 20%;
  - inconclusive due to general degradation if a required loss is non-finite or GRAM
    all-on int4 mean retained-topic loss rises by at least 10% from FP32.
- Explain why recovery and erosion are separate. Quantization can move deadline-off loss
  toward all-on, shrink the gap because all-on worsens, do both, or do neither.
- Say explicitly that these rules and the 20% / 10% thresholds were committed before the
  full checkpoint matrix was evaluated.

## What GRAM changes mechanically

- Each Transformer layer has one always-active core MLP and one small auxiliary module per
  story domain; there is no learned per-token router.
- Gradient routing determines which core and auxiliary parameters update for each labeled
  training batch. At inference, a capability profile is the core plus a chosen subset of
  auxiliary outputs.
- Ablation multiplies the selected auxiliary output by a binary mask. Quantize-then-ablate
  and ablate-then-quantize are therefore identical for this intervention: quantization
  changes weights, while masking removes the same module output from the computation.
- Contrast carefully with behavioral unlearning. GRAM does not prove that all information
  about a topic resides only in one module; it tests whether disabling the specialized
  path preserves the measured loss separation after other active weights are rounded.

## Checkpoints and qualitative replication

- Identify the upstream repository and commit, this fork's relevant commit, and every MPS
  portability change. Link `docs/NOTES.md` and the exact diff.
- Describe the three seed-1 checkpoints: GRAM, dense baseline, and deadline-filtered.
  Report optimizer steps, parameter counts, corpus budgets, and SHA-256 checkpoint hashes.
- Briefly establish the Phase 2 gate before discussing quantization: deadline ablation
  worsened its own topic substantially more than retained topics and moved toward the
  filtered control. Link the canonical Phase 2 table rather than repeating every value.

## Quantization methodology

- Weight-only symmetric fake quantization in pure PyTorch; inference still executes in
  floating point. Biases and RMSNorm parameters remain FP32.
- For \(k\) bits use the narrow signed range
  \([-q_{max},q_{max}]\), \(q_{max}=2^{k-1}-1\), scale by maximum absolute
  weight, round and clamp, then dequantize. All-zero rows remain exactly zero.
- Primary granularity: one scale per output row, including embedding vocabulary rows.
  Per-tensor scaling is sensitivity evidence only.
- Parameter groups: core MLPs, auxiliary modules, attention, and input/output embeddings.
  The headline condition quantizes every applicable matrix; singleton groups diagnose
  where error enters.
- Precisions: FP32, int8, int6, and int4. No activation quantization, packed kernels,
  GPTQ/AWQ calibration, int3/int2, or adversarial finetuning.

## Matrix and evaluation

- Report 290 unique records across all 27 canonical condition families.
- Explain the matrix: all five GRAM profiles and five topics for primary all-weight
  per-channel conditions; all-on/deadline-off singleton-group diagnostics; dense and
  filtered controls; and per-tensor GRAM sensitivity runs.
- State evaluation sample count, tokenizer/data provenance, device/dtype, deterministic
  checkpoint-only loading, exact expert masks, and finite-loss validation.
- Retained topics are core, alien encounters, bygone eras, and cultural traditions.
- One seed means there are no confidence intervals. Lines connect evaluated categorical
  precisions and should not imply a continuous fitted relationship.

## Headline result

Place `phase3_headline.png` here, followed by a caption that names all three panels:

1. Raw deadline-topic cross-entropy for GRAM all-on, GRAM deadline-off, dense baseline,
   and deadline-filtered; lower is better.
2. Signed, unclipped capability recovery and isolation erosion, with zero and the 20%
   verdict threshold.
3. Mean retained-topic loss change from each model's FP32 mean, with zero and the 10%
   utility guard; lower is better.

Call out a compact table containing exact raw losses, signed gaps, recovery, erosion, and
retained changes at every precision. A second table should give same-bit deadline-off
distance from the filtered control. Avoid putting secondary diagnostics in the figure.

## Interpretation branch — choose one

### Branch A: robust

- Use only if both int4 recovery and erosion are below 20% and the utility guard passes.
- Say that this experiment found no pre-registered evidence that per-channel int4 fake
  quantization restored the ablated deadline capability or materially eroded its measured
  isolation gap.
- Discuss whether negative recovery reflects ordinary degradation rather than stronger
  removal. Do not relabel negative recovery as an improvement in safety.
- Bound the claim to the tested checkpoint, quantizer, topic, and loss metric.

### Branch B: capability recovery

- Use only if int4 recovery is at least 20% and the utility guard passes.
- State the fraction of the FP32 isolation gap recovered and show the raw deadline-off
  movement so the normalized ratio cannot hide scale.
- Compare the same-bit filtered model and retained-topic controls. Discuss whether the
  active shared network plausibly supplied the recovered behavior despite the mask.
- Frame this as a deployment caveat for structural isolation, not proof that GRAM and all
  behavioral unlearning fail for the same reason.

### Branch C: isolation erosion without recovery

- Use only if erosion is at least 20%, recovery is below 20%, and utility passes.
- Explain which endpoint moved: an all-on degradation can shrink the gap without improving
  deadline-off behavior. Lead with raw losses before the normalized erosion number.
- Describe the result as reduced distinguishability of the two profiles, not capability
  restoration.

### Branch D: inconclusive due to general degradation

- Use if any required loss is non-finite or GRAM int4 retained degradation reaches 10%.
- State which guard triggered and report the headline curves descriptively without a
  recovery/robustness verdict.
- Discuss whether int6 remains informative only as exploratory evidence. Do not move the
  pre-registered endpoint after seeing int4.

## Secondary diagnostics

- Deadline and alien: normalized signed recovery and erosion curves.
- Bygone and cultural: raw all-on/off curves only, because their FP32 isolation gaps are
  too small for stable normalized ratios.
- Singleton groups: core MLP, auxiliaries, attention, and embeddings, separated by topic
  and all-on/deadline-off profile. Treat these as localization clues, not independent
  confirmatory tests.
- Quantization error: overall and per-group error by precision and granularity. Do not
  equate parameter error magnitude directly with behavioral importance.
- Per-tensor sensitivity: compare with per-channel direction and magnitude, explicitly
  labeling it non-primary.

## Limitations

- One seed: no uncertainty interval or estimate of checkpoint-to-checkpoint variance.
- Synthetic SimpleStories domains may be cleaner and more modular than real capabilities.
- Small scale: 26M dense core / 32.57M GRAM; routing and redundancy may change with scale.
- One strongly isolated primary topic; weaker bygone/cultural FP32 gaps cannot support the
  same normalized claims.
- Fake weight quantization omits packed-kernel numerical behavior, activation and KV-cache
  quantization, calibration, and hardware-specific kernels.
- Cross-entropy is a behavioral proxy, not a demonstration that latent knowledge is absent.
- No adversarial elicitation or finetuning after quantization.

## Reproducibility

```bash
source .venv/bin/activate

python -m src.run.experiment.stories.quantization.run --device mps
python -m analysis.stories_phase3.compile \
  --result-dir results/stories_phase3/20260718174851965051/full
python -m analysis.stories_phase3.plot \
  --result-dir results/stories_phase3/20260718174851965051/full \
  --output results/stories_phase3/20260718174851965051/full/phase3_headline.png
```

- Link the validated manifest, compiled JSON, flat CSV, generated summary, figure, source
  checkpoint hashes, resolved config, and quantization implementation.
- Record the final git commit and environment versions. Note that rerunning the matrix is
  resumable and completed condition files are immutable.

## What would change the conclusion?

- Does the result replicate across seeds and topics with large FP32 isolation gaps?
- Does it survive at 800M scale on realistic domains, where shared-core knowledge and
  auxiliary specialization may differ?
- How do activation quantization and real packed int4 kernels compare with this controlled
  weight perturbation?
- Does adversarial finetuning recover the topic differently before and after quantization?
- Are other structural interventions similarly robust, and can training make robustness
  to a specified deployment quantizer explicit?

## Closing

- Return to the narrow deployment question and the selected pre-registered verdict.
- Separate the observed result from the larger open question: whether structural knowledge
  isolation remains reliable as models, domains, and compression stacks become realistic.

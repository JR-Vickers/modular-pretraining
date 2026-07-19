# Phase 2 Summary

**Gate: PASS.** Proceed to quantization.

## Inputs

- GRAM: `/Users/jarrett/dev/gram-quant/modular-pretraining/results/stories_phase2/seed_1/20260718174851965051`
- Baseline: `/Users/jarrett/dev/gram-quant/modular-pretraining/results/stories_phase2/seed_1/20260719012010132266`
- Deadline-filtered: `/Users/jarrett/dev/gram-quant/modular-pretraining/results/stories_phase2/seed_1/20260719070035561998`
- Checkpoint-only evaluation: `/Users/jarrett/dev/gram-quant/modular-pretraining/results/stories_phase2/evaluations/20260718174851965051`

All runs use seed 1, eager FP32 on MPS, micro-batch 16, accumulation 8, effective batch 128, and the paper model shape.

## Gate metrics

| Condition | Value | Result |
|---|---:|:---:|
| Deadline forget effect | +0.11411285 | Pass |
| Median absolute off-topic change | 0.00218302 | Pass |
| Filter distance: all-on / ablated | 0.08930361 / 0.02480924 | Pass |
| Mean absolute retained change | 0.00221005 | Pass |

## Primary losses

| Label | Baseline | GRAM all-on | Deadline off | Filtered |
|---|---:|---:|---:|---:|
| core | 1.63167393 | 1.65065944 | 1.64733434 | 1.63246596 |
| a-deadline-or-time-limit | 1.71857691 | 1.72407925 | 1.83819211 | 1.81338286 |
| alien-encounters | 1.60794282 | 1.61175525 | 1.60993826 | 1.60726237 |
| bygone-eras | 1.65203333 | 1.67120099 | 1.66865194 | 1.65447843 |
| cultural-traditions | 1.68839550 | 1.70479941 | 1.70365036 | 1.69228470 |

## All leave-one-out effects

| Module | Own-topic delta | Mean absolute retained delta |
|---|---:|---:|
| a-deadline-or-time-limit | +0.11411285 | 0.00221005 |
| alien-encounters | +0.09043288 | 0.00180209 |
| bygone-eras | +0.00427902 | 0.00120521 |
| cultural-traditions | +0.01942885 | 0.00111687 |

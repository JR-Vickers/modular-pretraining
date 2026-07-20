# Phase 3 Quantization Summary

## Pre-registered decision rules

The authoritative int4 verdict is *capability recovery* when signed recovery is at least 20%; *isolation erosion without recovery* when signed erosion is at least 20% but recovery is below 20%; and *robust* when both are below 20%. A non-finite required loss or at least 10% degradation in GRAM's int4 mean retained-topic loss makes the result inconclusive due to general degradation. Ratios are signed and never clipped.

## Result: robust

| Precision | GRAM all-on loss | GRAM deadline-off loss | Signed isolation gap | Recovery | Erosion |
|---|---:|---:|---:|---:|---:|
| FP32 | +1.72407925 | +1.83819211 | +0.11411285 | +0.00% | +0.00% |
| int8 | +1.72438669 | +1.83861852 | +0.11423182 | -0.37% | -0.10% |
| int6 | +1.72709167 | +1.84115911 | +0.11406744 | -2.60% | +0.04% |
| int4 | +1.81474698 | +1.92660880 | +0.11186182 | -77.48% | +1.97% |

Recovery measures movement of deadline-off loss toward the FP32 all-on model; erosion measures shrinkage of the on/off isolation gap. Negative values therefore mean movement away from recovery or a widening gap, not zero effect.

## Retained-topic utility guard

Retained topics are core, alien encounters, bygone eras, and cultural traditions. Values are the signed relative change in their mean loss from each model's own FP32 value; lower is better.

| Precision | GRAM all-on | Dense baseline | Deadline-filtered |
|---|---:|---:|---:|
| FP32 | +0.00% | +0.00% | +0.00% |
| int8 | +0.01% | +0.01% | +0.01% |
| int6 | +0.17% | +0.17% | +0.18% |
| int4 | +5.46% | +5.39% | +5.69% |

GRAM's int4 retained-topic change was +5.46%; the pre-registered 10% utility guard passed.

## Deadline-off distance from the filtered control

| Precision | GRAM deadline-off | Deadline-filtered | Signed distance | Absolute distance |
|---|---:|---:|---:|---:|
| FP32 | +1.83819211 | +1.81338286 | +0.02480924 | 0.02480924 |
| int8 | +1.83861852 | +1.81372130 | +0.02489722 | 0.02489722 |
| int6 | +1.84115911 | +1.81685925 | +0.02429986 | 0.02429986 |
| int4 | +1.92660880 | +1.90152407 | +0.02508473 | 0.02508473 |

## Secondary evidence and limitations

Deadline and alien normalized isolation, bygone/cultural raw curves, singleton parameter-group diagnostics, quantization error, and per-tensor sensitivity remain in `phase3_report.json` and `phase3_records.csv`. They are secondary evidence and do not alter the pre-registered verdict; per-tensor quantization is a sensitivity analysis only.

This is one seed on synthetic SimpleStories data, using a 26M-parameter dense-core model and a 32.57M-parameter GRAM model. Fake weight quantization measures weight-grid perturbation while inference remains floating point; it is not a benchmark of packed integer kernels. The study does not establish transfer to larger models, realistic domains, other quantizers, activation quantization, or adversarial finetuning.

# Results

## Phase 1 — MPS port and GRAM smoke test

**Decision: PASS (2026-07-18).** A paper-shaped GRAM with 32,571,904 parameters was
trained for one seed in eager FP32 on Apple MPS using a nominal 10,000,000-token
SimpleStories budget. The benchmark projected 0.101 hours. Median training loss decreased
from 5.849 to 3.834, all recorded losses were finite, and all four auxiliary ablations
passed the predefined selectivity criterion. Three ablation effects were small, so this
establishes pipeline correctness rather than strong paper-replication evidence.

### Run identity

| Item | Value |
|---|---|
| Training-code commit | `526ef78cc7f0af66c40096aa0c9769f33f77d260` |
| Benchmark run | `20260718164905895611` |
| Training run | `20260718165018541363` |
| Model | Paper-shaped GRAM, 32,571,904 parameters |
| Training budget | 10,000,000 nominal tokens; 9,535,488 processed token positions after batch alignment |
| Seed / epochs | 1 / 1 |
| Runtime | Eager FP32 MPS, single process |
| Batch | Micro-batch 16, accumulation 8, effective batch 128, context 256 |
| Routing | `p_cr=0.5`, `p_as=0.3`, 91.6/8.4 nominal core/aux token split |
| Host | MacBook Pro (`Mac16,5`), Apple M4 Max, 16-core CPU, 64 GB memory |
| Software | macOS 15.7.7 (`24G720`), Python 3.12.10, PyTorch 2.13.0 |

The parameter count and token budget describe different quantities: this was a
**32.57M-parameter model** trained with a **nominal 10M-token corpus budget**. It was not a
10M-parameter model or a 30M-token run.

### Commands

```bash
source .venv/bin/activate

python -m src.run.experiment.stories.smoke.run \
  --benchmark-only --model-shape paper --device mps --dtype float32

python -m src.run.experiment.stories.smoke.run \
  --model-shape paper --device mps --dtype float32

python -m analysis.stories.smoke_check \
  results/stories_smoke/seed_1/20260718165018541363
```

### Timing and acceptance results

The benchmark timed ten effective synthetic routed batches in 11.950 seconds and projected
364.696 seconds (0.101 hours) for the nominal token budget, comfortably below the six-hour
paper-shape cutoff. From the training log timestamps, routed training took approximately
6m44s (`16:50:28`–`16:57:12`), evaluation took approximately 13s, and the complete command
took approximately 7m07s including tokenizer/data setup (`16:50:18`–`16:57:25`).

| Gate | Result |
|---|---:|
| Finite recorded losses | Pass |
| First-10%-median training loss | 5.8492947 |
| Final-10%-median training loss | 3.8335514 |
| Training-loss decrease | 34.46% |
| Auxiliary selectivity | 4/4 pass |

| Ablated auxiliary | Own-topic signed loss increase | Median signed delta on core + other auxiliaries | Result |
|---|---:|---:|---|
| `a-deadline-or-time-limit` | +0.002961 | -0.000979 | Pass |
| `alien-encounters` | +0.027961 | -0.002799 | Pass |
| `bygone-eras` | +0.001534 | +0.000212 | Pass |
| `cultural-traditions` | +0.002130 | -0.000662 | Pass |

The alien auxiliary has the clearest selective effect. The other three pass the
pre-registered directional gate but have small absolute effects that could be sensitive to
the 128-sequence evaluation sample. Phase 2 must establish a stronger qualitative
replication before quantization conclusions are drawn.

### Versioned evidence

- [Benchmark projection](../results/stories_smoke/seed_1/20260718164905895611/benchmark_projection.json)
- [Resolved training config](../results/stories_smoke/seed_1/20260718165018541363/config.json)
- [Raw evaluation records](../results/stories_smoke/seed_1/20260718165018541363/stats.jsonl)
- [Training losses](../results/stories_smoke/seed_1/20260718165018541363/routed/losses.pkl)
- [Smoke-gate summary](../results/stories_smoke/seed_1/20260718165018541363/smoke_summary.json)

Checkpoints, tokenized bins, stage bookkeeping, and the verbose training log are retained
locally but excluded from version control.

### Limitations

This was one seed, a shortened smoke run on synthetic data, and only 128 evaluation
sequences per evaluated label. It did not train the dense or filtered controls and did not
test quantization. Passing Phase 1 validates the MPS port, training stability, routing,
masking, and directional ablation behavior; it is not a full replication of the paper.

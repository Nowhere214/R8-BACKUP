# Group Meeting Report: Stage 2 Attribution Design

Date: 2026-09-16

## Goal

The goal for this checkpoint is to turn the August Stage 2 fallback result into
an attribution question.

The August result showed a large decode improvement:

| Model | Decode speedup | Wall-time ratio |
|---|---:|---:|
| Qwen2 0.5B | 9.054× | 0.113× |
| Qwen2 1.5B smoke | 13.673× | 0.077× |

This is strong evidence that full-context decode recomputation was the right
bottleneck to target. The remaining question is which implementation change
explains how much of the gain.

## Attribution Problem

The current comparison has two arms:

1. Stage 1 full-context decode
2. Stage 2 KV-cache fallback

This is enough to measure total progress, but it mixes several effects:

- removing full-context recomputation during decode,
- changing from the Stage 1 reference stack to the self implementation stack,
- using MLX built-in quantized matmul fallback,
- keeping Python-level token-loop overhead.

For a stronger research story, the next benchmark should separate these effects.

## Benchmark Design

Use four benchmark arms on the same workload:

| Arm | Role |
|---|---|
| Stage 1 full-context reference | Baseline |
| Stage 2 full-context self path | Implementation-stack control |
| Stage 2 KV-cache fallback | Current optimized fallback path |
| mlx-lm | Production reference |

The benchmark reports prefill throughput, decode throughput, wall time, and
peak memory using the same summary schema as the existing research table.

## Interpretation Plan

| Comparison | Meaning |
|---|---|
| Stage 2 full-context vs Stage 1 full-context | Effect of self implementation and MLX fallback stack |
| Stage 2 KV-cache vs Stage 2 full-context | Direct KV-cache contribution |
| mlx-lm vs Stage 2 KV-cache | Remaining production gap |

## Implementation Status

The benchmark script has been added:

- `benches/stage2_attribution_benchmark.py`

The planned output files are:

- `benches/stage2_attribution_benchmark_results.csv`
- `benches/stage2_attribution_benchmark_summary.csv`
- `benches/stage2_attribution_benchmark_ratios.csv`
- `benches/stage2_attribution_benchmark.png`

## Recommended Next Step

Run the attribution benchmark first on Qwen2 0.5B. Only after that decide
whether the next implementation work should be C++/Metal kernels,
batching/chunked prefill, Stage 2 profiling, or benchmark CLI consolidation.

## Evidence References

- `docs/AUGUST_PROGRESS_LEDGER_2026.md`
- `docs/AUGUST_MONTH_END_SUMMARY_2026-08-31_EN.md`
- `docs/PHASE2_KV_FALLBACK_BENCHMARK_2026-05-29.md`
- `benches/research_progress_unified_summary.csv`

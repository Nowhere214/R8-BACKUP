# SiliconLLM August Progress Summary

Date: 2026-08-31

## Summary

August turned Phase 2 from a plan into a measured fallback implementation.

The main progress was:

1. Stage 2 KV-cache correctness path started.
2. Stage 2 fallback benchmark was run on Qwen2 0.5B.
3. A narrower Qwen2 1.5B smoke benchmark was added.
4. Phase 2 results were added to the research-level long table.

## Main Result

The strongest August result is the decode improvement from Stage 1
full-context decode to Stage 2 KV-cache fallback.

| Model | Benchmark type | Decode speedup | Wall-time ratio |
|---|---|---:|---:|
| Qwen2 0.5B | N=3 benchmark | 9.054× | 0.113× |
| Qwen2 1.5B | N=2 smoke benchmark | 13.673× | 0.077× |

This is the first implementation-side result that directly supports the Phase
0 diagnosis: full-context decode recomputation was the main problem.

## Evidence Package

Main records:

- `docs/PHASE2_KV_CACHE_CORRECTNESS_2026-05-29.md`
- `docs/PHASE2_KV_FALLBACK_BENCHMARK_2026-05-29.md`
- `docs/GROUP_MEETING_REPORT_2026-08-12_EN.md`
- `docs/GROUP_MEETING_REPORT_2026-08-26_EN.md`
- `docs/RESEARCH_PROGRESS_SUMMARY_2026-05-29.md`

The research-level table now has 199 data rows:

- Phase 0: 151 rows
- Phase 1: 24 rows
- Phase 2: 24 rows

## Caveat

The August benchmark should be labeled as:

```text
Stage 2 KV-cache fallback
```

It is not a custom C++/Metal kernel benchmark. The Stage 2 path uses MLX
built-in quantized matmul.

## Next Decision

The next major decision is whether to:

1. continue with fallback-based attribution benchmarks, or
2. install full Xcode and begin custom C++/Metal kernel implementation.

My recommendation is to do one more attribution benchmark before installing
Xcode. The current fallback result is already strong, so the next step should
separate which part of the gain comes from KV cache and which part comes from
quantized execution.

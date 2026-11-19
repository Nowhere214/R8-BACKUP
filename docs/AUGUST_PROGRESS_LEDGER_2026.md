# August 2026 Progress Ledger

Prepared on: 2026-05-29

Purpose: pre-organize the August reporting trail without overstating results.
For August I recommend two group meetings plus an end-of-month summary:

- 2026-08-12: Stage 2 KV-cache fallback benchmark on Qwen2 0.5B
- 2026-08-26: 1.5B smoke extension and attribution caveats
- 2026-08-31: month-end summary and next decision point

## 1. Ready for 2026-08-12

Status: completed and documented.

Main story:

The first Phase 2 fallback benchmark connects the Phase 0 diagnosis to an
implementation result. Removing full-context decode through the Stage 2
KV-cache fallback path improves decode throughput substantially on Qwen2 0.5B.

Evidence:

| Item | Artifact |
|---|---|
| Benchmark script | `benches/stage2_kv_fallback_benchmark.py` |
| 0.5B summary CSV | `benches/stage2_kv_fallback_benchmark_summary.csv` |
| 0.5B figure | `benches/stage2_kv_fallback_benchmark.png` |
| Benchmark record | `docs/PHASE2_KV_FALLBACK_BENCHMARK_2026-05-29.md` |
| Group meeting draft | `docs/GROUP_MEETING_REPORT_2026-08-12_EN.md` |

Key result:

| Metric | Stage 1 full-context | Stage 2 KV fallback | Stage 2 / Stage 1 |
|---|---:|---:|---:|
| Decode throughput | 7.27 ± 0.20 tok/s | 65.78 ± 0.41 tok/s | 9.054× |
| Wall time | 55.97 ± 1.46 s | 6.32 ± 0.03 s | 0.113× |
| Peak memory | 1.89 GB | 0.93 GB | 0.493× |

Important label:

```text
Stage 2 KV-cache fallback
```

Do not call it custom quantized matmul or custom Flash Attention.

## 2. Ready for 2026-08-26

Status: completed as a smoke extension.

Main story:

The 0.5B result also appears at the next model size in a narrower 1.5B smoke
benchmark. The result strengthens the direction of the finding, but it is not
a full N=3 sweep.

Evidence:

| Item | Artifact |
|---|---|
| 1.5B smoke summary CSV | `benches/stage2_kv_fallback_benchmark_1p5b_smoke_summary.csv` |
| 1.5B smoke figure | `benches/stage2_kv_fallback_benchmark_1p5b_smoke.png` |
| Research-level table | `benches/research_progress_unified_summary.csv` |
| Group meeting draft | `docs/GROUP_MEETING_REPORT_2026-08-26_EN.md` |

Key result:

| Metric | Stage 1 full-context | Stage 2 KV fallback | Stage 2 / Stage 1 |
|---|---:|---:|---:|
| Decode throughput | 3.41 ± 0.01 tok/s | 46.66 ± 0.55 tok/s | 13.673× |
| Wall time | 50.03 ± 0.10 s | 3.85 ± 0.05 s | 0.077× |
| Peak memory | 4.50 GB | 1.93 GB | 0.428× |

Caveat:

The 1.5B result is N=2 with one request per repetition. It should be presented
as a smoke benchmark.

## 3. Ready for End of August

Status: summarized, no extra experiment required.

Main story:

By the end of August, the project can say:

1. The original bottleneck was measured and profiled.
2. The self Stage 1 baseline matches the reference implementation.
3. The first Stage 2 fallback path removes full-context decode and improves
   decode throughput strongly.
4. The next unresolved question is attribution beyond the fallback path:
   custom quantized matmul, custom Flash Attention, and batching.

Evidence:

- `docs/PHASE2_KV_CACHE_CORRECTNESS_2026-05-29.md`
- `docs/PHASE2_KV_FALLBACK_BENCHMARK_2026-05-29.md`
- `docs/RESEARCH_PROGRESS_SUMMARY_2026-05-29.md`

Recommended next decision:

Choose whether to install full Xcode and start custom C++/Metal kernels, or
stay with MLX fallback and do a cleaner attribution ablation first.

## 4. Memory Notes

The August experiments completed without needing special memory cleanup.

Observed peak memory:

- 0.5B Stage 1 full-context: 1.89 GB
- 0.5B Stage 2 KV fallback: 0.93 GB
- 1.5B Stage 1 full-context: 4.50 GB
- 1.5B Stage 2 KV fallback: 1.93 GB

No 7B or long-context experiment should be run without closing memory-heavy
apps first.

# SiliconLLM September Progress Ledger

Date range: 2026-09-01 to 2026-09-30

## Meetings

- 2026-09-16: Stage 2 attribution benchmark design
- 2026-09-30: Stage 2 attribution result and Xcode/C++/Metal decision point

## September Narrative

September should not repeat the August benchmark with the same two arms. The
August result already shows that the Stage 2 KV-cache fallback removes the main
full-context decode bottleneck. The next useful step is attribution: separate
how much of the improvement comes from KV-cache decode, how much comes from the
self implementation stack, and how much gap remains against mlx-lm.

## Attribution Benchmark

The four-arm attribution benchmark has now been run on Qwen2 0.5B:

| Arm | Purpose |
|---|---|
| Stage 1 full-context reference | Original baseline and bottleneck anchor |
| Stage 2 full-context self path | Implementation-stack control without KV-cache decode |
| Stage 2 KV-cache fallback | Current correctness-first optimized path |
| mlx-lm | Production engine reference |

Main result:

| Arm | Decode tok/s | Wall time |
|---|---:|---:|
| Stage 1 full-context ref | 7.16 | 56.84 s |
| Stage 2 full-context | 7.69 | 52.87 s |
| Stage 2 KV-cache | 65.65 | 6.34 s |
| mlx-lm | 189.43 | 2.30 s |

The key ratios are now measured:

| Ratio | Interpretation |
|---|---|
| Stage 2 full-context / Stage 1 full-context = 1.073x decode | Self stack and MLX fallback alone are a small effect |
| Stage 2 KV-cache / Stage 2 full-context = 8.542x decode | Persistent KV-cache is the dominant gain |
| mlx-lm / Stage 2 KV-cache = 2.886x decode | A production-engine gap remains |

## Meeting 1: 2026-09-16

Recommended title:

```text
Stage 2 Attribution: Separating KV-Cache Gain from Implementation Stack Effects
```

Main message:

The August benchmark establishes that Stage 2 fallback decode is much faster
than Stage 1 full-context decode. The September deliverable is the four-arm
attribution design and the benchmark script used to measure it.

Deliverable:

- `docs/GROUP_MEETING_REPORT_2026-09-16_EN.md`

## Meeting 2: 2026-09-30

Recommended title:

```text
Stage 2 Decision Point: Attribution First, Then C++/Metal
```

Main message:

The next technical decision is whether the remaining 2.886x mlx-lm-over-Stage
2-KV-cache decode gap is large enough to justify Xcode and custom C++/Metal
work immediately. The safest sequence is:

1. profile the Stage 2 KV-cache decode path,
2. add a Qwen2 1.5B narrow attribution smoke variant if memory is stable,
3. decide whether the next implementation target should be custom kernels,
   batching/chunked prefill, or a Stage 3 speculative decoding prototype.

Deliverable:

- `docs/GROUP_MEETING_REPORT_2026-09-30_EN.md`

## Memory Guidance

No memory release is required for writing the September plan. Before running the
actual attribution benchmark:

- Qwen2 0.5B N=3 should be acceptable under normal memory pressure.
- Qwen2 1.5B should remain a narrow smoke benchmark unless other large apps are
  closed.
- Qwen2 7B or longer-context Stage 1 runs should wait until browser, IDE, and
  chat apps are closed.

## Evidence

- `docs/PHASE2_ATTRIBUTION_BENCHMARK_2026-05-30.md`
- `benches/stage2_attribution_benchmark.py`
- `benches/stage2_attribution_benchmark_summary.csv`
- `benches/stage2_attribution_benchmark_ratios.csv`
- `benches/stage2_attribution_benchmark.png`

## Status

Measured on Qwen2 0.5B N=3. The Qwen2 1.5B attribution variant remains optional.

# Group Meeting Report: Stage 2 Decision Point

Date: 2026-09-30

## Goal

The goal for this checkpoint is to decide whether the project should move
directly into custom C++/Metal implementation, or first use the new attribution
result to choose a narrower next step.

## Current State

The project has three layers of evidence:

| Layer | Status |
|---|---|
| Phase 0 diagnosis | Full-context decode is the main bottleneck |
| Stage 1 self baseline | Self implementation matches the reference path closely |
| Stage 2 KV-cache fallback | Decode speed improves strongly on 0.5B and 1.5B smoke |
| Stage 2 attribution | KV-cache contribution isolated on Qwen2 0.5B |

This is enough to justify Stage 2 as the right direction. The attribution result
now makes the next decision sharper.

## Attribution Result

Qwen2 0.5B, N=3:

| Arm | Decode tok/s | Wall time |
|---|---:|---:|
| Stage 1 full-context ref | 7.16 | 56.84 s |
| Stage 2 full-context | 7.69 | 52.87 s |
| Stage 2 KV-cache | 65.65 | 6.34 s |
| mlx-lm | 189.43 | 2.30 s |

Key ratios:

| Comparison | Decode ratio | Wall-time ratio |
|---|---:|---:|
| Stage 2 full-context / Stage 1 ref | 1.073x | 0.930x |
| Stage 2 KV-cache / Stage 2 full-context | 8.542x | 0.120x |
| mlx-lm / Stage 2 KV-cache | 2.886x | 0.362x |
| Stage 2 KV-cache / Stage 1 ref | 9.168x | 0.111x |

The result says that the large improvement comes from persistent KV-cache
decode, not merely from the Stage 2 implementation stack.

## Decision Options

### Option A: Profile Stage 2 KV-Cache First

Use the attribution result to profile the remaining Stage 2 KV-cache decode
path:

1. attention and KV-cache update,
2. quantized projections,
3. MLP,
4. LM head,
5. elementwise/norm/reshape work,
6. Python token loop.

### Option B: Install Xcode and Start C++/Metal

This unlocks custom extension work, but it has higher setup cost and more
failure modes:

- Xcode.app is a large install,
- Metal shader compilation adds toolchain uncertainty,
- kernel work can consume time before a clean research result appears.

## Recommendation

Choose Option A first.

The September result already gives the attribution breakdown. The remaining
question is why mlx-lm is still 2.886x faster than the Stage 2 KV-cache path.
A Stage 2 profiler will make the later C++/Metal work easier to justify.

## Proposed October Entry Criterion

Move into custom C++/Metal work only if profiling shows that the remaining gap
to mlx-lm is dominated by lower-level kernel behavior rather than Python loop,
cache bookkeeping, or benchmark harness overhead.

## Memory Guidance

For the actual benchmark run:

- the Qwen2 0.5B N=3 attribution benchmark is complete,
- keep Qwen2 1.5B attribution as N=2 smoke,
- do not run Stage 1 7B or longer-context variants unless memory-heavy apps are
  closed first.

## Evidence References

- `docs/SEPTEMBER_PROGRESS_LEDGER_2026.md`
- `docs/GROUP_MEETING_REPORT_2026-09-16_EN.md`
- `docs/PHASE2_ATTRIBUTION_BENCHMARK_2026-05-30.md`
- `docs/AUGUST_MONTH_END_SUMMARY_2026-08-31_EN.md`
- `benches/research_progress_unified_summary.csv`

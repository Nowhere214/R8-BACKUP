# SiliconLLM Group Meeting Update

Date: 2026-07-15

## What I Worked On

Since the formal 7/1 progress update, I focused on preparing the project for
the next phase and starting the smallest correctness-first implementation
step.

The main work was:

- cleaning up the Phase 0 to Phase 1 result record
- adding a research-level progress ledger for July
- auditing the Stage 2 implementation path before starting KV-cache work
- implementing a Stage 2 KV-cache correctness path with an MLX quantized
  fallback

This is a small but useful transition step. The project now has a clearer
boundary between measurement, baseline implementation, KV-cache correctness,
and custom kernel optimization.

## Result Package Cleanup

The 7/1 update already had three main pieces:

1. Phase 0 measured the decode bottleneck.
2. Phase 0.5 profiled the Stage 1 decode step.
3. Phase 1 implemented a faithful Stage 1 baseline and checked it against the
   reference implementation.

I turned this into a broader research-level table:

- `benches/build_research_progress_summary.py`
- `benches/research_progress_unified_summary.csv`
- `docs/RESEARCH_PROGRESS_SUMMARY_2026-05-29.md`

The table currently has 175 data rows:

| Phase | Rows |
|---|---:|
| Phase 0 evidence package | 151 |
| Phase 1 self-baseline checks | 24 |

The important detail is that the table keeps a `phase_id` column. Phase 0 and
Phase 1 are in the same auditable file, but they are not mixed together
conceptually.

## Baseline Check Recap

The self-implemented Stage 1 baseline is close to the reference path.

| Model | Check | Decode ratio |
|---|---|---:|
| Qwen2 0.5B | N=3 benchmark | 1.004× |
| Qwen2 1.5B | N=2 smoke benchmark | 1.001× |

For both models, the first 16 greedy generated token IDs matched exactly
between the self implementation and the reference implementation.

The 1.5B check is only a smoke benchmark, not a full N=3 benchmark. I am using
it as consistency evidence, not as a final performance claim.

## Phase 2 Readiness Audit

Before starting Stage 2, I audited the local Stage 2 materials:

- `book/src/week2-overview.md`
- `book/src/week2-01-kv-cache.md`
- `book/src/week2-02-quantized-matmul.md`
- `book/src/week2-04-flash-attention.md`
- `tests_refsol/test_week_2_day_1.py`
- `tests_refsol/test_week_2_day_2.py`
- `tests_refsol/test_week_2_day_4.py`
- `tests_refsol/test_week_2_day_6.py`
- `src/tiny_llm_ref/qwen2_week2.py`
- `src/tiny_llm_ref/kv_cache.py`

The audit is recorded in:

- `docs/PHASE2_READINESS_AUDIT_2026-05-29.md`

The main finding is that Stage 2 has two different kinds of work:

1. algorithmic serving work, mainly KV cache and batching
2. toolchain-dependent kernel work, mainly quantized matmul and Flash Attention

These should not be mixed into one vague "optimization" task.

## KV-Cache Correctness Path

After the audit, I implemented the first correctness-first Stage 2 path.

The new implementation work is recorded in:

- `docs/PHASE2_KV_CACHE_CORRECTNESS_2026-05-29.md`

Implemented pieces:

- `TinyKvFullCache`
- `BatchingKvCache`
- Stage 2 Qwen2 attention, MLP, block, and model wrapper
- `simple_generate_with_kv_cache`
- `quantized_linear` using MLX built-in quantized matmul as a fallback
- `flash_attention` using an MLX fallback wrapper for small correctness checks

This is not a custom kernel implementation yet. It is a correctness path that
lets me test KV-cache behavior before taking on extension work.

Validation so far:

| Scope | Result |
|---|---|
| Stage 2 Day 1 Qwen2 0.5B subset | 3 passed, 6 deselected |
| Stage 2 Day 1 Qwen2 1.5B subset | 3 passed, 6 deselected |
| Stage 2 Day 2 quantized matmul small tests | 6 passed |
| Stage 2 Day 6 batching cache unit test | 1 passed, 25 deselected |
| Stage 2 Day 4 flash fallback CPU small sample | 1 passed, 19 deselected |

The Day 1 model tests include both full prompt forward correctness and
step-by-step cached forward correctness against the MLX model output.

## Current Risk

The main risk is still the local extension toolchain.

Earlier checks showed that the machine does not have the full Xcode Metal
toolchain available:

```text
xcrun: error: unable to find utility "metal"
```

That affects the custom C++/Metal extension path for quantized matmul and Flash
Attention.

So the next phase should not start by trying to build all Stage 2 kernels at
once. That would mix implementation risk with environment risk.

## Next Step

The next implementation step should be a small benchmark of the fallback
KV-cache path.

The recommended order is:

1. compare Stage 1 vs Stage 2 fallback decode on Qwen2 0.5B
2. keep the result labeled as MLX quantized fallback
3. measure decode throughput and per-token wall time
4. only then decide whether to move toward custom C++/Metal kernels

This keeps the attribution clean. If decode improves later, it should be clear
whether the gain came from KV cache, quantized matmul, Flash Attention, or
batching.

## Short Version for the Meeting

After the 7/1 formal update, I did not jump directly into custom kernels. I
first organized the Phase 0 and Phase 1 results into one research-level table,
then audited the Stage 2 path, and then implemented the first KV-cache
correctness path with an MLX quantized fallback.

The key point is that Phase 2 is now split cleanly: KV-cache correctness first,
custom C++/Metal kernels later as a separate toolchain-dependent milestone.

This makes the next step smaller and easier to evaluate: benchmark the fallback
KV-cache path first, then measure what bottleneck remains.

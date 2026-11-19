# Phase 2 Readiness Audit

Prepared on: 2026-05-29

Target reporting window: 2026-07-15

## Goal

This document records what needs to be true before starting Phase 2
implementation.

The purpose is not to claim Phase 2 optimization is already done. The purpose
is to turn the next phase into a clear checklist, so the 2026-07-15 group
meeting can report real planning and risk reduction rather than vague
"starting optimization" language.

Update: the first correctness-first Stage 2 path has now started. See
`docs/PHASE2_KV_CACHE_CORRECTNESS_2026-05-29.md` for implementation and test
results. This audit remains useful as the risk map.

## Local Sources Reviewed

Book chapters:

- `book/src/week2-overview.md`
- `book/src/week2-01-kv-cache.md`
- `book/src/week2-02-quantized-matmul.md`
- `book/src/week2-04-flash-attention.md`
- `book/src/week2-06-prefill-and-batch.md`

Code paths:

- `src/tiny_llm/kv_cache.py`
- `src/tiny_llm/qwen2_week2.py`
- `src/tiny_llm/quantize.py`
- `src/tiny_llm/generate.py`
- `src/tiny_llm_ref/kv_cache.py`
- `src/tiny_llm_ref/qwen2_week2.py`
- `src/tiny_llm_ref/quantize.py`

Tests:

- `tests_refsol/test_week_2_day_1.py`
- `tests_refsol/test_week_2_day_2.py`
- `tests_refsol/test_week_2_day_4.py`
- `tests_refsol/test_week_2_day_6.py`

## Stage 2 Task Map

| Area | Local chapter | Main files | Test entrypoint | Toolchain risk |
|---|---|---|---|---|
| KV cache | Stage 2 Day 1 | `kv_cache.py`, `qwen2_week2.py`, `generate.py` | `pdm run test --week 2 --day 1` | Medium |
| Quantized matmul | Stage 2 Day 2-3 | `quantize.py`, `src/extensions/*` | `pdm run test --week 2 --day 2 -- -k cpu` | High |
| Flash Attention | Stage 2 Day 4-5 | `attention.py`, `src/extensions/*` | `pdm run test --week 2 --day 4 -- -k cpu` | High |
| Continuous batching | Stage 2 Day 6 | `batch.py`, `kv_cache.py`, `qwen2_week2.py` | `pdm run test --week 2 --day 6` | High |

## Current Implementation State

The Stage 1 self implementation is now usable as the baseline.

Phase 2 files are still mostly stubs:

- `TinyKvFullCache.update_and_fetch` is not implemented.
- `BatchingKvCache.update_and_fetch`, `add_request`, and `remove_request` are
  not implemented.
- `Qwen2ModelStage 2` and its attention, MLP, and block classes are not
  implemented.
- `quantized_matmul` and `quantized_linear` are not implemented in
  `src/tiny_llm/quantize.py`.
- `simple_generate_with_kv_cache` is not implemented.

The reference implementation provides a clear local template, but copying it
directly is not the research goal. It should be used to understand API
contracts and test expectations.

## Important Constraint

Stage 2 cannot be treated as one single implementation task.

The KV-cache model path depends on quantized weights. In the reference path,
`qwen2_week2.py` uses `quantized_linear`, which calls the custom extension.
Earlier environment checks showed that building Stage 2 extensions is blocked by
the missing full Xcode Metal toolchain:

```text
xcrun: error: unable to find utility "metal"
```

This means Phase 2 has two possible entry strategies.

## Recommended Entry Strategy

### Strategy A: correctness-first Python fallback

Implement a fallback path for Stage 2 that uses MLX built-ins or dequantized
weights first, so KV-cache correctness can be validated before custom kernels.

Pros:

- keeps Phase 2 unblocked without installing Xcode immediately
- connects directly to the Phase 0/0.5 finding that KV cache is the first
  optimization target
- allows a Stage 2 Day 1 correctness story before kernel work

Cons:

- performance will not represent final Stage 2 optimization
- must be labeled as a fallback/correctness path, not as the final optimized
  path

### Strategy B: install Xcode and follow Stage 2 order

Install full Xcode, then implement quantized matmul and Flash Attention as
custom C++/Metal extensions.

Pros:

- follows the course structure directly
- makes CPU/GPU extension tests meaningful
- prepares for real optimized kernels

Cons:

- large disk cost
- higher setup risk
- can distract from the clean 7/1 and 7/15 reporting story

## Recommendation

For 2026-07-15, use Strategy A as the planning result:

1. Keep the Stage 1 self baseline as the fixed reference point.
2. Start Phase 2 with KV-cache correctness, using a fallback path if needed.
3. Treat custom quantized matmul and Flash Attention as separate later
   milestones gated by the Xcode/toolchain decision.

This keeps the research sequence coherent:

1. Phase 0 measured the decode bottleneck.
2. Phase 0.5 explained that the bottleneck is full-context repeated compute.
3. Phase 1 built and validated a faithful baseline.
4. Phase 2 should first remove full-context decode with KV cache.

## Concrete Next Steps

Safe next implementation steps:

1. Implement `TinyKvFullCache.update_and_fetch`.
2. Add a tiny unit or script-level check for cache concatenation and offset.
3. Decide whether Stage 2 `quantized_linear` should use a temporary MLX fallback
   before custom extension work.
4. Only then implement `Qwen2MultiHeadAttention` in `qwen2_week2.py`.

Avoid for now:

- running full Stage 2 model tests with Qwen2 7B
- installing Xcode during a reporting window
- presenting Stage 2 throughput until the path is clearly labeled and measured

## Meeting-Ready Summary

After 7/1, the next step is not immediately "optimize everything." I audited
the Stage 2 path and found that Phase 2 has two separate risks: the algorithmic
change, KV cache, and the toolchain-dependent kernel work, quantized matmul and
Flash Attention.

For the next phase, I plan to separate them. I will first implement and test
the KV-cache correctness path, possibly with a fallback implementation, and
only then move to custom C++/Metal kernels. This keeps the optimization story
incremental and makes each result easier to attribute.

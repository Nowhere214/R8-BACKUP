# Phase 2 KV-Cache Correctness Progress

Date: 2026-05-29

Target reporting window: 2026-07-15

## Goal

This records the first Phase 2 correctness-first implementation pass.

The goal was not to claim a final optimized Stage 2 serving stack. The goal was
to make the KV-cache path executable and testable without relying on the local
C++/Metal extension toolchain.

## Implemented Files

| File | Implemented content |
|---|---|
| `src/tiny_llm/kv_cache.py` | `TinyKvFullCache`, `BatchingKvCache`, cache offset tracking, rewind |
| `src/tiny_llm/quantize.py` | `quantized_matmul` and `quantized_linear` using MLX built-in quantized matmul fallback |
| `src/tiny_llm/qwen2_week2.py` | Stage 2 Qwen2 attention, MLP, block, model wrapper with per-layer KV cache |
| `src/tiny_llm/generate.py` | `simple_generate_with_kv_cache` greedy generation path |
| `src/tiny_llm/attention.py` | `flash_attention` MLX fallback wrapper for correctness checks |

## Important Boundary

This is a correctness fallback path.

The current `quantized_matmul` uses MLX built-in `mx.quantized_matmul`. It does
not use the project's custom C++/Metal extension, because the local full Xcode
Metal toolchain is still unavailable.

So this pass should be presented as:

- KV-cache correctness path started
- Stage 2 model path is now executable for small Qwen2 checks
- custom kernel optimization remains a later milestone

It should not be presented as:

- final Stage 2 optimized serving
- custom quantized matmul implementation
- custom Flash Attention implementation

## Validation

Commands run:

```bash
pdm run python -m py_compile src/tiny_llm/kv_cache.py src/tiny_llm/quantize.py src/tiny_llm/qwen2_week2.py src/tiny_llm/generate.py src/tiny_llm/attention.py
pdm run test --week 2 --day 2 -- -k cpu
pdm run test --week 2 --day 1 -- -k qwen_2_05b
pdm run test --week 2 --day 2
pdm run test --week 2 --day 6 -- -k test_task_2_batching_kv_cache
pdm run test --week 2 --day 4 -- -k "test_task_2_flash_attention_cpu_small and no_mask"
pdm run test --week 2 --day 1 -- -k qwen_2_15b
```

Results:

| Scope | Result |
|---|---|
| Python syntax check | passed |
| Stage 2 Day 2 CPU quantized matmul fallback | 4 passed, 2 deselected |
| Stage 2 Day 1 Qwen2 0.5B subset | 3 passed, 6 deselected |
| Stage 2 Day 2 full small quantized matmul tests | 6 passed |
| Stage 2 Day 6 batching cache unit test | 1 passed, 25 deselected |
| Stage 2 Day 4 flash fallback CPU small no-mask sample | 1 passed, 19 deselected |
| Stage 2 Day 1 Qwen2 1.5B subset | 3 passed, 6 deselected |

The Day 1 model tests include both full prompt forward correctness and
step-by-step cached forward correctness against the MLX model output.

## What Was Not Run

I did not run:

- Qwen2 7B Stage 2 tests
- full Stage 2 Day 4 Flash Attention suite
- full Stage 2 Day 6 continuous batching model tests
- custom extension build or extension tests
- Stage 2 throughput benchmarks

Reasons:

- 7B remains a memory-risk path on this 18 GB machine.
- full Flash Attention and continuous batching tests are not the next
  correctness boundary.
- extension build remains blocked by the missing full Xcode Metal toolchain.
- throughput claims should wait until the implementation path is clearly
  defined and measured.

## Current Interpretation

This is a real Phase 2 step, but a narrow one.

The project now has an executable Stage 2 correctness path for Qwen2 0.5B and
1.5B using MLX built-in quantized matmul. That is enough to say Phase 2 has
started in a controlled way.

The next research question is still attribution:

If decode improves, which change caused the improvement?

The correct order remains:

1. validate KV-cache correctness
2. benchmark the fallback Stage 2 path against Stage 1
3. only then replace fallback pieces with custom kernels

## Next Steps

Update: the first Stage 1 vs Stage 2 fallback benchmark has now been completed.
See `docs/PHASE2_KV_FALLBACK_BENCHMARK_2026-05-29.md`.

Recommended next order:

1. Design one attribution benchmark to separate KV-cache effects from
   quantized-execution effects.
2. Keep current results clearly labeled as MLX quantized fallback, not custom
   kernel work.
3. Defer custom C++/Metal quantized matmul and Flash Attention until the
   toolchain decision is explicit.

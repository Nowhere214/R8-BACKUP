# SiliconLLM

> A personal research fork of [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm).
> Goal: build an LLM inference engine on Apple Silicon from scratch (using only MLX array APIs),
> and benchmark it against production engines (vLLM, llama.cpp, mlx-lm) as a **Phase 0** baseline.

Upstream project by [Chi (skyzh)](https://github.com/skyzh) and
[Connor](https://github.com/Connor1996), licensed under CC BY-NC-SA 4.0
(course material) and the Apache License 2.0 (code, see `LICENSE`).

The internal Python package is still named `tiny_llm` / `tiny_llm_ref` to keep
the upstream test suite working. Renaming it would require rewriting every
test and reference implementation, which isn't useful for research work.
Only the *project identity* (this README, the `pyproject.toml` name, etc.) has
been renamed to `silicon-llm`.

---

## What this project is

An educational but end-to-end LLM serving stack for Apple Silicon, implemented
with **only MLX matrix APIs** — no high-level NN module reuse. By the end of
Week 2, it implements a minimal vLLM-style engine:

| Week | Capability                                    | Status (upstream) |
| ---- | --------------------------------------------- | ----------------- |
| 1    | Attention, RoPE, GQA, RMSNorm, MLP, decoding, sampling | complete       |
| 2    | KV cache, quantized matmul (CPU+GPU), Flash Attention 2, continuous batching, chunked prefill | complete |
| 3    | Paged attention, MoE, speculative decoding, RAG, tool calling, long context | in progress    |

## Research roadmap (mine)

1. **Phase 0 — Baseline benchmarking** (see [`docs/PHASE0_BENCHMARK.md`](docs/PHASE0_BENCHMARK.md))
   Measure Qwen2 inference on the same Mac against three reference engines:
   - `mlx-lm`       (same framework as SiliconLLM, production-optimized)
   - `llama.cpp`    (the de-facto Apple-Silicon engine, C++ + Metal)
   - `vLLM`         (CPU/MPS build — the gold standard on GPU, included for API/throughput comparison)
2. **Phase 1** — Implement Week 1 (pure Python) and rerun the same benchmark.
3. **Phase 2** — Implement Week 2 (Metal kernels, flash attention, continuous batching) and rerun.
4. **Phase 3** — Pick *one* Week-3 topic to extend beyond upstream (e.g. paged attention
   with a different block size, or a custom speculative decoding policy) and measure its impact.

## Quick start

```bash
# One-time setup
pdm install            # create virtualenv + install mlx, torch, mlx-lm, ...
pdm run check-installation
pdm run test-refsol -- -- -k week_1

# Run a reference (complete) implementation end-to-end
hf download Qwen/Qwen2-0.5B-Instruct-MLX
pdm run main --solution ref --loader week1

# Run my own implementation's tests
pdm run test --week 1 --day 1

# Benchmark (upstream bench script, measures prefill / decode throughput)
pdm run bench --model qwen2-0.5b --solution ref --loader week2
```

## Directory layout (unchanged from upstream)

```
src/tiny_llm/            # my implementation (you'll fill this in)
src/tiny_llm_ref/        # reference implementation
src/extensions/          # C++/Metal kernels (week 2, my code)
src/extensions_ref/      # reference C++/Metal kernels
tests/                   # tests against src/tiny_llm
tests_refsol/            # tests against src/tiny_llm_ref
book/                    # the mdbook source (the course guide)
bench.py                 # throughput benchmark (prefill/decode tok/s)
main.py                  # interactive chat entrypoint
batch-main.py            # batched serving entrypoint (week 2)
```

See the original book at <https://skyzh.github.io/tiny-llm/> for task-by-task
instructions; this fork keeps that structure intact.

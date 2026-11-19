# Phase 0: Baseline Benchmarking

> Per advisor feedback: *"Add a Phase 0 that benchmarks vLLM and llama.cpp to establish measurable metrics before we build anything."*

This document defines:
1. **What** we measure
2. **Against which baselines** (and which ones actually make sense on Apple Silicon)
3. **How** we measure (hardware, model, workload)
4. **What numbers we measured** — and what those numbers imply for Phase 1/2

It also serves as the "spec" for the SiliconLLM side: once Phase 1 and Phase 2 are done,
we must re-run the **same** script on SiliconLLM and compare.

Status as of 2026-05-29: the main Phase 0 baselines have been measured
for SiliconLLM Stage 1 ref, mlx-lm, and llama.cpp Metal. vLLM-CPU is not
included in the current Phase 0 result set because it is not a meaningful
Apple-Silicon production baseline. The separate CSV outputs have also been
collected into one long-format reproducibility table. Measured numbers are
recorded in `docs/PHASE0_RESULTS.md`.

---

## 1. Is a fair comparison even possible?

**Short answer: yes, for most metrics — but with caveats.**

| Engine       | Hardware target              | Kernel backend   | On Apple Silicon M-series? |
| ------------ | ---------------------------- | ---------------- | -------------------------- |
| SiliconLLM   | Apple Silicon (MLX)          | MLX + custom Metal | native                     |
| mlx-lm       | Apple Silicon (MLX)          | MLX + fused Metal  | native (same framework)    |
| llama.cpp    | CPU / Metal / CUDA / ROCm    | GGML + Metal     | native (the standard engine) |
| vLLM         | CUDA (primary) / CPU / ROCm / MPS (experimental) | CUDA kernels / Triton | **partial**: CPU build works, MPS backend is experimental and slow |

### Implications for the three baselines

- **mlx-lm** — the **most apples-to-apples** comparison. Same framework, same weights
  (`Qwen2-7B-Instruct-MLX`), same int4 quantization, same Metal graph compiler.
  Any gap vs. mlx-lm is purely *our* code quality.
- **llama.cpp** — the **most production-relevant** Apple-Silicon baseline. Uses a
  completely different runtime (GGML instead of MLX). Requires GGUF weights
  (different quantization — Q4_K_M is the closest practical match to MLX int4).
  This is the engine real macOS users actually run, so beating/approaching it
  is the meaningful goal.
  **Install note**: building llama.cpp from source with Metal support needs
  the `metal` shader compiler from full Xcode.app (same issue as our own
  extensions). Phase 0 uses the **pre-built `llama-*-macos-arm64.zip`
  release** from <https://github.com/ggerganov/llama.cpp/releases>, which
  ships a pre-compiled Metal library. This lets us include llama.cpp without
  the 15 GB Xcode download.
- **vLLM** — trickier. vLLM is the *reference* for batched LLM serving, but its
  CUDA kernels are the whole point. On Mac we have two options, both imperfect:
    - (a) vLLM **CPU build** — runs but is slow, isn't what people use vLLM for.
    - (b) Run vLLM on a rented cloud GPU (e.g. an A10G) and normalize by FLOPs.
  Current Phase 0 decision: do **not** include vLLM-CPU unless an advisor
  explicitly asks for a vLLM comparison. "Beating vLLM-CPU" would not be a
  meaningful Apple-Silicon result. If we need vLLM later, the cleaner option is
  a small cloud-GPU Phase 0.5 measurement.

**Current recommendation**: keep Phase 0 focused on mlx-lm and llama.cpp as
the real production baselines, with SiliconLLM Stage 1 reference as the lower
internal baseline.

---

## 2. Metrics to measure

These mirror what upstream `bench.py` already reports, so we can compare directly:

| Metric                     | Definition                                                          | Why it matters              |
| -------------------------- | ------------------------------------------------------------------- | --------------------------- |
| **TTFT** (prefill latency) | Time from request arrival until the first generated token           | User-perceived latency      |
| **Prefill throughput**     | `prompt_tokens / prefill_time` (tok/s)                              | Prefill-bound workloads (long prompts, short replies) |
| **ITL** (inter-token latency) | Median time between successive generated tokens                  | Streaming "typing" speed    |
| **Decode throughput**      | `generated_tokens / decode_time` (tok/s)                            | Decode-bound workloads      |
| **End-to-end throughput**  | `total_tokens / total_time` (tok/s), averaged over many requests    | Server capacity proxy       |
| **Peak memory**            | Max RSS (via `/usr/bin/time -l` on macOS)                           | Feasibility on consumer Macs |
| **Output quality spot-check** | Greedy decode N prompts, compare text output                     | Correctness sanity (should be ~identical across engines with `temperature=0`, same seed is hopeless across engines but text should be semantically equivalent) |

Not measured in Phase 0 (added later):
- Batched throughput under load (needs Stage 2's continuous batching done)
- Cold-start vs. warm latency split
- Power draw (interesting on Mac but requires `powermetrics`, out of scope)

Metric caveat added after the llama.cpp workload scan: llama.cpp
`llama-bench -pg <prompt,gen>` reports mixed prompt+generation throughput,
not pure decode throughput. It should not be plotted as a decode-only metric
next to MLX `stream_generate` measurements.

---

## 3. Experimental setup

- **Hardware**: Apple Silicon MacBook Pro, 18 GB unified memory.
- **Models**: Qwen2 0.5B, 1.5B, and 7B where feasible.
  Quantization per engine:
    - SiliconLLM / mlx-lm: Qwen2 MLX int4 models
    - llama.cpp: Qwen2 GGUF Q4_K_M models
    - vLLM-CPU: currently not included
- **Main model-size workload**:
    - SiliconLLM Stage 1 ref / mlx-lm: 4 synthetic requests, input length
      sampled from [64, 256], output length sampled from [64, 256], seed 0,
      1 warmup run, N=3.
    - llama.cpp: `llama-bench -p 128 -n 128 -r 3` using the pre-built b9354
      macOS arm64 release with Metal + BLAS.
- **Prompt-length scan workload**:
    - Prompt length: 64, 128, 512 tokens
    - Generation length: 64 tokens
    - Number of requests: 2
    - Repetitions: N=2
- **Reporting rule**: report mean ± standard deviation, and keep raw CSV files
  under `benches/`.

**Disk budget rule**:

The measured Phase 0 setup already includes MLX models, GGUF models, the
pre-built llama.cpp release, and the project `.venv`. Do not add vLLM CPU
or bf16 7B weights unless there is a specific reason; that would consume
multiple additional GB while adding a weak comparison point.

Before and after large downloads or Git-heavy operations, check for stale
Git temporary pack files as described in `AGENTS.md`.

---

## 4. Measured Phase-0 baseline summary

The original pre-run expectation table has been superseded by measured data.
The current summary is:

| Engine | Model | Prefill tok/s | Decode tok/s |
| ------ | ----- | ------------: | -----------: |
| SiliconLLM Stage 1 ref | Qwen2 0.5B | 2,539 ± 47 | 9.98 ± 0.40 |
| SiliconLLM Stage 1 ref | Qwen2 1.5B | 998 ± 0.7 | 3.86 ± 0.004 |
| mlx-lm | Qwen2 0.5B | 4,015 ± 8 | 291.6 ± 0.08 |
| mlx-lm | Qwen2 1.5B | 1,276 ± 12 | 126.4 ± 0.18 |
| mlx-lm | Qwen2 7B | 287 ± 0.07 | 32.4 ± 0.007 |
| llama.cpp Metal | Qwen2 0.5B Q4_K_M | 2,924.7 ± 36.2 | 137.3 ± 1.0 |
| llama.cpp Metal | Qwen2 1.5B Q4_K_M | 1,103.7 ± 8.3 | 71.9 ± 0.5 |
| llama.cpp Metal | Qwen2 7B Q4_K_M | 266.2 ± 2.0 | 22.2 ± 0.03 |

Main interpretation:

- The largest gap is decode, not prefill.
- The same-framework gap between Stage 1 ref and mlx-lm is about 29–33× on
  decode for 0.5B and 1.5B.
- llama.cpp Metal is much faster than Stage 1 ref, but still slower than
  mlx-lm on decode in this setup.
- Stage 1 ref cannot run 7B on the 18 GB machine because it dequantizes
  quantized weights at load.
- A separate prompt-length scan shows the decode gap grows from about
  14–15× at 64 prompt tokens to about 58× at 512 prompt tokens.
- A synthetic KV-cache microbenchmark supports the mechanism: full-context
  recompute gets much slower with context length, while cached one-token
  decode stays close to flat.
- A follow-up cost-model analysis estimates that Stage 1 ref adds about
  0.327 ms per extra context token at 0.5B and 0.977 ms at 1.5B, while
  mlx-lm is close to flat in the same scan.
- A reproducibility pass collects the current Phase 0 evidence into
  `benches/phase0_unified_summary.csv`, a 151-row long-format table with
  source files, units, repetition counts, and metric caveats.

Measured details live in `docs/PHASE0_RESULTS.md`.

---

## 5. Deliverables and current implementation status

Completed artifacts:

1. `benches/biweekly_n3.py` for the Stage 1 ref and mlx-lm N=3 comparison.
2. `benches/llama_cpp/llama_cpp_workload_length_scan.py` and the llama.cpp
   b9354 CSVs for the production-engine Metal baseline.
3. `benches/workload_length_scan.py` for prompt-length decode scans.
4. `benches/kv_cache_microbench.py` for the controlled KV-cache mechanism
   check.
5. `benches/analyze_decode_gap_scaling.py` for the context-length decode
   cost model.
6. `benches/build_phase0_unified_summary.py` and
   `benches/phase0_unified_summary.csv` for the reproducibility manifest.
7. Plot scripts under `benches/` and `benches/llama_cpp/`, with scientific
   figures exported as PNG/PDF/SVG/TIFF where applicable.

Still worth doing later:

1. Consolidate the benchmark scripts into one CLI, for example
   `benches/phase0_compare.py`.
2. Extend the unified long-format schema into the measurement scripts instead
   of only building it as a post-processing step.
3. Keep decode-only metrics separate from mixed prompt+generation throughput.

---

## 6. Open questions (to discuss with advisor)

1. Should vLLM be excluded from Phase 0 on the grounds that vLLM-on-Mac CPU
   is not a meaningful production baseline? Current recommendation: exclude
   it unless the advisor specifically asks for a vLLM number.
2. Should future cross-engine comparisons report both practical quantized
   results and stricter same-precision results, or is practical int4 enough
   for this project?
3. How far should the prompt-length scan go before Phase 1? Current
   recommendation: keep the 64/128/512-token result for the next progress
   report, then use profiler work rather than spending another week on a
   larger sweep.
4. Should the next mechanism study be MLX profiling of the Stage 1 decode path,
   or a non-extension Stage 2/KV-cache fallback? Current recommendation:
   profile first; defer full Stage 2 reference measurement until Xcode.app is
   needed for Phase 2.

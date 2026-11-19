# SiliconLLM Project — Month-1 Research Progress Report

**Topic**: Phase 0 — Establishing a benchmark methodology and the first quantitative characterization of the optimization gap in LLM inference
**Reporting period**: late March 2026 — late April 2026
**Author**: WANG PEIRAN

---

## Abstract

This month completes the Phase 0 stage of this research. The study adopts the third-party open-source educational project `skyzh/tiny-llm` as a controlled experimental platform, upon which I have established a personal research fork named **SiliconLLM**. A point that must be stated upfront: tiny-llm is itself an LLM-inference-systems course authored by others, and its content consists of **didactic re-implementations of techniques already published in the literature** (attention, KV cache, Flash Attention, PagedAttention, etc.); **its codebase does not constitute a research contribution of this project**. The research contributions of this project are (i) the measurement framework designed on top of that platform, (ii) the empirical data obtained through it, and (iii) the data-driven research trajectory inferred from those measurements.

Concrete deliverables produced this month under that framework:

1. **Methodology design** — Following my Koike Sensei's guidance at the end of last month ("before building any optimization, first benchmark vLLM / llama.cpp to establish measurable metrics"), I designed a reproducible LLM-inference benchmarking methodology for Apple Silicon (metric set, baseline selection, workload specification), targeted at answering the research question of *how individual systems optimizations contribute, independently, to end-to-end throughput*.
2. **First empirical data point** — On a single machine, I measured the SiliconLLM Stage 1 reference implementation against the production-grade engine mlx-lm on Qwen2-0.5B-Instruct.
3. **Core finding** — **The performance gap between the decode and prefill stages is markedly asymmetric**: the unoptimized implementation is **34× slower** than mlx-lm on decode throughput (7.6 vs 260 tok/s), but only **2.4× slower** on prefill throughput (1,467 vs 3,562 tok/s). This distribution directly implies that subsequent engineering effort should be concentrated on the decode pipeline (KV cache + fused attention), not on the matmul primitive itself, providing an empirical basis for the Phase 1/2 work plan.

---

## 1. Background and Research Questions

Systems-level optimization of LLM inference has been one of the most active areas in the systems community over the past two years. From vLLM's PagedAttention to FlashAttention, continuous batching, speculative decoding, and MoE routing, every proposed optimization claims a significant performance improvement. However, the existing literature suffers from two general limitations:

1. Evaluations of individual optimizations are typically performed by *stacking* them on top of already heavily-optimized baselines (e.g., production-grade vLLM). This **lacks a decomposition that quantifies each optimization's contribution in isolation**.
2. Apple Silicon has rapidly proliferated in consumer-grade inference scenarios (workstations, on-device agents), yet systems-level optimization research targeted at Apple Silicon is far sparser than for the NVIDIA GPU ecosystem.

The research questions this project aims to answer are:

> **Q1**: On Apple Silicon, how is the performance gap between an LLM inference implementation that uses only matrix-operation APIs (no systems optimizations) and production-grade engines (mlx-lm / llama.cpp / vLLM) distributed across the stages of the inference pipeline?
>
> **Q2**: When KV cache, fused quantized matmul, Flash Attention, continuous batching, PagedAttention, and other optimizations are introduced incrementally, what is the marginal contribution of each to end-to-end throughput?
>
> **Q3**: Do these optimizations yield on Apple Silicon the same speedup curves reported in their original papers on NVIDIA GPUs? If they differ, what are the physical root causes of the difference?

It must be emphasized: **this project does not claim "reinventing LLM inference algorithms" as a research contribution**. Every underlying technique mentioned in Q1–Q3 (KV cache, Flash Attention 2, PagedAttention, etc.) was proposed and open-sourced in top-tier systems conference papers between 2022 and 2024. This project's contribution lies **at the measurement and decomposition layer**: existing papers predominantly follow the pattern of "stacking a new optimization on top of an already-optimized vLLM/TGI," and there is a gap in quantifying each optimization stage independently between a "zero-optimization baseline" and a "production baseline." This project fills that gap.

This month's work focuses on **Q1** — converting the question *"how large is the gap, and where does it lie?"* from a qualitative description into concrete numbers via Phase 0 benchmarking.

---

## 2. Methodology

### 2.1 Experimental platform: tiny-llm as an external substrate

The experimental code base of this study builds on the third-party open-source project `skyzh/tiny-llm` (authors: Alex Chi and Connor; GitHub 4.1k stars; license Apache-2.0 / CC-BY-NC-SA-4.0). **The project explicitly positions itself as a course on LLM serving** — the first sentence of its README reads *"A course on LLM serving,"* and the author explicitly likens it to CMU's Deep Learning Systems course's `needle` project. Every component in tiny-llm is an MLX re-implementation of a published paper or an existing open-source implementation, and the project does **not** contain novel algorithmic or systems-level contributions:


| Component in tiny-llm                           | Original source               |
| ----------------------------------------------- | ----------------------------- |
| Scaled Dot-Product Attention, Transformer Block | Vaswani et al., NeurIPS 2017  |
| Rotary Positional Encoding (RoPE)               | Su et al., 2021               |
| Grouped Query Attention (GQA)                   | Ainslie et al., 2023          |
| Flash Attention 2                               | Dao, 2023                     |
| Continuous Batching                             | Yu et al., OSDI 2022 (Orca)   |
| Paged Attention                                 | Kwon et al., SOSP 2023 (vLLM) |
| Speculative Decoding                            | Leviathan et al., ICML 2023   |
| Quantized matmul (int4)                         | llama.cpp / GPTQ family       |


The rationale for adopting it as the experimental platform — rather than building one from scratch — rests on the following engineering considerations:

1. **The "optimization staircase" structure is a natural fit for Q2's decomposition measurement.** Stage 1 of tiny-llm provides a zero-systems-optimization baseline; Stage 2 introduces industrial-grade optimizations (KV cache, Flash Attention, continuous batching); Stage 3 introduces frontier optimizations (Paged Attention, Speculative Decoding). A researcher can therefore **toggle optimizations on one by one within the same codebase and measure each one's incremental contribution**, avoiding the large amount of noise that cross-codebase comparisons would introduce.
2. **Reducing engineering overhead that is not research contribution.** Implementing a Qwen2 MLX backend from scratch is a multi-week engineering effort, but that work does *not* fall within the research contribution of this project. Adopting an existing platform lets research time be concentrated on the activities that are research-valuable: measurement, analysis, and the design of original optimizations.
3. **Reproducibility.** Using a public platform means that any findings produced by this research can be independently verified by a third party.

### 2.2 Division of labor: this project's contribution vs. what comes from tiny-llm

To prevent any confusion about the attribution of "research results of this project," the table below makes the division explicit:


| Inherited from tiny-llm (NOT counted as contribution of this project)                | Independent contribution of this project (this month and going forward)                                                           |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| MLX implementation code for attention, RoPE, GQA, MLP, and the Qwen2 model structure | Design of the Phase 0 benchmarking methodology (metric set, baseline selection, workload standardization)                         |
| Stage 1–3 task decomposition and instructional pathway                                | Academic justification of the methodology (why vLLM-on-CPU is uninformative; why mlx-lm is the correct "same-framework ceiling")  |
| The reference implementation `tiny_llm_ref/`, usable as a correctness oracle         | Empirical measurement data and its interpretation (this month's output: see Section 3, Table 1)                                   |
| The basic scaffold of `bench.py`                                                     | **Key finding: the asymmetry of the decode–prefill performance gap** (see Section 3.3)                                            |
| The build scaffolding for Stage 2 Metal/C++ extensions                                | Falsifiable success criteria for Phase 2/3 (see Section 4.2)                                                                      |
| The unit-testing framework                                                           | Original optimization topic for Phase 3 (TBD — to be selected from one of the unfinished Stage 3 chapters and designed originally) |


In other words: **the role that tiny-llm plays in this research is equivalent to the role PyTorch / MLX plays in the vast majority of deep-learning papers — it is experimental infrastructure, not the research contribution itself.**

### 2.3 Phase 0 benchmarking methodology

**Metric set** (aligned with industry-standard metrics):


| Metric                     | Definition                                                | What it captures                                              |
| -------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
| Prefill throughput (tok/s) | `prompt_tokens / prefill_time`                            | Long-prompt, short-reply scenarios (e.g., RAG, summarization) |
| Decode throughput (tok/s)  | `generated_tokens / decode_time`                          | Long-generation scenarios (chat, code generation)             |
| TTFT (Time To First Token) | Latency from request arrival to the first generated token | Interactive user experience                                   |
| Peak memory (GB)           | Maximum RSS                                               | Feasibility on consumer-grade hardware                        |


**Baseline selection and justification**:

- **mlx-lm** (production-grade MLX engine) — the "same-framework, same-hardware performance ceiling." Since it shares a runtime (MLX) with SiliconLLM, any gap between them is attributable purely to the quality of the systems-optimization layer, making it **the most tightly-controlled comparability baseline**.
- **llama.cpp** (GGML + Metal) — the "cross-framework, cross-ecosystem industry baseline." It represents the inference engine Apple Silicon users actually run in production; any research result that purports to approach it carries practical meaning.
- **vLLM** (CUDA-native) — because its core optimizations (the CUDA implementation of PagedAttention, Triton fused kernels) are incompatible with the Apple Silicon architecture, on a Mac it can only run in CPU-only mode, which fundamentally contradicts its design intent. This month I decided to **downgrade vLLM to an optional supplementary baseline** (see Section 6 for discussion); if necessary, it can be added later via cloud-GPU rental.

**Workload specification** (inherits upstream tiny-llm `bench.py` defaults, so that this research's own measurements across stages and the external-engine measurements share an identical input distribution):

- Synthetic requests, random seed = 0
- Prompt length: uniform in [64, 256] tokens
- Output length: uniform in [64, 256] tokens
- Greedy decoding (temperature = 0)
- 1 warm-up run, followed by formal measurement
- 4 requests per measurement this round / 16 requests for a full run (scaling up with disk capacity and model size)

The complete, reproducible description of the methodology is archived in `docs/PHASE0_BENCHMARK.md`.

---

## 3. Experimental Results

### 3.1 Hardware and model


| Item      | Value                                            |
| --------- | ------------------------------------------------ |
| Hardware  | MacBook Pro, Apple Silicon, macOS 14.6.1         |
| Model     | Qwen2-0.5B-Instruct, int4 quantized (MLX format) |
| Framework | MLX 0.29.1 + Python 3.12.13                      |


> Note: the full 7B-scale head-to-head measurement is postponed to early May due to disk capacity constraints this month. The conclusions drawn at the 0.5B scale remain representative at the order-of-magnitude level; the asymmetric-gap pattern at the major stages is expected to persist or amplify at larger model sizes.

### 3.2 Head-to-head measurement results

**Table 1**: SiliconLLM Stage 1 reference implementation vs. mlx-lm; same hardware, same model, same 4-request synthetic workload.


| Metric                   | SiliconLLM Stage 1 ref | mlx-lm        | Ratio     |
| ------------------------ | --------------------- | ------------- | --------- |
| Prefill throughput       | 1,467 tok/s           | 3,562 tok/s   | **2.4×**  |
| Decode throughput        | **7.6 tok/s**         | **260 tok/s** | **34.2×** |
| Wall time for 4 requests | 89.4 s                | 2.83 s        | 31.6×     |
| Peak memory              | not measured          | 0.53 GB       | —         |


> Full per-request raw numbers and reproduction commands are archived in `docs/PHASE0_RESULTS.md`.

### 3.3 Principal observation

The most salient feature of these results is **the asymmetry of the gap**. If one looks only at the end-to-end wall-time ratio (31.6×), it is easy to conclude "uniformly far behind." But decomposing into the two sub-stages reveals a very different picture:

- **Prefill gap is only 2.4×**. Prefill is a single large forward pass dominated by matrix multiplication, and MLX's native matmul kernel already provides substantial acceleration even to the unoptimized implementation. The remaining 2.4× comes primarily from mlx-lm's fused int4-dequantize-and-matmul (a single kernel performs dequantization and multiplication together), which saves a full-size-activation memory round trip.
- **Decode gap is 34×**. This is not a kernel-efficiency issue but an **algorithmic-complexity difference**. For each token generated, the unoptimized implementation recomputes attention over the entire `[prefill + already-generated context]` (O(L²) compute per token). Introducing a KV cache reduces this to O(L) per token. For the context lengths used here (L ≈ 100–400), this is consistent with the observed 1–2 orders of magnitude of speedup — the 34× ratio matches this analysis at the right order of magnitude.

---

## 4. Implications

### 4.1 Locating the optimization budget

The results of Table 1 directly inform how to allocate subsequent engineering effort:

- Optimizations targeted at **prefill** (2.4× gap) are bounded from above by fused-quantization kernels; the marginal room for improvement is limited.
- Optimizations targeted at **decode** (34× gap) have a vast headroom concentrated in a small set of well-known mechanisms (KV cache, Flash Attention, continuous batching).

It is worth noting that this distribution agrees in spirit with the tiny-llm author's choice to concentrate Stage 2 on KV cache + Flash Attention + continuous batching — but that choice was previously grounded more in the author's engineering intuition than in open quantitative evidence. The measurements of this project **independently validate** that prioritization on Apple Silicon, and provide citable numbers for future researchers (or for engineers reading that course in the future). In addition, having these numbers in hand means that the work priorities during this project's Phase 1 self-implementation are no longer dependent on the third-party course's assumed ordering, but have an independent data-driven basis.

### 4.2 Falsifiable targets for subsequent stages

Based on the measurements above, I am setting falsifiable success criteria for subsequent stages so that progress becomes quantitatively evaluable:

- **Phase 2 (after Stage 2 self-implementation)**: SiliconLLM's decode throughput should reach **at least 1/3** of the same-machine mlx-lm measurement (≈ 87 tok/s on this hardware).
  - If it falls below 1/5 (≤ 52 tok/s): the Stage 2 KV cache or Flash Attention implementation quality is insufficient, and a deeper bottleneck analysis is needed.
  - If it exceeds 1/1.5 (≥ 173 tok/s): suspiciously high; I would need to verify that I am not accidentally invoking mlx-lm's internal fused operators.
- **Phase 3 (one advanced Stage 3 topic)**: Expected selection is one of {Paged Attention, Speculative Decoding}. The output will be the **incremental speedup curve relative to the Stage 2 baseline**, compared against the speedups that the original paper reports on NVIDIA GPUs (answering Q3).

---

## 5. Plan for the Next Stage (wrap up Phase 0 → launch Phase 1)

**Wrapping up Phase 0 (first half of month 2)**:

1. Complete the llama.cpp Metal baseline measurement on Qwen2-0.5B (the cross-framework comparison point).
2. Formalize the benchmark scripts as `benches/phase0_compare.py` with a unified `--engine {mlx_lm, llama_cpp, silicon_llm}` entry point, producing CSV output for subsequent plotting and documentation.
3. Produce a short-paper-style summary of Phase 0 (2–3 pages), to serve as Chapter 1 of the formal research report.

**Launching Phase 1 (second half of month 2)**:

1. Using tiny-llm's reference implementation (`tiny_llm_ref/`) as a correctness oracle, re-implement from scratch the "zero-optimization" Qwen2 inference stack on Apple Silicon (attention, RoPE, GQA, RMSNorm, MLP, sampling). **The research value of this stage lies not in the code itself (which will be functionally equivalent to the reference) but in step 2 below.**
2. After self-implementation is complete, re-run the Phase 0 benchmark and verify that the self-implemented version lies within ±5% of the reference implementation on tok/s. If the deviation exceeds 5%, the deviation itself is a research question worth investigating in its own right (e.g., *"Does MLX graph compilation fuse functionally-equivalent but syntactically-different implementations differently?"*) and may form a standalone short note.

**Identified blockers**:

- Measurement of the Stage 2 reference implementation's GPU path requires the full Xcode.app (~15 GB); this is scheduled to be installed before Phase 2 launches, and does not block this month's or next month's work.
- A complete 7B-scale three-engine comparison requires ≥ 15 GB of free disk space. The current plan is to complete the full 0.5B comparison first, with the 7B run deferred.

---

## 6. Discussion

### 6.1 The research value of Phase 0 (beyond merely "establishing a yardstick")

Koike Sensei's suggestion at the end of last month — "measure before you build" — proved to deliver threefold value in execution this month:

1. **Directional guidance**: Had I skipped Phase 0 and gone straight to implementing Stage 1, I would not have known that *prefill has only 2.4× of headroom left, while decode is the real battleground*. Phase 0 converts "where should I spend my time?" from a guess into a data-driven decision.
2. **Falsifiable progress**: With mlx-lm's 260 tok/s serving as a hard ceiling, any implementation in Phase 1/2 is no longer judged by "does it run?" but against a concrete quantitative target.
3. **Avoiding pseudo-optimization traps**: Without the mlx-lm baseline, when a self-implemented Stage 2 version reaches, say, 100 tok/s, it would be easy to fall into the illusion of "fast enough." With a 260 tok/s ceiling visible, it is immediately clear that a 2.6× gap remains.

### 6.2 On whether to include vLLM

This month's methodology discussion downgrades vLLM to an "optional supplementary baseline." The reasoning:

- vLLM's core technical contributions (the CUDA implementation of PagedAttention, Triton fused kernels, CUDA graph capture) all depend on the NVIDIA GPU ecosystem.
- On a Mac, vLLM must fall back to its CPU backend. In this degraded form, vLLM (a) does not represent its real production performance (which is tens of times faster on a GPU), and (b) cannot be used to evaluate the effectiveness of its technical design. Numbers produced in this mode carry **no information value** for any research question we might pose.
- If a vLLM comparison is ultimately needed, I recommend an hourly cloud-GPU rental (a single full measurement set can be completed for roughly $5) to obtain numbers that reflect its design value. I treat this as a parked "Phase 0.5 optional extension."

**Recommendation for the Koike Sensei**: Use mlx-lm + llama.cpp as the main baseline pair of Phase 0; leave vLLM as an optional cloud-side comparison.

### 6.3 Validity of extrapolating from 0.5B to 7B

This month's measurements are concentrated at the 0.5B scale, primarily for reasons of disk capacity and iteration speed. I expect the qualitative conclusion to hold at 7B (the decode gap will remain substantially larger than the prefill gap), but the *absolute* gap ratios may move:

- **Hypothesis A**: Both gaps shrink at 7B. As the model grows, per-step compute time is dominated by kernel efficiency rather than overhead, so Python overhead as a fraction of the total drops.
- **Hypothesis B**: The decode gap widens further at 7B. Because the no-KV-cache O(L²) cost grows super-linearly with context length, and the L ≈ 200 regime is already pronounced at 0.5B, it will be more so at 7B.

The 7B measurement next month will directly test these two hypotheses.

---

## 7. References

**External experimental platform used by this study**:

- Alex Chi (skyzh) and Connor. *Tiny LLM — LLM Serving in a Week*. Open-source educational project. GitHub: [https://github.com/skyzh/tiny-llm](https://github.com/skyzh/tiny-llm); course site: [https://skyzh.github.io/tiny-llm/](https://skyzh.github.io/tiny-llm/) (accessed 2026-04). The experimental code, Qwen2 MLX model loading, testing framework, and the base `bench.py` structure used in this research are all sourced from this project. Licenses: Apache-2.0 (code) / CC BY-NC-SA 4.0 (course materials).

**Open-source software and models used**:

- Apple. *MLX: An array framework for Apple Silicon*. [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx).
- Apple. *mlx-lm: LM serving on MLX*. [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm).
- G. Gerganov et al. *llama.cpp: Inference of LLaMA in pure C/C++*. [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).
- Alibaba (Qwen Team). *Qwen2-0.5B/7B-Instruct-MLX*. Hugging Face.

**Academic works cited** (original sources for the techniques mentioned in the text):

- Vaswani et al. *Attention Is All You Need*. NeurIPS 2017.
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021.
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. 2023.
- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022 (continuous batching).
- T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv:2307.08691.
- W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023 (vLLM).
- Leviathan et al. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
- A. Yang et al. *Qwen2 Technical Report*. arXiv:2407.10671.

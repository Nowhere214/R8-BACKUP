# Phase 0: Measured Results Log

This file is the authoritative Phase 0 result log. All measurements were
run on the same Mac. Workload definitions are kept fixed within each
experiment and documented below, because not every engine exposes the
same benchmark metric.

## Environment

- Hardware: MacBook Pro (Apple Silicon, arm64, macOS 14.x)
- Python: 3.12.13 (pdm-managed)
- Key packages: `mlx 0.29.1`, `mlx-lm 0.27.1+`, `torch 2.7.1`
- Note: Xcode.app **not** installed yet, so Stage 2 of the SiliconLLM
  reference implementation cannot run (needs `metal` shader compiler).
  llama.cpp was run from its pre-built macOS-arm64 release binary for the
  same reason.

## Main workload

Mirrors the upstream `bench.py` defaults (so future SiliconLLM runs are
directly comparable):

- 4 synthetic requests (scaled down from the default 16 for fast iteration)
- Input length uniform in [64, 256]
- Output length uniform in [64, 256]
- `seed = 0`
- 1 warmup run
- Device: GPU (Metal)
- Temperature: 0 (greedy)

Additional workload-length and microbenchmark settings are documented in
their own sections below.

## Data version note (updated 2026-05-29)

The current Phase 0 log contains three groups of measurements:

1. **Model-size engine comparison**: SiliconLLM Stage 1 ref and mlx-lm,
   N=3, measured on 2026-05-14 via `benches/biweekly_n3.py`.
2. **llama.cpp Metal baseline**: Qwen2 Q4_K_M GGUF models, N=3, measured
   on 2026-05-27 via the pre-built llama.cpp b9354 release.
3. **Mechanism checks**: prompt-length decode scan and KV-cache
   microbenchmark, measured on 2026-05-28.

The 2026-05-14 N=3 measurement supersedes the N=1 numbers reported in
the April monthly report; see `docs/BIWEEKLY_REPORT_2026-05-14_ZH.md`
Section 3 for the divergence analysis. The N=1 numbers are kept here for
historical comparison only.

## Results table

### Model-size engine comparison (current authoritative data)

SiliconLLM Stage 1 ref and mlx-lm use the shared synthetic workload from
`bench.py`: 4 requests, input length sampled from [64, 256], output length
sampled from [64, 256], seed 0, 1 warmup, N=3.

llama.cpp uses `llama-bench -p 128 -n 128 -r 3` with Q4_K_M GGUF weights.
This is not identical to the random-request workload, but it reports the
same two throughput concepts: prompt processing and token generation.
Because the runtime and quantization format differ, llama.cpp is treated
as a production-engine baseline rather than a same-framework replacement
for mlx-lm.

| Run                                              | N | Prefill tok/s (mean ± stdev, CV%) | Decode tok/s (mean ± stdev, CV%) | Wall (s, mean) | Peak mem | Notes |
| ------------------------------------------------ |--:| ---------------------------------:| --------------------------------:| --------------:| --------:| ----- |
| **SiliconLLM Stage 1 ref (Qwen2-0.5B)**           | 3 | **2,539 ± 47  (1.8%)**            | **9.98 ± 0.40 (4.0%)**           | 66.83          |  1.89 GB | Highest CV in the set, but still < 5% |
| **SiliconLLM Stage 1 ref (Qwen2-1.5B)**           | 3 | **998 ± 0.7   (0.1%)**            | **3.86 ± 0.004 (0.1%)**          | 172.65         |  4.71 GB | Extremely tight runs |
| **mlx-lm (Qwen2-0.5B)**                          | 3 | **4,015 ± 8   (0.2%)**            | **291.6 ± 0.08 (0.0%)**          |   2.50         |  0.53 GB |  |
| **mlx-lm (Qwen2-1.5B)**                          | 3 | **1,276 ± 12  (0.9%)**            | **126.4 ± 0.18 (0.1%)**          |   5.93         |  1.35 GB |  |
| **mlx-lm (Qwen2-7B)**                            | 3 | **287 ± 0.07  (0.0%)**            | **32.4 ± 0.007 (0.0%)**          |  23.39         |  4.77 GB | Per-rep wall times agreed to within 10 ms |
| **llama.cpp Metal (Qwen2-0.5B Q4_K_M)**          | 3 | **2,924.7 ± 36.2 (1.2%)**         | **137.3 ± 1.0 (0.7%)**           | —              | —        | llama.cpp b9354, Metal + BLAS |
| **llama.cpp Metal (Qwen2-1.5B Q4_K_M)**          | 3 | **1,103.7 ± 8.3 (0.8%)**          | **71.9 ± 0.5 (0.7%)**            | —              | —        | llama.cpp b9354, Metal + BLAS |
| **llama.cpp Metal (Qwen2-7B Q4_K_M)**            | 3 | **266.2 ± 2.0 (0.8%)**            | **22.2 ± 0.03 (0.1%)**           | —              | —        | llama.cpp b9354, Metal + BLAS |
| **SiliconLLM Stage 2 ref (any size)**             | — | —                                 | —                                | —              | —        | BLOCKED until Xcode.app installed (C++/Metal extension won't compile) |
| **SiliconLLM Stage 1 ref (Qwen2-7B)**             | — | —                                 | —                                | —              | —        | INFEASIBLE: Stage 1 ref dequantizes int4 → bf16 at load (~17 GB RSS) > 18 GB unified memory on test machine. This itself is a Phase-0 finding. |
| **vLLM CPU (Qwen2-*, fp16)**                     | — | —                                 | —                                | —              | —        | Not included in Phase 0 unless advisor asks; Mac CPU path is not a meaningful production baseline |

Raw per-rep records: `benches/biweekly_n3_results.csv`
Summary: `benches/biweekly_n3_summary.csv`
llama.cpp summary: `benches/llama_cpp/llama_cpp_b9354_summary.csv`

### Historical N=1 results (2026-04-23, **superseded** — kept for divergence analysis)

| Run                                              | Prefill tok/s | Decode tok/s | Wall (s) | Peak mem |
| ------------------------------------------------ | ------------: | -----------: | -------: | -------: |
| SiliconLLM Stage 1 ref (Qwen2-0.5B), N=1          |       1,467.4 |          7.6 |    89.44 |      n/a |
| SiliconLLM Stage 1 ref (Qwen2-1.5B), N=1          |         262.0 |         3.83 |   175.72 |      n/a |
| mlx-lm (Qwen2-0.5B), N=1                         |        ~3,562 |         ~260 |     2.83 |  0.53 GB |
| mlx-lm (Qwen2-1.5B), N=1                         |        ~1,297 |         ~124 |     6.01 |  1.35 GB |
| mlx-lm (Qwen2-7B), N=1                           |          ~282 |        ~31.9 |    23.80 |  4.77 GB |

**Divergence summary** (N=3 vs N=1):

- **mlx-lm**: ±2–13% on all five measurements. N=1 was acceptable.
- **Stage 1 ref @ 0.5B**: +73% prefill, +31% decode under N=3. Substantial.
- **Stage 1 ref @ 1.5B**: **+281% prefill** under N=3, decode unchanged.
  This is a >3× discrepancy — clear evidence that N=1 measurement of the
  Stage 1 reference is unreliable for prefill, likely due to MLX kernel JIT
  / shader-cache warmup being far more impactful for the high-overhead
  unoptimized path than for the well-tuned mlx-lm path.

**Methodological consequence**: from now on, all Phase-0 numbers are
reported as N≥3 mean ± stdev. Single-shot measurements are documentation
of "ran successfully," not "performance estimate."

_(N=1 per-request breakdowns from 2026-04-23 are kept in
`docs/MONTHLY_REPORT_MONTH1_ZH.md` Section 3 for record-keeping; not
duplicated here to avoid stale data being mistaken for current results.)_

## Key comparison — gap evolution across model sizes (N=3 data)

### Direct measurements

| Metric             | 0.5B Stage 1 ref | 0.5B mlx-lm | **0.5B gap** | 1.5B Stage 1 ref | 1.5B mlx-lm | **1.5B gap** | 7B Stage 1 ref | 7B mlx-lm | **7B gap** |
| ------------------ | --------------: | ----------: | -----------: | --------------: | ----------: | -----------: | ------------: | --------: | ---------: |
| Prefill tok/s      |           2,539 |       4,015 |     **1.58×**|             998 |       1,276 |     **1.28×**|    OOM (n/m)  |       287 |    not measurable |
| Decode tok/s       |            9.98 |       291.6 |    **29.2×** |            3.86 |       126.4 |    **32.7×** |    OOM (n/m)  |      32.4 |    not measurable |
| Peak mem           |         1.89 GB |     0.53 GB |          —   |         4.71 GB |     1.35 GB |          —   |     n/a       |   4.77 GB |              —  |

### Three-engine comparison after adding llama.cpp

llama.cpp Metal sits between the Stage 1 reference path and mlx-lm. It is
much faster than the Stage 1 reference implementation, but mlx-lm remains
faster on decode in this measurement.

| Model | Stage 1 ref decode | llama.cpp decode | mlx-lm decode | llama.cpp / Stage 1 ref | mlx-lm / llama.cpp |
| ----- | -----------------:| ----------------:| -------------:| ---------------------:| ------------------:|
| Qwen2 0.5B | 9.98 | 137.3 | 291.6 | 13.8× | 2.1× |
| Qwen2 1.5B | 3.86 | 71.9 | 126.4 | 18.6× | 1.8× |
| Qwen2 7B | n/m | 22.2 | 32.4 | n/m | 1.5× |

Prefill is closer across production engines:

| Model | Stage 1 ref prefill | llama.cpp prefill | mlx-lm prefill | mlx-lm / llama.cpp |
| ----- | ------------------:| -----------------:| --------------:| ------------------:|
| Qwen2 0.5B | 2,539 | 2,924.7 | 4,015 | 1.37× |
| Qwen2 1.5B | 998 | 1,103.7 | 1,276 | 1.16× |
| Qwen2 7B | n/m | 266.2 | 287 | 1.08× |

This reinforces the main Phase 0 result: the largest implementation gap
is still decode. Adding a second production engine did not overturn that
conclusion; it made the baseline story stronger.

### mlx-lm scaling alone (production-engine ceiling curve)

Since Stage 1 ref cannot be measured at 7B, the cleanest available
extrapolation surface is the mlx-lm scaling curve itself:

| mlx-lm metric        | 0.5B  | 1.5B  | 7B     | 0.5B → 7B ratio | Model param ratio | Verdict |
| -------------------- | ----: | ----: | -----: | --------------: | ----------------: | :------ |
| Prefill tok/s        | 4,015 | 1,276 |    287 |          14.0× |               14× | linear scaling (saturated) |
| Decode tok/s         | 291.6 | 126.4 |   32.4 |           9.0× |               14× | **sub-linear** — bigger matrices saturate Apple Silicon GPU better |
| Peak memory (GB)     |  0.53 |  1.35 |   4.77 |           9.0× |               14× | sub-linear — fixed runtime overhead amortized at scale |

### Findings (revised after N=3 measurement, 2026-05-14)

**Finding 1 (still holds): The decode gap is roughly invariant in model size**

Both engines saw decode throughput drop by ~2.3–2.6× going 0.5B → 1.5B
(Stage 1 ref: 9.98 → 3.86, ratio 2.6×; mlx-lm: 291.6 → 126.4, ratio 2.3×),
so the relative gap stays in the **29–33× band**: 29.2× at 0.5B → 32.7× at 1.5B.
The slight drift is within run-to-run variance and is not statistically
significant.

This refutes both hypotheses originally written in Section 6.3 of the Month-1
report:

- Hypothesis A predicted the gap would *shrink* (Python overhead → kernel
  efficiency dominance with bigger matrices). Did not happen.
- Hypothesis B predicted the gap would *widen* (O(L²) attention cost grows
  super-linearly with L). Did not happen at L ≈ 200.

Most likely physical interpretation: at L ≈ 200, the constant
**multiplicative cost** of "no KV cache" (a roughly L/2-fold extra work per
token, independent of model size) dominates the per-token cost asymmetry,
and both engines amortize this cost at the same proportional rate as the
model grows.

**Finding 2 (REVISED — original direction was wrong): The prefill gap NARROWS slightly with model size**

The April monthly report claimed the prefill gap *widened* (2.4× → 5.0×) as
the model grew 0.5B → 1.5B. **N=3 re-measurement showed this was an artifact
of N=1 noise on the Stage 1 ref @ 1.5B prefill measurement** (which was off
by +281% in the April single-shot). The actual pattern is:

- 0.5B prefill gap: 1.58× (was reported as 2.4×)
- 1.5B prefill gap: 1.28× (was reported as 5.0×)
- Trend: **narrowing** as model size grows.

New physical interpretation: at 0.5B, the per-step Python / kernel-launch
overhead in Stage 1 ref dominates the per-token cost; at 1.5B, the matmul
cost dominates and the fixed overhead amortizes. As a result Stage 1 ref
becomes *relatively* more efficient at larger model sizes. mlx-lm has no
such fixed overhead (already at GPU saturation at 0.5B), so its prefill
scales nearly linearly with parameter count — confirmed independently by
the mlx-lm 0.5B → 7B measurement, which gives a 14.0× drop for a 14×
parameter ratio (perfectly linear).

**Methodological lesson worth recording**: among the 10 measurements
re-run with N=3, the four mlx-lm measurements moved by < 13% vs N=1, but
the SiliconLLM Stage 1 ref measurements moved by up to **+281%** on
prefill. The unoptimized path's high per-step overhead makes it
disproportionately sensitive to first-call JIT / shader-cache warmup
state. This finding generalizes: **for any benchmark where the path
under test has high overhead, N=1 is unsafe**; the production-engine
baseline tolerates N=1 better. From now on all Phase-0 numbers are
reported as N≥3 mean ± stdev.

**Finding 3 (new): llama.cpp confirms the decode bottleneck but does not replace mlx-lm as the ceiling**

The llama.cpp Metal baseline makes Phase 0 a three-engine comparison. Its
decode throughput is 13.8× faster than Stage 1 ref at 0.5B and 18.6×
faster at 1.5B, so a production runtime clearly removes a large portion
of the Stage 1 decode bottleneck.

At the same time, mlx-lm remains faster than llama.cpp in this setup:
2.1× faster at 0.5B, 1.8× faster at 1.5B, and 1.5× faster at 7B on
decode. For prefill, the two production engines are much closer,
especially at 7B.

The interpretation is therefore not "llama.cpp replaces mlx-lm as the
ceiling." The cleaner interpretation is: mlx-lm remains the same-framework
upper baseline, while llama.cpp provides an independent production-engine
sanity check. Both production baselines agree that the Stage 1 path's main
deficit is decode.

### Phase-2 success criterion (updated using N=3 mlx-lm data)

The April-month criterion ("Decode within 1/3 of mlx-lm") is preserved.
Concrete numerical targets, now per-model-size, using the N=3-measured
mlx-lm decode throughputs as the same-machine ceiling:

| Model | mlx-lm decode (ceiling, N=3 mean) | Phase-2 target (≥ 1/3) | Concerning if (≤ 1/5) | Suspicious if (≥ 1/1.5) |
| ----- | ---------------------------------: | ---------------------: | --------------------: | ----------------------: |
| 0.5B  |                       291.6 tok/s  |              ≥ 97 tok/s|               ≤ 58    |               ≥ 194     |
| 1.5B  |                       126.4 tok/s  |              ≥ 42 tok/s|               ≤ 25    |               ≥  84     |
| 7B    |                        32.4 tok/s  |             ≥ 11 tok/s |               ≤  6    |               ≥  22     |

The 7B target is interestingly *low in absolute terms* (≥ 11 tok/s),
suggesting that — perhaps counter-intuitively — passing the criterion at
7B may be easier in absolute throughput than at 0.5B, simply because the
ceiling itself is much lower. This is consistent with how all engines
become bandwidth-bound at large scale.

### Memory feasibility (a side finding) — now with 7B data

mlx-lm peak memory measured directly across three sizes:

- 0.5B → 0.53 GB (1.07 GB / B params)
- 1.5B → 1.35 GB (0.88 GB / B params)
- 7.0B → 4.77 GB (0.68 GB / B params)

Per-billion-parameter memory cost decreases at scale, indicating fixed
runtime overhead amortizes as the model grows. Projecting to a 14B Qwen2:
~9 GB peak, still fits in 18 GB unified memory. Projecting to 30B+: would
need >16 GB working set, increasingly tight on consumer hardware.

In contrast, the Stage 1 reference implementation's choice to dequantize
weights at load makes 7B inference **infeasible at all** on this test
machine (requires ~17–20 GB RSS, exceeding the 18 GB unified memory).
This is itself a Phase-0 finding worth reporting: on consumer-grade
unified-memory hardware, "no quantized matmul" is not merely a performance
optimization — it is a **feasibility gate**. The transition between
"performance optimization" and "feasibility gate" depends on the ratio
of (dequantized model size) to (available unified memory), which on an
18 GB Mac happens between the 1.5B scale (5 GB dequantized, fits) and the
7B scale (24 GB dequantized, doesn't fit).

## Workload-length scan (2026-05-28)

The model-size comparison above used short synthetic prompts. To check
whether the decode gap changes with context length, I ran a small
prompt-length scan with fixed generation length.

Setup:

- Engines: SiliconLLM Stage 1 ref and mlx-lm
- Models: Qwen2 0.5B and Qwen2 1.5B
- Prompt lengths: 64, 128, 512 tokens
- Generation length: 64 tokens
- Number of requests: 2
- Repetitions: N=2
- mlx-lm EOS stopping disabled so every run generates the same number of tokens

| Model | Prompt tokens | SiliconLLM Stage 1 ref decode | mlx-lm decode | Gap |
| ----- | -------------:| ----------------------------:| -------------:| ---:|
| Qwen2 0.5B | 64  | 20.98 ± 0.012 | 314.39 ± 0.026 | 15.0× |
| Qwen2 0.5B | 128 | 16.12 ± 0.005 | 312.84 ± 0.562 | 19.4× |
| Qwen2 0.5B | 512 | 5.22 ± 0.047  | 301.33 ± 0.519 | 57.7× |
| Qwen2 1.5B | 64  | 7.39 ± 0.137  | 103.19 ± 0.757 | 14.0× |
| Qwen2 1.5B | 128 | 5.30 ± 0.005  | 104.60 ± 0.037 | 19.7× |
| Qwen2 1.5B | 512 | 1.76 ± 0.035  | 102.78 ± 0.262 | 58.5× |

Raw and summary files:

- `benches/workload_length_scan_results.csv`
- `benches/workload_length_scan_summary.csv`
- `benches/workload_length_scan_1p5b_results.csv`
- `benches/workload_length_scan_1p5b_summary.csv`
- Figure: `benches/workload_length_scan_combined.png`

The result adds a second dimension to the Phase 0 conclusion. At fixed
model size, mlx-lm decode throughput stays nearly flat as prompt length
grows, while the Stage 1 reference path slows down sharply. The decode gap
therefore grows from about 14–15× at 64 prompt tokens to about 58× at 512
prompt tokens.

This supports the explanation that the Stage 1 path is paying repeated
full-context attention work during decode.

### llama.cpp workload-length note

I also ran a matching prompt-length scan with llama.cpp Metal using
`llama-bench -pg <prompt,gen>`.

Important metric caveat: `-pg` reports mixed prompt+generation throughput,
not pure decode throughput. It should therefore not be plotted as a
decode-only number next to the mlx-lm and Stage 1 ref rows above.

| Model | Prompt tokens | Gen tokens | Mixed prompt+gen tok/s |
| ----- | -------------:| ----------:| ----------------------:|
| Qwen2 0.5B Q4_K_M | 64  | 64 | 301.2 ± 1.64 |
| Qwen2 0.5B Q4_K_M | 128 | 64 | 437.1 ± 1.71 |
| Qwen2 0.5B Q4_K_M | 512 | 64 | 1010.9 ± 1.66 |
| Qwen2 1.5B Q4_K_M | 64  | 64 | 156.7 ± 0.19 |
| Qwen2 1.5B Q4_K_M | 128 | 64 | 221.7 ± 0.62 |
| Qwen2 1.5B Q4_K_M | 512 | 64 | 468.0 ± 0.62 |

The mixed throughput rises with longer prompts because the workload
contains a larger prefill fraction, and llama.cpp prompt processing is
fast. The useful methodological point is that "throughput" has different
semantics across benchmark tools; decode-only and mixed workload metrics
must be reported separately.

## KV-cache microbenchmark (2026-05-28)

To connect the workload-length result to a mechanism, I ran a synthetic
Qwen2 0.5B-style grouped-attention microbenchmark. It does not load real
model weights. The goal is to isolate the difference between full-context
recompute and cached one-token decode.

Setup:

- Hidden size: 896
- Query heads: 14
- KV heads: 2
- Head dimension: 64
- Batch size: 1
- Context length: 128, 256, 512, 1024, 2048 tokens
- Repetitions: N=20

| Context tokens | Full recompute median ms | Cached decode median ms | Speedup |
| -------------: | -----------------------: | ----------------------: | ------: |
| 128  | 0.876  | 0.249 | 3.5× |
| 256  | 1.169  | 0.251 | 4.6× |
| 512  | 1.834  | 0.239 | 7.7× |
| 1024 | 6.832  | 0.278 | 24.6× |
| 2048 | 23.140 | 0.334 | 69.2× |

Files:

- Script: `benches/kv_cache_microbench.py`
- Summary: `benches/kv_cache_microbench_summary.csv`
- Figure: `benches/kv_cache_microbench.png`
- Vector exports: `benches/kv_cache_microbench.pdf`,
  `benches/kv_cache_microbench.svg`

Full-context recompute becomes much slower as context length grows, while
cached one-token decode remains close to flat. This does not replace a real
Stage 2 benchmark, but it gives a controlled explanation for why the
Stage 1 decode gap grows with prompt length.

The real Stage 2 reference path remains blocked locally because the full
Xcode toolchain is not installed:

```text
xcrun: error: unable to find utility "metal", not a developer tool or in PATH
```

So the next real-model comparison should wait until Phase 2, or use a
non-extension fallback if we decide it is worth building one.

## Decode gap scaling analysis (2026-05-29)

I also converted the prompt-length scan into a simple decode cost model.
This analysis does not run new model inference; it reuses the existing
prompt-length CSV files.

Method:

```text
decode_ms_per_token = 1000 / decode_tok_s
effective_context_len = prompt_len + (gen_len - 1) / 2
decode_ms_per_token = intercept + slope * effective_context_len
```

The fitted slope estimates how much extra decode latency is added by one
additional context token.

| Model | Engine | Slope ms per context token | R2 | Projected decode tok/s at prompt 1024 |
| ----- | ------ | -------------------------: | --:| ------------------------------------: |
| Qwen2 0.5B | mlx-lm | 0.0003 | 0.999 | 287.55 |
| Qwen2 0.5B | SiliconLLM Stage 1 ref | 0.3268 | 0.998 | 2.79 |
| Qwen2 1.5B | mlx-lm | 0.0002 | 0.328 | 101.78 |
| Qwen2 1.5B | SiliconLLM Stage 1 ref | 0.9773 | 1.000 | 0.94 |

Files:

- Script: `benches/analyze_decode_gap_scaling.py`
- Per-point analysis: `benches/decode_gap_scaling_analysis.csv`
- Fit summary: `benches/decode_gap_scaling_fit.csv`
- Figure: `benches/decode_gap_scaling_analysis.png`
- Record: `docs/DECODE_GAP_SCALING_ANALYSIS_2026-05-29.md`

Interpretation:

- mlx-lm has near-zero fitted context-length cost in this scan.
- Stage 1 ref has a measurable per-context-token cost.
- The 1.5B Stage 1 slope is about 3.0× the 0.5B slope, consistent with
  larger-model work per additional context token.

The prompt-1024 values are projections, not measured benchmark results.
They are useful for planning and explanation, but should not be reported as
actual measurements.

## Phase 0 reproducibility manifest (2026-05-29)

For the 6/30 group-meeting target, I added a reproducibility pass over the
existing Phase 0 evidence package. This did not run model inference. It only
reads existing CSV files and normalizes the results into one long-format
summary table.

Files:

- Build script: `benches/build_phase0_unified_summary.py`
- Unified summary: `benches/phase0_unified_summary.csv`
- Coverage figure: `benches/phase0_reproducibility_matrix.png`
- Vector exports: `benches/phase0_reproducibility_matrix.pdf`,
  `benches/phase0_reproducibility_matrix.svg`
- TIFF export: `benches/phase0_reproducibility_matrix.tiff`
- Manifest: `docs/PHASE0_REPRODUCIBILITY_MANIFEST_2026-05-29.md`
- Group-meeting draft: `docs/GROUP_MEETING_REPORT_2026-06-30_EN.md`

The unified CSV currently contains 151 rows:

| Experiment | Rows |
| ---------- | ---: |
| Model-size engine comparison | 20 |
| llama.cpp Metal baseline | 12 |
| Prompt-length decode scan | 54 |
| llama.cpp mixed workload scan | 24 |
| KV-cache microbenchmark | 25 |
| Decode cost-model fit | 16 |

The main value is auditability rather than a new performance claim. Each row
keeps the original source file, engine, model, workload, metric name, unit,
and caveats. In particular, the table records that llama.cpp `-pg` is mixed
prompt+generation throughput, prompt-1024 decode numbers are projections, and
the KV-cache microbenchmark is synthetic mechanism evidence rather than
end-to-end model throughput.

## Stage 1 decode component profile (2026-05-29)

As a Phase 0.5 diagnostic step, I added synchronized component timing around
one Stage 1 reference decode step. This does not replace the throughput
benchmark, because synchronizing after each component changes MLX scheduling.
It is useful for coarse attribution.

Files:

- Script: `benches/stage1_decode_component_profile.py`
- Raw CSV: `benches/stage1_decode_component_profile_results.csv`
- Summary CSV: `benches/stage1_decode_component_profile_summary.csv`
- Figure: `benches/stage1_decode_component_profile.png`
- Vector exports: `benches/stage1_decode_component_profile.pdf`,
  `benches/stage1_decode_component_profile.svg`
- TIFF export: `benches/stage1_decode_component_profile.tiff`
- Record: `docs/STAGE1_DECODE_COMPONENT_PROFILE_2026-05-29.md`
- 7/1 progress draft: `docs/FORMAL_PROGRESS_UPDATE_2026-07-01_EN.md`

The profile uses Qwen2 0.5B, one full-context Stage 1 decode step, and N=3
repetitions per prompt length.

| Prompt tokens | Total measured step | QKV projection | Attention core | Attention out proj | MLP | Elementwise/norm | LM head | Python loop/sampler |
| -------------:| ------------------: | -------------: | -------------: | -----------------: | --: | ---------------: | ------: | ------------------: |
| 64 | 66.07 ms | 10.77 ms | 6.80 ms | 5.07 ms | 29.38 ms | 6.16 ms | 7.69 ms | 0.20 ms |
| 128 | 76.50 ms | 7.17 ms | 7.79 ms | 5.58 ms | 37.95 ms | 6.36 ms | 11.46 ms | 0.20 ms |
| 512 | 190.90 ms | 11.38 ms | 31.83 ms | 8.89 ms | 96.64 ms | 7.98 ms | 33.96 ms | 0.22 ms |

Interpretation:

- Python loop/sampler overhead is not the main issue in this profile.
- At prompt length 512, the largest components are MLP (50.6%), LM head
  (17.8%), and attention core (16.7%).
- This refines the "no KV cache" explanation: because Stage 1 decode re-runs
  the whole sequence, the extra cost is paid by MLP and LM head as well as
  attention.

## Commands used (for reproducibility)

### Model-size comparison: SiliconLLM Stage 1 ref and mlx-lm

```bash
pdm run python benches/biweekly_n3.py
```

### llama.cpp (pre-built release)

```bash
tools/llama.cpp/llama-b9354/llama-bench \
  -m models/gguf/qwen2-0_5b-instruct-q4_k_m.gguf,models/gguf/qwen2-1_5b-instruct-q4_k_m.gguf,models/gguf/qwen2-7b-instruct-q4_k_m.gguf \
  -p 128 -n 128 -t 4 -ngl 99 -r 3 -o csv
```

### Workload-length scan

```bash
pdm run python benches/workload_length_scan.py \
  --models qwen2-0.5b \
  --out-prefix benches/workload_length_scan
pdm run python benches/workload_length_scan.py \
  --models qwen2-1.5b \
  --out-prefix benches/workload_length_scan_1p5b
pdm run python benches/plot_workload_length_scan_combined.py
```

### KV-cache microbenchmark

```bash
pdm run python benches/kv_cache_microbench.py
pdm run python benches/plot_kv_cache_microbench.py
```

### Decode gap scaling analysis

```bash
pdm run python benches/analyze_decode_gap_scaling.py
```

### Unified Phase 0 result manifest

```bash
pdm run python benches/build_phase0_unified_summary.py
```

### Stage 1 decode component profile

```bash
pdm run python benches/stage1_decode_component_profile.py
```

## Current synthesis

The Phase 0 conclusion is now stronger than the original 5/14 version:

1. **The largest gap is decode, not prefill.** This holds across the
   same-framework comparison and the llama.cpp production-engine baseline.
2. **The decode gap is not just a model-size effect.** It grows sharply
   with prompt length in the Stage 1 reference path, while mlx-lm stays
   nearly flat.
3. **KV cache is the most direct next optimization target.** The synthetic
   microbenchmark shows full-context recompute scaling poorly with context
   length, while cached one-token decode stays close to flat.
4. **Quantized matmul is also a feasibility gate.** Stage 1 ref cannot run
   the 7B model on the 18 GB machine because it dequantizes weights at load.
5. **Benchmark metric definitions matter.** llama.cpp `-pg` mixed
   throughput cannot be compared directly to pure decode throughput.
6. **The context-length cost is now quantified.** Stage 1 ref adds about
   0.327 ms per extra context token at 0.5B and 0.977 ms at 1.5B, while
   mlx-lm is close to flat in this scan.
7. **The evidence package is now easier to audit.** The separate Phase 0 CSVs
   are collected into one long-format table with explicit source files,
   units, and caveats.
8. **The decode bottleneck is not mainly Python loop overhead.** The Stage 1
   component profile shows that full-context MLP, LM head, and attention core
   dominate one decode step; the outer Python loop/sampler is below 1% in this
   synchronized timing setup.

## Status of follow-up measurements

### Completed

1. SiliconLLM Stage 1 ref @ 0.5B (2026-04-23 N=1; 2026-05-14 N=3)
2. mlx-lm @ 0.5B (2026-04-23 N=1; 2026-05-14 N=3)
3. SiliconLLM Stage 1 ref @ 1.5B (2026-05-14 N=1 + N=3)
4. mlx-lm @ 1.5B (2026-05-14 N=1 + N=3)
5. mlx-lm @ 7B (2026-05-14 N=1 + N=3 — ran cleanly on 18 GB Mac)
6. N=3 statistical re-measurement of all of the above (2026-05-14)
7. llama.cpp Metal @ {0.5B, 1.5B, 7B}, Q4_K_M GGUF, N=3 (2026-05-27)
8. Prompt-length decode scan @ {0.5B, 1.5B}, N=2 (2026-05-28)
9. llama.cpp mixed prompt+generation workload scan @ {0.5B, 1.5B}
   (2026-05-28)
10. Synthetic KV-cache microbenchmark, N=20 (2026-05-28)
11. Decode gap scaling analysis over the prompt-length scan (2026-05-29)
12. Phase 0 unified reproducibility manifest and coverage figure
    (2026-05-29)
13. Stage 1 decode component profile, Qwen2 0.5B, N=3 per prompt length
    (2026-05-29)

### To do (next two-week period)

14. Phase 1 Stage 1 implementation: start filling `src/tiny_llm/` in book order
   and validate each day with the corresponding tests.
15. Optional deeper MLX/Metal capture: if the coarse component profile is not
   enough, capture lower-level kernel traces for the largest components.
16. Optional longer-context scan: extend prompt buckets to 1024 or 2048
    tokens for mlx-lm first, and only run Stage 1 ref where runtime and
    memory remain reasonable.
17. Decide whether to include vLLM-CPU. Current recommendation: do not
    include it in Phase 0 unless an advisor explicitly asks for a vLLM
    comparison, because the Mac CPU path is not a meaningful production
    baseline.
18. Output-quality spot-check: generate three human-authored prompts greedily
    on each engine, diff the resulting text (should be "similar" not
    identical — different tokenizers / quantizations).
19. Regularize the benchmark scripts into one CLI with
    `--engine {silicon_w1ref,mlx_lm,llama_cpp}` and a shared CSV schema.

# Research Progress Summary Manifest

Date: 2026-05-29

## Goal

This manifest extends the Phase 0 reproducibility package with Phase 1
self-baseline results, Phase 2 fallback benchmark rows, the Stage 2 attribution
benchmark, the Stage 2 decode component profile, and the first Stage 3
speculative decoding accounting results.

The Phase 0 manifest stays unchanged because it should remain a clean record
of measurement-only work. This file creates a broader research-level table for
the 2026-07-01 formal progress update.

No model inference was run for this manifest. It only reads existing CSV files.

## New Artifacts

| Artifact | Path |
|---|---|
| Build script | `benches/build_research_progress_summary.py` |
| Unified research table | `benches/research_progress_unified_summary.csv` |

## Included Evidence

| Phase | Experiment | Rows |
|---|---|---:|
| Phase 0 | Model-size engine comparison | 20 |
| Phase 0 | llama.cpp Metal baseline | 12 |
| Phase 0 | Prompt-length decode scan | 54 |
| Phase 0 | llama.cpp mixed workload scan | 24 |
| Phase 0 | KV-cache microbenchmark | 25 |
| Phase 0 | Decode cost-model fit | 16 |
| Phase 1 | Stage 1 self baseline comparison, 0.5B | 12 |
| Phase 1 | Stage 1 self baseline smoke, 1.5B | 12 |
| Phase 2 | Stage 2 KV-cache fallback benchmark, 0.5B | 12 |
| Phase 2 | Stage 2 KV-cache fallback smoke, 1.5B | 12 |
| Phase 2 | Stage 2 attribution benchmark, 0.5B | 32 |
| Phase 2 | Stage 2 decode component profile, 0.5B | 48 |
| Phase 3 | Stage 3 speculative accounting, 0.5B self-pair | 55 |
| Phase 3 | Stage 3 speculative accounting, 1.5B target / 0.5B draft N=3 | 67 |
| Phase 3 | Stage 3 speculative accounting, 1.5B target / 0.5B draft smoke | 31 |
| Phase 3 | Stage 3 speculative output-length scan | 153 |

Total: 585 data rows, plus one CSV header row.

## Phase 1 Rows

The Phase 1 rows come from:

- `benches/stage1_self_baseline_compare_summary.csv`
- `benches/stage1_self_generation_sanity.csv`
- `benches/stage1_self_baseline_compare_1p5b_smoke_summary.csv`
- `benches/stage1_self_baseline_compare_1p5b_smoke_generation_sanity.csv`

They include:

- self Stage 1 prefill throughput
- self Stage 1 decode throughput
- self Stage 1 wall time
- self Stage 1 peak memory
- reference Stage 1 values for the same metrics
- self/reference throughput and wall-time ratios
- greedy token match as a boolean row

The Phase 2 rows come from:

- `benches/stage2_kv_fallback_benchmark_summary.csv`
- `benches/stage2_kv_fallback_benchmark_1p5b_smoke_summary.csv`
- `benches/stage2_attribution_benchmark_summary.csv`
- `benches/stage2_attribution_benchmark_ratios.csv`
- `benches/stage2_decode_component_profile_summary.csv`

They include Stage 1 full-context and Stage 2 KV-cache fallback metrics, plus
Stage 2 / Stage 1 ratios.

The Phase 3 rows come from:

- `benches/stage3_speculative_accounting_summary.csv`
- `benches/stage3_speculative_accounting_ratios.csv`
- `benches/stage3_speculative_accounting_qwen2_1p5b_draft0p5b_summary.csv`
- `benches/stage3_speculative_accounting_qwen2_1p5b_draft0p5b_ratios.csv`
- `benches/stage3_speculative_accounting_qwen2_1p5b_draft0p5b_smoke_summary.csv`
- `benches/stage3_speculative_accounting_qwen2_1p5b_draft0p5b_smoke_ratios.csv`
- `benches/stage3_speculative_length_scan_summary.csv`
- `benches/stage3_speculative_length_scan_ratios.csv`

They include speculative acceptance rate, target calls per decode token, draft
calls per decode token, decode throughput, wall-time ratios, and exact-match
sanity rows.

## Key Phase 1 Results

Qwen2 0.5B benchmark:

| Metric | Self Stage 1 | Reference Stage 1 | Self/reference |
|---|---:|---:|---:|
| Prefill throughput | 2463.8 ± 23.2 tok/s | 2353.7 ± 100.7 tok/s | 1.047× |
| Decode throughput | 9.94 ± 0.03 tok/s | 9.90 ± 0.10 tok/s | 1.004× |
| Wall time | 67.03 ± 0.18 s | 67.32 ± 0.65 s | 0.996× |

Qwen2 1.5B smoke benchmark:

| Metric | Self Stage 1 | Reference Stage 1 | Self/reference |
|---|---:|---:|---:|
| Prefill throughput | 975.4 ± 5.0 tok/s | 980.8 ± 1.9 tok/s | 0.994× |
| Decode throughput | 4.30 ± 0.00 tok/s | 4.30 ± 0.01 tok/s | 1.001× |
| Wall time | 39.68 ± 0.05 s | 39.72 ± 0.10 s | 0.999× |

The 1.5B result is a narrow N=2 smoke benchmark with one request per
repetition, not a full N=3 sweep.

For both models, the short greedy generation sanity check matched exactly for
the first 16 generated tokens.

## Command

```bash
pdm run python benches/build_research_progress_summary.py
```

## Phase 1 Conclusion

The research package now has a clean transition:

1. Phase 0 measures and explains the Stage 1 decode bottleneck.
2. Phase 0.5 profiles the full-context decode step and shows that Python loop
   overhead is not the main cost.
3. Phase 1 starts from a faithful self-implemented Stage 1 baseline.
4. The self baseline matches the reference baseline on Qwen2 0.5B throughput,
   with a narrow 1.5B smoke check showing the same pattern.

This is enough to discuss Phase 1 progress on 2026-07-01 without starting
optimization prematurely.

## Key Phase 2 Results

Qwen2 0.5B benchmark:

| Metric | Stage 1 full-context | Stage 2 KV fallback | Stage 2 / Stage 1 |
|---|---:|---:|---:|
| Prefill throughput | 2217.6 ± 54.7 tok/s | 2436.7 ± 92.4 tok/s | 1.099× |
| Decode throughput | 7.27 ± 0.20 tok/s | 65.78 ± 0.41 tok/s | 9.054× |
| Wall time | 55.97 ± 1.46 s | 6.32 ± 0.03 s | 0.113× |
| Peak memory | 1.89 GB | 0.93 GB | 0.493× |

Qwen2 1.5B smoke benchmark:

| Metric | Stage 1 full-context | Stage 2 KV fallback | Stage 2 / Stage 1 |
|---|---:|---:|---:|
| Prefill throughput | 803.5 ± 7.8 tok/s | 780.4 ± 22.3 tok/s | 0.971× |
| Decode throughput | 3.41 ± 0.01 tok/s | 46.66 ± 0.55 tok/s | 13.673× |
| Wall time | 50.03 ± 0.10 s | 3.85 ± 0.05 s | 0.077× |
| Peak memory | 4.50 GB | 1.93 GB | 0.428× |

The Stage 2 path is explicitly labeled as KV-cache fallback. It uses MLX
built-in quantized matmul and should not be presented as a custom kernel
benchmark.

## Stage 2 Attribution Result

Qwen2 0.5B attribution benchmark:

| Arm | Decode throughput | Wall time | Peak memory |
|---|---:|---:|---:|
| Stage 1 full-context ref | 7.16 tok/s | 56.84 s | 1.89 GB |
| Stage 2 full-context | 7.69 tok/s | 52.87 s | 1.20 GB |
| Stage 2 KV-cache | 65.65 tok/s | 6.34 s | 0.93 GB |
| mlx-lm | 189.43 tok/s | 2.30 s | 0.53 GB |

Attribution ratios:

| Comparison | Decode ratio | Wall-time ratio |
|---|---:|---:|
| Stage 2 full-context / Stage 1 ref | 1.073x | 0.930x |
| Stage 2 KV-cache / Stage 2 full-context | 8.542x | 0.120x |
| mlx-lm / Stage 2 KV-cache | 2.886x | 0.362x |
| Stage 2 KV-cache / Stage 1 ref | 9.168x | 0.111x |

This shows that the main Stage 2 fallback gain comes from persistent KV-cache
decode, not merely from switching to the Stage 2 implementation stack.

## Stage 2 Decode Component Profile

At prompt length 512, the Stage 2 KV-cache one-step decode profile is:

| Component | Mean time | Share |
|---|---:|---:|
| QKV projection | 5.75 ms | 13.6% |
| KV-cache update | 4.92 ms | 11.6% |
| Attention core | 8.93 ms | 21.1% |
| Attention output projection | 4.44 ms | 10.5% |
| MLP | 8.95 ms | 21.1% |
| Elementwise/norm | 6.43 ms | 15.2% |
| LM head | 2.66 ms | 6.3% |
| Python loop/sampler | 0.32 ms | 0.8% |

The Python loop/sampler share is below 1%, so the remaining mlx-lm gap should
not be framed as a simple Python loop problem.

## Stage 3 Speculative Decoding Accounting

The first Stage 3 prototype measures greedy speculative decoding against the
Stage 2 KV-cache greedy baseline.

Qwen2 0.5B self-pair sanity:

| Draft length | Acceptance | Target calls / decode token | Decode ratio | Exact match |
|---:|---:|---:|---:|---:|
| 1 | 100.0% | 0.508 | 0.655x | yes |
| 2 | 100.0% | 0.333 | 0.734x | yes |
| 4 | 100.0% | 0.238 | 0.820x | yes |
| 8 | 100.0% | 0.111 | 0.856x | yes |

Qwen2 1.5B target with Qwen2 0.5B draft N=3:

| Draft length | Acceptance | Target calls / decode token | Decode ratio | Exact match |
|---:|---:|---:|---:|---:|
| 2 | 76.5% | 0.405 | 0.780x | yes |
| 4 | 63.6% | 0.294 | 0.710x | yes |
| 8 | 45.0% | 0.286 | 0.555x | yes |

The Stage 3 conclusion is that the greedy speculative mechanism is correct and
does reduce target-model calls, but the current prototype does not yet convert
that call-count reduction into decode speedup.

The timing split explains why: draft-model work takes 64.4-71.5% of
speculative decode time in the formal N=3 sweep, while target verification is
already reduced to 35.5-28.5%.

## Stage 3 Output-Length Scan

The follow-up output-length scan tests generated lengths 64, 128, and 256 with
draft lengths 2, 4, and 8.

| Output tokens | Best draft length | Best decode ratio | Acceptance | Target calls / decode token |
|---:|---:|---:|---:|---:|
| 64 | 2 | 0.788x | 76.5% | 0.405 |
| 128 | 2 | 0.812x | 87.4% | 0.374 |
| 256 | 4 | 0.874x | 87.9% | 0.227 |

This shows that longer output improves the speculative decode ratio, but it
still does not cross 1.0x in the current prototype.

## Current Conclusion

The evidence chain now extends into Phase 3:

1. Phase 0 measures the decode bottleneck.
2. Phase 0.5 shows full-context decode spends most time in model computation,
   not Python loop overhead.
3. Phase 1 validates that the self Stage 1 baseline matches the reference path.
4. Phase 2 fallback removes full-context decode and gives a 9.05× decode
   improvement on 0.5B, with a 1.5B smoke check showing 13.67×.
5. Stage 2 attribution shows that the isolated persistent KV-cache contribution
   is 8.542× over the Stage 2 full-context control.
6. Stage 2 decode profiling shows the remaining cached-decode cost is spread
   across attention, MLP, projections, KV-cache update, and elementwise work,
   while Python loop/sampler overhead is below 1%.
7. Stage 3 speculative decoding now has a measured tradeoff result: target
   calls per decode token drop substantially, but the current prototype remains
   below Stage 2 KV-cache throughput.
8. Stage 3 output-length scanning shows the best measured decode ratio improves
   to 0.874x at 256 output tokens, so longer outputs help but do not solve the
   overhead problem alone.

The next question is whether Stage 3 can improve wall time with a cheaper draft
path or lower-overhead verification, since longer decode workloads alone did
not cross the speedup threshold.

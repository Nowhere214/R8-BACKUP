# Group Meeting Report: Final Research Consolidation

Date: 2026-11-25

## Goal

The goal for this checkpoint is to consolidate the full research story before
the late-November completion target.

## Final Story

The project should be presented as a staged measurement study:

| Stage | Main question | Evidence |
|---|---|---|
| Phase 0 | Where is the production gap? | mlx-lm vs Stage 1 reference benchmarks |
| Stage 1 | Can the baseline be reproduced? | self/reference agreement and component profiling |
| Stage 2 | Which optimization explains the first major gain? | KV-cache fallback and attribution benchmark |
| Stage 3 | Does an advanced serving technique pay off on Apple Silicon? | speculative decoding result or failure mode |

## Locked Measured Results

Measured results that should be safe to cite:

| Result | Value |
|---|---:|
| Stage 2 KV-cache / Stage 1 ref decode | 9.168x |
| Stage 2 KV-cache / Stage 2 full-context decode | 8.542x |
| Stage 2 full-context / Stage 1 ref decode | 1.073x |
| mlx-lm / Stage 2 KV-cache decode | 2.886x |
| Stage 2 KV-cache Python loop/sampler share | < 1% |
| Stage 3 0.5B self-pair exact greedy match | yes |
| Stage 3 0.5B self-pair target calls/token at draft 8 | 0.111 |
| Stage 3 1.5B target / 0.5B draft acceptance at draft 2 | 76.5% |
| Stage 3 1.5B target / 0.5B draft target calls/token at draft 2 | 0.405 |
| Stage 3 1.5B target / 0.5B draft decode ratio at draft 2 | 0.780x |
| Stage 3 1.5B target / 0.5B draft draft-time share at draft 2 | 64.4% |
| Stage 3 output-length scan best decode ratio | 0.874x |
| Stage 3 output-length scan best point | 256 output tokens, draft length 4 |

The Stage 2 rows should be treated as Qwen2 0.5B N=3 attribution results. The
Stage 3 1.5B/0.5B rows are also N=3, but on a narrower synthetic workload.

## Figure Inventory

Core figure set:

- `benches/phase0_n3.png`
- `benches/workload_length_scan_combined.png`
- `benches/decode_gap_scaling_analysis.png`
- `benches/stage1_decode_component_profile.png`
- `benches/stage1_self_baseline_compare.png`
- `benches/stage2_kv_fallback_benchmark.png`
- `benches/stage2_attribution_benchmark.png`
- `benches/stage2_decode_component_profile.png`
- `benches/stage3_speculative_accounting.png`
- `benches/stage3_speculative_accounting_qwen2_1p5b_draft0p5b.png`
- `benches/stage3_speculative_length_scan.png`

## Limitations to State Clearly

- The work is an Apple Silicon measurement and decomposition study.
- tiny-llm is an external educational platform, not the project's original
  codebase contribution.
- Stage 2 custom C++/Metal kernels are not claimed unless implemented later.
- Stage 3 is intentionally narrow and should not be described as a production
  serving system.
- Stage 3 currently shows a call-count reduction, not a decode speedup.
- Longer output improves the Stage 3 decode ratio but still does not cross 1.0x
  in the current prototype.
- Qwen2 7B Stage 1 full-context runs remain memory-risky on the 18 GB machine.

## December Deliverables

By early December, only writing and presentation work should remain:

- final synthesis: `docs/DECEMBER_FINAL_RESEARCH_SYNTHESIS_2026.md`,
- final report,
- presentation deck,
- speaker notes,
- appendix with commands and CSV references,
- final reproducibility checklist.

## Status

Stage 3 measured values are now filled in as a tradeoff result. The final
research story can close on the distinction between algorithmic target-call
reduction and realized wall-time speedup.

# July 2026 Progress Ledger

Prepared on: 2026-05-29

Purpose: keep the July reporting trail clear. This file separates completed
facts from optional future work, so later group meeting reports can be written
without searching through every benchmark file.

## How to Use This Ledger

Use this as the first lookup file when preparing any July update.

- For the 2026-07-01 formal update, use Section 1.
- For the 2026-07-15 group meeting, use Section 2.
- For late July or a possible 2026-07-31 update, use Section 3 only if the
  corresponding work is actually done.

Do not present optional items as completed results.

## 1. Ready for 2026-07-01

Status: completed and documented.

Main story:

Phase 0 measured the bottleneck, Phase 0.5 explained it, and Phase 1 now has a
self-implemented Stage 1 baseline that matches the reference implementation.

Evidence:

| Item | Artifact |
|---|---|
| Phase 0 result package | `docs/PHASE0_RESULTS.md` |
| Phase 0 reproducibility manifest | `docs/PHASE0_REPRODUCIBILITY_MANIFEST_2026-05-29.md` |
| Stage 1 component profile | `docs/STAGE1_DECODE_COMPONENT_PROFILE_2026-05-29.md` |
| Stage 1 implementation progress | `docs/PHASE1_STAGE1_IMPLEMENTATION_PROGRESS_2026-05-29.md` |
| Self baseline benchmark | `docs/PHASE1_SELF_BASELINE_BENCHMARK_2026-05-29.md` |
| Research-level manifest | `docs/RESEARCH_PROGRESS_SUMMARY_2026-05-29.md` |
| Formal progress draft | `docs/FORMAL_PROGRESS_UPDATE_2026-07-01_EN.md` |

Key numbers:

| Result | Value |
|---|---:|
| Phase 0 unified rows | 151 |
| Research-level data rows | 175 |
| Stage 1 decode component profile, prompt 512 MLP share | 50.6% |
| Stage 1 decode component profile, prompt 512 Python loop/sampler share | 0.1% |
| 0.5B self/reference decode ratio | 1.004× |
| 1.5B self/reference decode ratio, smoke benchmark | 1.001× |

Figures ready:

- `benches/stage1_decode_component_profile.png`
- `benches/stage1_self_baseline_compare.png`
- `benches/stage1_self_baseline_compare_1p5b_smoke.png`

Important caveat:

The 1.5B comparison is a narrow N=2 smoke benchmark with one request per
repetition. It is useful consistency evidence, not a full N=3 sweep.

## 2. Ready for 2026-07-15

Status: completed and documented.

Main story:

After the formal 7/1 update, the work moved from result production to Phase 2
readiness and the first correctness implementation step. I cleaned up the
reporting structure, audited the Stage 2 implementation path, then made the
KV-cache path executable with an MLX quantized fallback.

Evidence:

| Item | Artifact |
|---|---|
| Benchmark script output-prefix support | `benches/stage1_self_baseline_compare.py` |
| Research-level table including Phase 1 checks | `benches/research_progress_unified_summary.csv` |
| Phase 2 readiness audit | `docs/PHASE2_READINESS_AUDIT_2026-05-29.md` |
| Phase 2 KV-cache correctness progress | `docs/PHASE2_KV_CACHE_CORRECTNESS_2026-05-29.md` |
| 7/15 group meeting draft | `docs/GROUP_MEETING_REPORT_2026-07-15_EN.md` |

What changed after 7/1:

- the self-baseline benchmark script can write model-specific output files
  without overwriting previous results
- the research-level CSV now separates Phase 0 and Phase 1 rows with
  `phase_id`
- the Stage 2 path has been audited against local book chapters, tests, stubs,
  and reference implementation
- the first KV-cache correctness path has been implemented using MLX built-in
  quantized matmul as a fallback
- custom C++/Metal kernel work remains separated from the correctness path

Meeting-ready conclusion:

The next phase is now split into two layers. First, KV-cache correctness is
being validated through a fallback implementation. Second, quantized matmul and
Flash Attention remain toolchain-dependent kernel milestones. This avoids
mixing algorithmic progress with Xcode/extension setup risk.

Validation available for 7/15:

| Scope | Result |
|---|---|
| Stage 2 Day 1 Qwen2 0.5B subset | 3 passed, 6 deselected |
| Stage 2 Day 1 Qwen2 1.5B subset | 3 passed, 6 deselected |
| Stage 2 Day 2 quantized matmul small tests | 6 passed |
| Stage 2 Day 6 batching cache unit test | 1 passed, 25 deselected |
| Stage 2 Day 4 flash fallback CPU small sample | 1 passed, 19 deselected |

## 3. Late-July Optional Progress

Status: partially started. KV-cache correctness has started; benchmark and
custom kernel work remain planned only.

Recommended order if more progress is needed after 2026-07-15:

1. Add a small Stage 1 vs Stage 2 KV-cache benchmark for Qwen2 0.5B.
2. Keep it clearly labeled as MLX quantized fallback, not custom kernel work.
3. Compare decode throughput and per-token wall time.
4. Add stable benchmark rows to the research-level summary.
5. Decide whether to install full Xcode before custom C++/Metal kernel work.

Do not do these as late-July filler unless the current reporting materials are
already polished:

- full 7B Stage 2 runs
- Xcode installation
- custom Metal Flash Attention
- broad benchmark sweeps before Stage 2 correctness is stable

## 4. One-Line Status by Date

| Date | Status line |
|---|---|
| 2026-07-01 | Phase 0/0.5 closed into a measured and profiled bottleneck story; Phase 1 self Stage 1 baseline matches reference on 0.5B and a 1.5B smoke check. |
| 2026-07-15 | Phase 2 readiness audit completed; KV-cache correctness path started with MLX quantized fallback; extension-dependent kernels remain later milestones. |
| Late July | Optional only: benchmark Stage 1 vs Stage 2 fallback decode before starting custom kernels. |

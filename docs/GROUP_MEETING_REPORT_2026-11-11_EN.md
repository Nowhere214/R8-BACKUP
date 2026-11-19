# Group Meeting Report: Stage 3 Prototype Checkpoint

Date: 2026-11-11

## Goal

The goal for this checkpoint is to evaluate whether the Stage 3 speculative
decoding prototype is on track to become the final advanced-serving result.

## Prototype Scope

The implemented prototype stays narrow:

| Component | Scope |
|---|---|
| Draft model | Qwen2 0.5B |
| Target model | Qwen2 1.5B |
| Decode mode | Greedy speculative decoding |
| Draft lengths | 2, 4, 8 in the 1.5B/0.5B N=3 sweep; 1, 2, 4, 8 in the 0.5B self-pair sanity run |
| Baseline | Stage 2 KV-cache decode |

## Measured Results

Qwen2 0.5B self-pair sanity:

| Draft length | Acceptance | Target calls / decode token | Decode ratio vs baseline | Exact match |
|---:|---:|---:|---:|---:|
| 1 | 100.0% | 0.508 | 0.655x | yes |
| 2 | 100.0% | 0.333 | 0.734x | yes |
| 4 | 100.0% | 0.238 | 0.820x | yes |
| 8 | 100.0% | 0.111 | 0.856x | yes |

Qwen2 1.5B target with Qwen2 0.5B draft:

| Draft length | Acceptance | Target calls / decode token | Decode ratio vs baseline | Exact match |
|---:|---:|---:|---:|---:|
| 2 | 76.5% | 0.405 | 0.780x | yes |
| 4 | 63.6% | 0.294 | 0.710x | yes |
| 8 | 45.0% | 0.286 | 0.555x | yes |

## Decision Rule

The harness is valid enough to continue as the Stage 3 topic because:

- exact-match sanity passes;
- acceptance is non-trivial for the 1.5B/0.5B pair;
- target calls per decode token drop well below the Stage 2 KV-cache baseline.

However, the current prototype should be presented as a tradeoff rather than a
speedup. Decode throughput is still below the Stage 2 KV-cache baseline, which
means draft and verification overhead erase the target-call reduction in this
small workload.

The time split explains the regression: draft-model calls take 64.4-71.5% of
the speculative decode time, while target verification falls below 36%.

## Evidence References

- `docs/PHASE3_SPECULATIVE_ACCOUNTING_2026-05-30.md`
- `docs/STAGE3_SPECULATIVE_DECODING_PLAN_2026-05-30.md`
- `docs/OCTOBER_PROGRESS_LEDGER_2026.md`
- `docs/PHASE2_ATTRIBUTION_BENCHMARK_2026-05-30.md`

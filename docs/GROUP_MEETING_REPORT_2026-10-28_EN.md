# Group Meeting Report: Stage 2 Profile and Stage 3 Topic Selection

Date: 2026-10-28

## Goal

The goal for this checkpoint is to use the Stage 2 decode profile to choose the
Stage 3 topic under the constraint that the full research should be completed
by late November.

## Current Evidence

The Stage 2 attribution benchmark gives a clear result:

| Comparison | Decode ratio |
|---|---:|
| Stage 2 full-context / Stage 1 ref | 1.073x |
| Stage 2 KV-cache / Stage 2 full-context | 8.542x |
| mlx-lm / Stage 2 KV-cache | 2.886x |

This means the main Stage 2 structural gain is already measured. The next
advanced topic should answer a new systems question rather than simply repeat
KV-cache evidence.

## Stage 2 Decode Profile

The Stage 2 KV-cache decode profile shows that the remaining cost is spread
across model compute and cache/layout work. At prompt length 512:

| Component | Mean time | Share |
|---|---:|---:|
| Attention core | 8.93 ms | 21.1% |
| MLP | 8.95 ms | 21.1% |
| QKV projection | 5.75 ms | 13.6% |
| KV-cache update | 4.92 ms | 11.6% |
| Attention output projection | 4.44 ms | 10.5% |
| Elementwise/norm | 6.43 ms | 15.2% |
| LM head | 2.66 ms | 6.3% |
| Python loop/sampler | 0.32 ms | 0.8% |

This makes a direct Python-loop optimization a weak Stage 3 candidate. The
remaining gap is more likely in kernels, layout, runtime behavior, or a higher
level serving method that reduces target-model work.

## Candidate Topics

| Topic | Research value | Risk | Deadline fit |
|---|---|---|---|
| Speculative Decoding | Tests draft/verify serving tradeoff | Medium | Good |
| Paged Attention | Tests memory-management serving design | High | Risky |
| Longer-context scan | Extends Phase 0 evidence | Low | Too incremental |
| Custom C++/Metal kernels | Targets remaining mlx-lm gap | High | Risky |

## Recommendation

Choose Speculative Decoding as the Stage 3 topic.

The planned research question is:

```text
On Apple Silicon, when does speculative decoding improve end-to-end decode
speed, and when does verifier overhead erase the benefit?
```

## Proposed Stage 3 Setup

| Role | Candidate |
|---|---|
| Draft model | Qwen2 0.5B |
| Target model | Qwen2 1.5B |
| Baseline | Stage 2 KV-cache decode |
| Variant sweep | draft length 2, 4, 8 |

The initial implementation should use greedy speculative decoding and avoid
batching, 7B models, and long-context prompts.

## Deliverable

The October deliverable is a Stage 3 plan, not a claimed Stage 3 result:

- `docs/STAGE3_SPECULATIVE_DECODING_PLAN_2026-05-30.md`
- `docs/PHASE2_STAGE2_DECODE_COMPONENT_PROFILE_2026-05-30.md`

## Next Step

Implement the smallest greedy speculative decoding harness and measure target
calls per generated token, acceptance rate, and wall time per output token.

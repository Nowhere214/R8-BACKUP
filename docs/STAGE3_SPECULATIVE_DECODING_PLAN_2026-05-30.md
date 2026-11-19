# Stage 3 Speculative Decoding Plan

Date: 2026-05-30

## Decision

For the 11月底 completion target, Stage 3 should use Speculative Decoding as
the primary advanced-serving topic.

Paged Attention remains important, but it is higher risk on this machine
because it requires more memory-management design and is harder to turn into a
small, clean, measured result before the deadline.

## Research Question

The Stage 3 question should be:

```text
On Apple Silicon, under a controlled small-model setup, when does speculative
decoding improve end-to-end decode speed, and when does verifier overhead erase
the benefit?
```

This connects naturally to the previous roadmap:

- Phase 0 measured the production gap.
- Stage 1 established the unoptimized baseline.
- Stage 2 measured the KV-cache contribution.
- Stage 3 tests a frontier serving optimization against the Stage 2 baseline.

## Minimal Experimental Design

Use a target/draft pair:

| Role | Candidate |
|---|---|
| Draft model | Qwen2 0.5B |
| Target model | Qwen2 1.5B |

Main metrics:

| Metric | Purpose |
|---|---|
| acceptance rate | Whether the draft model is useful |
| target calls per generated token | Whether verification reduces target work |
| wall time per output token | End-to-end serving impact |
| output agreement / sanity prompts | Basic correctness guard |

## Implementation Boundaries

The first implementation should be deliberately narrow:

- greedy speculative decoding first;
- fixed draft length candidates, for example 2, 4, and 8;
- no batching in the first pass;
- no 7B model in the first pass;
- no claim of production-level speculative serving.

## Success Criteria

Stage 3 is considered successful if it produces:

1. a working greedy speculative decoding prototype;
2. a measured speed/acceptance curve for at least one draft length sweep;
3. a comparison against the Stage 2 KV-cache baseline;
4. a clear explanation of whether Apple Silicon behavior matches the expected
   mechanism from the original speculative decoding literature.

## Fallback Criteria

If implementation risk becomes too high, the fallback deliverable is still a
valid Stage 3 research result:

- implement the verifier/draft accounting harness,
- report why the expected speedup does or does not appear,
- present the failure mode as a systems limitation rather than hiding it.

## Why Not Paged Attention First

Paged Attention is a better long-term serving topic, but it is less suitable as
the first Stage 3 deliverable here:

- it is more memory-layout heavy;
- its strongest value appears under multi-request serving pressure;
- it may require a larger batching harness before the result is meaningful;
- it is harder to finish cleanly by 2026-11-30.

Speculative Decoding gives a smaller, more testable Stage 3 target.

## First Measured Checkpoint

The first greedy speculative accounting prototype has now been run:

- `docs/PHASE3_SPECULATIVE_ACCOUNTING_2026-05-30.md`
- `benches/stage3_speculative_accounting.py`

Measured N=3 result for Qwen2 1.5B target / Qwen2 0.5B draft:

- exact greedy match passes for draft lengths 2, 4, and 8;
- target calls per decode token drop to 0.405, 0.294, and 0.286;
- acceptance falls to 76.5%, 63.6%, and 45.0%;
- draft-model work takes 64.4-71.5% of speculative decode time;
- the current prototype does not yet beat the Stage 2 KV-cache greedy baseline
  in decode throughput.

So Stage 3 should be framed as a measured systems tradeoff unless later runs
turn the target-call reduction into wall-time speedup.

## Output-Length Follow-Up

A follow-up output-length scan has also been run:

- `docs/PHASE3_SPECULATIVE_LENGTH_SCAN_2026-05-30.md`
- `benches/stage3_speculative_length_scan.py`

Generated lengths 64, 128, and 256 all preserve exact greedy match for draft
lengths 2, 4, and 8. The best measured decode ratio improves to 0.874x at 256
output tokens with draft length 4, but still remains below the Stage 2 KV-cache
baseline.

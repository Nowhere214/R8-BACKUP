# SiliconLLM November Progress Ledger

Date range: 2026-11-01 to 2026-11-30

## Meetings

- 2026-11-11: Stage 3 speculative decoding prototype checkpoint
- 2026-11-25: final research consolidation checkpoint

## November Goal

November should complete the research, not open a broad new direction. The
recommended goal is to finish one narrow Stage 3 result and consolidate the
full story:

```text
Phase 0 diagnosis -> Stage 1 baseline -> Stage 2 attribution -> Stage 3
speculative decoding tradeoff.
```

## Stage 3 Minimum Result

The minimum acceptable Stage 3 result is:

1. greedy speculative decoding prototype,
2. draft length sweep over 2, 4, and 8,
3. acceptance-rate curve,
4. wall-time-per-output-token curve,
5. comparison against Stage 2 KV-cache baseline.

## 2026-11-11 Checkpoint

Expected report:

- speculative decoding harness runs and exact-match sanity passes;
- target/draft tokenizer compatibility is sufficient for Qwen2 1.5B / 0.5B;
- 0.5B self-pair reaches 100% draft acceptance;
- Qwen2 1.5B target / 0.5B draft N=3 sweep reaches 76.5% acceptance at draft
  length 2, 63.6% at draft length 4, and 45.0% at draft length 8;
- decode-time split shows draft-model work takes 64.4-71.5% of speculative
  decode time;
- verifier and draft overhead still dominate wall time in the current
  prototype.
- follow-up output-length scan shows decode ratio improves with longer outputs,
  reaching 0.874x at 256 output tokens with draft length 4, but still does not
  cross the Stage 2 KV-cache baseline.

Deliverable:

- `docs/GROUP_MEETING_REPORT_2026-11-11_EN.md`
- `docs/PHASE3_SPECULATIVE_ACCOUNTING_2026-05-30.md`
- `docs/PHASE3_SPECULATIVE_LENGTH_SCAN_2026-05-30.md`

## 2026-11-25 Checkpoint

Expected report:

- final figure/table inventory,
- main claims and limitations,
- which results are measured and which are future work,
- final completion percentage estimate.
- Stage 3 measured tradeoff: target calls per decode token drop below 0.5, but
  the current prototype does not yet beat the Stage 2 KV-cache baseline.
- Stage 3 output-length scan: longer outputs improve the decode ratio but do
  not yet produce speedup.

Deliverable:

- `docs/GROUP_MEETING_REPORT_2026-11-25_EN.md`
- `docs/DECEMBER_FINAL_RESEARCH_SYNTHESIS_2026.md`

## Completion Criteria

By 2026-11-30, the research should have:

- a locked evidence table,
- publication-ready figures,
- at least one Stage 3 result or documented Stage 3 failure mode,
- a final limitations section,
- a clear December presentation/writing package.

## Status

Stage 3 now has a measured greedy speculative decoding accounting result. The
result should be presented as a tradeoff, not as a production speedup claim.
The December final synthesis has also been prewritten as the handoff point for
report and presentation drafting.

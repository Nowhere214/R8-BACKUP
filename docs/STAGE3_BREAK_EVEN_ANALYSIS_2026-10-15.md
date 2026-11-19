# Stage 3 Break Even Analysis

Analysis date: 2026-10-15. Source measurements: the saved 2026-05-30
output-length scan, Qwen2 1.5B target / 0.5B draft, two requests and two
repetitions per configuration. No new model inference was run for this check.

## Reproduce

```bash
pdm run python benches/analyze_stage3_break_even.py
```

Input: `benches/stage3_speculative_length_scan_summary.csv`.

For equal output token counts, the throughput ratio here is baseline mean
decode time divided by speculative mean decode time. This is not the mean of
per-repetition throughput ratios; rounding matches the existing scan report.

## Calculation

Let B be baseline decode time, S speculative decode time, and D draft time.
Holding acceptance, verification, rollback and the execution schedule fixed,
the fraction of draft time that must be removed to break even is:

```text
required_draft_reduction = max(0, S - B) / D
```

This includes the saved draft-time accounting, including draft-cache alignment.
Prefill is excluded. A different drafting method may change acceptance and
verification cost, so this is a sensitivity calculation, not a speedup forecast.

| Output tokens | k | Decode ratio | Draft time reduction needed |
| ---: | ---: | ---: | ---: |
| 64 | 2 | 0.788x | 32.7% |
| 64 | 4 | 0.709x | 41.4% |
| 64 | 8 | 0.554x | 62.3% |
| 128 | 2 | 0.812x | 29.0% |
| 128 | 4 | 0.805x | 27.2% |
| 128 | 8 | 0.707x | 40.5% |
| 256 | 2 | 0.848x | 23.3% |
| 256 | 4 | 0.874x | 17.8% |
| 256 | 8 | 0.836x | 21.9% |

## Implication for the Controller

Choosing the best fixed configuration for each measured output-length workload,
with baseline available, selects baseline at all three lengths (1.000x before
selection overhead). This does not bound per-request variation within a bucket
or switching within a generation. The saved summary has neither the request
detail nor the per-step timing needed to evaluate those policies.

A controller needs online features available at decision time. Actual final
output length must not be used as a known feature; a configured token limit
can be used. Calibration, exploration, draft-cache catch-up after baseline
steps, and switching overhead must be charged to the measured controller time.
Skipping draft forwards leaves its cache behind the accepted target prefix;
resuming speculation requires explicit cache alignment or rebuilding.

## Pilot Protocol

Use one code-repair workflow with a maximum of two repair attempts. Start with
30 MBPP+ tasks chosen by a fixed seed before running any decoding method.
Run generated code in an isolated execution environment with timeouts. Use
visible tests for repair feedback and separate held-out tests for final scoring.
Do not expose reference solutions or held-out test results in repair prompts.

Record token IDs, phase, context length, draft/verify time, accepted tokens,
history lookup time, cache alignment cost and tool execution time. Only earlier
accepted output may enter the history cache. Initial natural-prompt tests should
check EOS, output limits and exact greedy agreement before timing comparisons.
The existing synthetic exact-match results are not sufficient for agent claims.

Compare ordinary decoding, fixed k, an acceptance-only rule and the cost-based
controller. Add global suffix and structure-isolated history baselines if a
history drafter is implemented. Use separate calibration and evaluation tasks,
and report both cold-history and warm-history behavior without looking ahead.

Status: break-even analysis completed. A first five-task MBPP sanitized pilot
has since run; see `docs/AGENT_CODE_REPAIR_PILOT_2026-10-15.md`. The 30-task
expansion, MBPP+, history drafter and online controller remain planned.
No agent speedup is claimed.

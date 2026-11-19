# Phase 1 Stage 1 Implementation Progress

Date: 2026-05-29

## Goal

This records the first Phase 1 implementation pass after the Phase 0 and
Phase 0.5 measurements.

The goal was not to optimize yet. The goal was to implement the faithful Stage 1
baseline in `src/tiny_llm/` and validate it against the reference tests.

## Implemented Files

| File | Implemented content |
|---|---|
| `src/tiny_llm/basics.py` | `linear`, `silu` |
| `src/tiny_llm/attention.py` | simple attention, simple MHA, causal mask, grouped attention |
| `src/tiny_llm/positional_encoding.py` | traditional and non-traditional RoPE |
| `src/tiny_llm/layer_norm.py` | RMSNorm with float32 accumulation |
| `src/tiny_llm/qwen2_week1.py` | Qwen2 attention, MLP, transformer block, Stage 1 model wrapper |
| `src/tiny_llm/embedding.py` | embedding lookup and embedding-as-linear |
| `src/tiny_llm/generate.py` | Stage 1 greedy generation loop |
| `src/tiny_llm/sampler.py` | temperature, top-k, and top-p sampling |

## Validation

Commands run:

```bash
pdm run test --week 1 --day 1
pdm run test --week 1 --day 2
pdm run test --week 1 --day 3
pdm run test --week 1 --day 4
pdm run test --week 1 --day 5 -- -k "task_1 or task_2_embedding or qwen_2_05b"
pdm run test --week 1 --day 5 -- -k "qwen_2_15b"
pdm run test --week 1 --day 6
pdm run test --week 1 --day 7
```

Results:

| Scope | Result |
|---|---|
| Stage 1 Day 1 | 36 passed |
| Stage 1 Day 2 | 16 passed |
| Stage 1 Day 3 | 60 passed |
| Stage 1 Day 4 | 22 passed |
| Stage 1 Day 5 targeted safe subset | 12 passed, 4 deselected |
| Stage 1 Day 5 Qwen2 1.5B targeted subset | 2 passed, 14 deselected |
| Stage 1 Day 6 | 1 skipped, official test file has no unit tests |
| Stage 1 Day 7 | 1 skipped, official test file has no unit tests |

Additional check:

```bash
pdm run python - <<'PY'
import mlx.core as mx
from tiny_llm.sampler import make_sampler
for kwargs in [
    dict(temp=0, top_p=None, top_k=None),
    dict(temp=0.7, top_p=None, top_k=3),
    dict(temp=0.7, top_p=0.8, top_k=None),
    dict(temp=0.7, top_p=0.8, top_k=3),
]:
    sampler = make_sampler(**kwargs)
    out = sampler(mx.array([[0.0, -1.0, -2.0, -3.0, -4.0]]))
    mx.eval(out)
    print(kwargs, out.tolist())
PY
```

This confirmed that the sampler paths execute for greedy, top-k, top-p, and
combined top-k/top-p settings.

I also ran the self baseline benchmark:

```bash
pdm run python benches/stage1_self_baseline_compare.py
```

Results:

| Engine | N | Prefill tok/s | Decode tok/s | Wall time |
|---|---:|---:|---:|---:|
| Self Stage 1 | 3 | 2463.8 ± 23.2 | 9.94 ± 0.03 | 67.03 ± 0.18 s |
| Reference Stage 1 | 3 | 2353.7 ± 100.7 | 9.90 ± 0.10 | 67.32 ± 0.65 s |

The self/reference ratio is 1.047× for prefill and 1.004× for decode.

The short greedy generation sanity check also matched exactly for the first 16
generated tokens. Details are recorded in
`docs/PHASE1_SELF_BASELINE_BENCHMARK_2026-05-29.md`.

## Why Day 5 Was Targeted

I did not run the full Day 5 suite.

The full Day 5 suite includes Qwen2 7B Stage 1 model construction if the local
model exists. This machine has the 7B model cached, and earlier Phase 0 work
already showed that Stage 1 7B dequantization requires about 17 GB or more of
memory on an 18 GB unified-memory machine.

So the safe validation target was:

- transformer block correctness
- Qwen2 0.5B embedding correctness
- Qwen2 0.5B full Stage 1 model wrapper correctness
- Qwen2 1.5B full Stage 1 model wrapper correctness

This is enough to validate the implementation path without repeating the known
7B feasibility failure.

## Current Interpretation

This changes the 2026-07-01 story.

Before this pass, the next step was only "start Phase 1 Stage 1 implementation."
Now the stronger statement is:

Phase 1 Stage 1 implementation has started and the core path through Qwen2 0.5B
and 1.5B is passing targeted tests. The self-implemented 0.5B Stage 1 baseline
also matches the reference path in a small N=3 throughput benchmark and a short
greedy generation sanity check.

## Next Steps

The self Stage 1 0.5B result has now been added into the broader research
manifest:

- `benches/build_research_progress_summary.py`
- `benches/research_progress_unified_summary.csv`
- `docs/RESEARCH_PROGRESS_SUMMARY_2026-05-29.md`

Recommended next order:

1. Use the current evidence chain for the 2026-07-01 formal update.
2. If the 1.5B smoke result needs to become formal benchmark evidence, rerun it
   as a full N=3 comparison with more than one request per repetition.
3. If correctness and throughput remain close, move to Phase 2 planning after
   the formal update.

Do not start KV-cache optimization before the 2026-07-01 baseline story is
finished.

# Local mlx-lm API Pilot: November 17, 2026

## Scope

This is a new, small local experiment. It is separate from the withdrawn October experiments and from the adopted external Claude results. It does not replace either set of records or update the presentation.

Question: can fixed historical proposals run through the complete mlx-lm streaming wrapper, use native cache trimming without replay, and match ordinary greedy output on a few short requests?

Ordinary decoding calls the installed, unmodified `mlx_lm.stream_generate`. Fixed history uses the same streaming function code with an isolated lower-level token-generator dispatch extension. Native target model, KVCache, EOS handling, detokenization, response construction and wired-memory context are retained. No installed package file or module global is changed. This is **an API-integrated history extension**, not an existing stock history-drafting API, and not the former custom synchronous-loop benchmark. It does not claim the fastest possible engine configuration.

## Configuration

- Apple M3 Pro, 18 GiB unified memory. MLX reports a recommended GPU working set of about 12 GiB.
- Installed mlx-lm 0.27.1, MLX 0.29.1; versions were not upgraded.
- Cached Qwen2-1.5B-Instruct-MLX, 4-bit weights, group size 64. Revision `42d944c83758c69dd75180f41d185ad704e7c3ad`; weight file about 828 MiB. No downloads, second model, or training.
- Greedy generation, maximum 192 tokens, fixed draft limit 16, four-token retrieval key. Actual outputs were 34-62 tokens including EOS; none hit the length cap.
- Five author-written requests: one cold generation control, one unchanged-code copy control, and three small behavioral edits. Source code is known before generation and included in the same prompt for both methods. Lookup can also use already-emitted tokens within the current request. No baseline output or future token is used as history.
- Unquantized, non-rotating native KVCache. Reject recovery uses trim only, with no snapshot/restore/replay. Replaying a known baseline to construct a cache is not performed.

## Timing and Validation

Each process warms both modes, runs five untimed output-validation pairs, then runs five requests x two modes x three shuffled repetitions. Shuffle seed is 20261117. Each request gets a fresh cache. The timer covers consumption of the complete streaming API, detokenization, draft lookup/index construction, native verification/cache handling, response collection, and final synchronization. It excludes model loading, warmup, prompt/chat-template tokenization and allocator-cache clearing. These are pretokenized-request API times, not full agent or raw-text-to-output wall times.

The run directory `runs/trim_k16_n3_confirm/` is the primary record. It contains manifest/configuration and source hashes, task prompts and histories, every output sequence, validation pairs, repeated timings and aggregate statistics. Matching includes the full emitted token sequence, EOS/stop status, terminal token and decoded text. All 5 validation pairs matched; all 30 timed calls matched their ordinary reference. These checks do not prove bitwise KV-cache equality or exact output for all prompts.

## Primary Results

The times below sum the three repeats. Speed ratio is ordinary time divided by fixed-history time, not a percentage reduction in time.

| Cases | Requests | Ordinary total | Fixed-history total | Ratio |
| --- | ---: | ---: | ---: | ---: |
| All cases | 5 | 6.906 s | 4.990 s | 1.384x |
| Cases with supplied history | 4 | 5.860 s | 3.987 s | 1.470x |
| Behavioral edits only | 3 | 4.699 s | 3.335 s | 1.409x |

The cold control invokes the same ordinary API in both modes. Its approximately 4% time difference is noise, not drafting benefit. The copy control alone was 1.783x and must not be presented as useful-edit performance. The case-insensitive edit output failed the requested original-spelling requirement by inspection; it was identical under both modes. Output preservation does not imply task success. No generated code was executed and no task-success score is claimed.

An earlier independent-process feasibility run, `runs/trim_k16_n3_v2/`, also had 5/5 matching pairs and 30/30 matching timed calls: all-case ratio 1.342x, history-case ratio 1.407x, edit-only ratio 1.350x. It is retained separately, not pooled. Its manifest described history too narrowly as supplied source only; the adapter also indexes already-emitted tokens. The confirmation run corrects that wording without changing the drafting algorithm. These are the same five cases across sessions, not ten independent tasks.

## Memory and Runtime

The confirmation process recorded a peak MLX allocation of **1.003 GiB** and process maximum resident memory of **1.193 GiB**. These are different measurements and should not be added together. Total work within the runner was **18.28 seconds**, including loading, warmup, output checks and repeated timings; Python startup/import time and implementation work are not included. Model load was 0.65 seconds with the local cache. No out-of-memory error occurred; no other app was closed. System-wide compression and historical swap were already present, so this does not establish swap-free execution.

This hardware is sufficient for this one-model short-request pilot without asking the user to release memory. Longer contexts, larger samples and concurrent apps require renewed checks. A 20-30-request extension with longer outputs is provisionally budgeted at 5-20 minutes for execution and inspection, not a measured guarantee.

## Limits and Next Step

The pilot supports local API integration feasibility and sampled exact-output agreement with trim-only recovery. It does **not** establish that replay can be removed for all contexts, reproduce the external 1.45x/1.65x results, evaluate the dynamic controller, validate its cost model, provide a public benchmark, or measure full agent episodes. The original replay-based cost formula cannot be validated by these trim-only results. Tiny sub-second requests and only three repeats have timing noise; no confidence interval is claimed.

Next: freeze public task/request inputs, include longer contexts and substantive edits, repeat exact-output checks in independent sessions, and compare fixed versus dynamic control using the same native verifier and trim path. Measure actual trim/policy cost before revising the cost model. Preserve failure cases instead of filtering for speed.

## Reproduction

From the repository root, with Metal access:

```bash
env PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python projects/mlx_lm_api_pilot_2026_11_17/run_pilot.py \
  --model /Users/laiqixiao/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct-MLX/snapshots/42d944c83758c69dd75180f41d185ad704e7c3ad \
  --out projects/mlx_lm_api_pilot_2026_11_17/runs/new_session \
  --repeats 3 --max-tokens 192
```

The output directory must not exist, to avoid overwriting previous results.

```bash
.venv/bin/python projects/mlx_lm_api_pilot_2026_11_17/verify_pilot.py
pdm run -k pre_run test -- -- -k 'mlx_lm_history_api or native_history or adaptive_decode'
```

The dependency-sync hook is skipped because the installed environment is already available and pinned in the manifest. The extra argument separator is required by this installed pdm/script combination. No extensions changed.

Verification outcome: 92 related tests passed (226 deselected), covering the new adapter, existing historical retrieval/decoding and native cache adapter. Ruff checks passed. The saved-record verifier passed output, count, hash and aggregate arithmetic checks; receipt: `runs/trim_k16_n3_confirm/verification.json`.

The initial `runs/trim_k16_n3/` attempt exited before generation because a tokenizer return object was not JSON serializable. It contains only a manifest, no timing results. All attempts remain on disk.

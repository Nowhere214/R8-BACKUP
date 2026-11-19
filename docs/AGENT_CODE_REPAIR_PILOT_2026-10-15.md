# First Code Repair Agent Pilot

Date: 2026-10-15

The first agent workflow now runs on the existing Apple M3 Pro with the cached
Qwen2 1.5B Instruct MLX checkpoint. It generates Python code, executes a visible
test, sends failure feedback back to the model, and allows up to two repairs.
Decoding uses our Stage 2 KV-cache greedy path. Each turn rebuilds the cache;
cross-turn cache reuse and speculative decoding are not enabled.

## Tasks and Test Separation

Source: [Google MBPP sanitized](https://github.com/google-research/google-research/tree/master/mbpp).
This is the original hand-verified MBPP subset, not MBPP+.

Sample five tasks from the official test ID range 11 through 510 using seed 42.
The selected IDs, fixed before inference, are 113, 61, 274, 257, and 245.
Each task has one visible example and two held-out tests. Only the visible
test can trigger a repair. Held-out tests run after the agent stops, and their
results never enter a repair prompt. Reference solutions are not used.

The test tool uses macOS sandbox-exec, denies network access, file writes and
forking, and restricts home-directory data access. Generated code also passes
an AST policy check. Test execution has a 2-second CPU limit and 5-second wall
limit. This is a local constrained pilot runner, not a general hostile-code
service; no memory quota is enforced.

## Results

Two runs used the same five tasks. Run 1 returned assertion/traceback feedback.
Run 2 added actual and expected values for simple equality assertions reached
by the test tool. An assertion inside the generated module can still fail
before that test instrumentation runs.

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Distinct tasks | 5 | same 5 |
| Initial generation calls | 5 | 5 |
| Repair calls | 4 | 4 |
| Final visible and held-out tests passed | 3/5 | 3/5 |
| Agent execution time | 27.04 s | 26.57 s |
| Model generation wall time | 26.27 s | 26.18 s |
| Visible test tool wall time | 0.47 s | 0.35 s |
| Peak MLX allocation | 2.85 GiB | 2.85 GiB |

The timing difference between runs is not evidence of a speedup: neither run
is a repeated performance benchmark. Agent time excludes model loading,
warmup and held-out evaluation. Peak MLX allocation is not total process RAM.

Tasks 113, 61 and 257 passed on the first attempt. Tasks 274 and 245 failed
both repair attempts. All four repair code strings in each run exactly repeated
the preceding failed solution. For example, task 274 kept returning 30 for
`even_binomial_Coeff_Sum(4)` when the visible test expected 8, even after that
mismatch appeared explicitly in the Run 2 feedback.

Run 2 generated 1,155 tokens including EOS across nine LLM calls, with 2.77 s
of prefill and 23.41 s of decode time. Every call ended at EOS rather than the
384-token cap. The traces include prompt and output token IDs, phase labels,
each step's latency, code, test feedback, and within-task repeated spans.

## Meaning for the Research

The generation, test, feedback and repair loop is implemented and has executed
real model calls. The pilot reveals a repeated-failure behavior in this small
sample. It does not establish that the model is generally unable to repair
code, and it does not establish a speculative decoding speedup.

Repeated failed code can inflate history coverage. Future analysis must keep
successful and unsuccessful trajectories separate, and must charge history
lookup, verification and cache alignment to actual latency. Acceptance cannot
be inferred from repeated-span coverage alone.

Next: expand the fixed sample to 30 tasks, check whether repair prompts and the
small model provide enough useful repair trajectories, then compare ordinary
decode with a cheaper historical draft under identical agent settings.
The online controller remains unimplemented.

## Reproduce and Inspect

```bash
curl -L --fail https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json -o /private/tmp/siliconllm-sanitized-mbpp.json
pdm run python benches/agent_code_repair_pilot.py --data /private/tmp/siliconllm-sanitized-mbpp.json --num-tasks 5 --seed 42 --max-repairs 2 --max-tokens 384 --output-dir benches/agent_pilot_new_run
pdm run pytest tests/test_agent_pilot.py -q
```

Run 2 is the current reference pilot: `benches/agent_pilot_20261015_run2/`.
It contains `metadata.json`, `tasks.json`, `turns.jsonl`, `outcomes.json` and
`summary.json`. Metadata records dataset SHA256, model revision, and script
hashes. Selected task files preserve the exact inputs without reference code.
Run 1 remains in `benches/agent_pilot_20261015_run1/`. The earlier directory
`benches/agent_pilot_20261015/` is an aborted tokenizer preflight, not a completed
run. The current script reproduces the Run 2 feedback behavior.

Validation: five focused harness tests pass, including visible/held-out prompt
separation, code extraction/policy, failure feedback, timeout, repeated-span
boundaries, and filesystem restrictions. The nine Run 2 traces were checked
for consistent token counts, EOS stopping and exact duplicate repairs.

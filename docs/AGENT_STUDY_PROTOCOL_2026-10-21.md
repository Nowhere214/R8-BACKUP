# Agent Decoding Study Protocol

> SUPERSEDED FOR CURRENT REPORTING on October 21, 2026. This is the historical
> procedure for the withdrawn local agent study, not validation of the new
> external results. Keep it for investigation; do not automatically rerun it.
> Current status: `EXPERIMENT_STATUS_2026-10-21.md`.

Experiment start: 2026-10-21. Reporting checkpoint: 2026-10-22.

Current status: the original 90-task comparison completed **one screening
repetition**, not three. A revised, repair-only comparison uses a disjoint
30-task cohort and three completed repetitions. See the final section before
interpreting the original design below.

## Research Question

Can historical token proposals remove the small-model drafting bottleneck, and
can a measured-cost controller improve latency without changing the SiliconLLM
target's greedy sequence?

## Frozen Inputs

The canonical run directory is `benches/agent_study_20261021_main/`.
`tasks.json` fixes ten calibration tasks, sixty MBPP sanitized evaluation
tasks, and thirty HumanEval evaluation tasks before their model inference.
The original five-task pilot is excluded from evaluation. Calibration uses
the official MBPP validation ID range, disjoint from its test range.

HumanEval inclusion requires a documented example convertible to an executable
equality check. Of 164 tasks, 110 meet this parser criterion. Thirty are sampled
with a fixed seed. This is a restricted workload study, not the full HumanEval
benchmark or an MBPP+ result. Reference solutions are omitted from saved tasks
and prompts. Data sources and SHA256 hashes are in `protocol.json`.

The agent uses a single visible example, at most two repairs, greedy decoding,
and a 384-token per-call limit. MBPP retains two tests for final evaluation;
HumanEval retains the official test function. Final test results never enter
the repair prompts. Identical prompts and stopping rules apply to all modes.

## Methods

- Ordinary SiliconLLM Stage 2 KV-cache decoding.
- Qwen2 0.5B drafting for the 1.5B target, fixed k=2.
- N-gram proposals from the full current prompt and the accepted current output.
- N-gram proposals from earlier code in the same task and accepted current output.
- Historical proposals with an online measured-cost controller.
- Historical proposals with a simple acceptance-rate rule.
- The installed same-checkpoint mlx-lm model/cache as a runtime reference.

Historical retrieval is a conventional n-gram mechanism. The code-only scope
is a scoped adaptation, not a reproduction of the full AgentSpec algorithm.
It does not use a batch scheduler, cross-query state, or training.

The history index uses n=4 and up to 32 recent matching positions. Historical
methods use a maximum proposal length of 16. The acceptance-rule comparator
uses k<=8 while its EMA acceptance is at least 0.6. The cost controller chooses
from k=0/2/4/8/16, estimating expected progress as
`1 + sum(p**j for j in range(1, k+1))`. It requires a predicted 5% margin over
ordinary decode. The acceptance-only comparator starts with p=0.8. The guarded
cost controller uses an independent validation estimate (described below).
Both update their acceptance estimate with weight 0.3.
The acceptance-independence assumption is an approximation to be assessed,
not an established performance guarantee.

Median ordinary and batched-forward costs come from ten calibration prompts,
five timed probes per shape after a warmup. These costs are frozen before
evaluation. Controller state resets per request. Historical state is cold
between tasks. All lookup, index building, decision and cache alignment time
is charged to the measured model call.

## Comparisons

Fixed-input replay holds prompts, earlier code and output limits identical.
Every method receives only the history visible at that request. Each replay
checks its token sequence against the original SiliconLLM baseline. Requests
and methods have deterministic shuffled execution order for each repetition.

Closed-loop evaluation runs actual model, visible test and repair interactions.
It checks output sequences, number of calls and final task outcomes against
the baseline trajectories. Model loading, warmup and held-out grading are
reported separately from agent execution time. Test execution uses the
existing constrained subprocess runner.

Primary speedup is paired baseline wall time divided by method wall time.
Aggregate repeated timings within each request, then sum by task. Bootstrap
95% intervals resample tasks, not tokens or repetitions. Report generation
and repair, successful and failed tasks, and both datasets separately.
Timing gains with output mismatches do not meet the preservation objective.
mlx-lm differences are recorded separately rather than treated as equivalence
with the custom target implementation.

## Correctness Gates and Limits

The old nine pilot prompts match native mlx-lm over at most 96 generated tokens.
The ten calibration prompts match in nine cases over at most 64 tokens; MBPP
574 diverges at token index 38. The formal preservation reference is therefore
ordinary decoding of the exact same SiliconLLM target, not the native engine.

Unit tests cover candidate acceptance, rejection, cache alignment, EOS, output
limits, no history across task boundaries, and ordinary fallback. Real-model
smoke tests precede full replay. Any numerical divergence must be reported and
investigated before claiming exact preservation.

Initial working directories ending in `_v2` and `_final`, plus the original
date-only directory, are preparation/calibration artifacts. The canonical main
manifest corrects docstring example parsing before evaluation inference.
No evaluation outcome informed task selection or controller thresholds.

## Guarded Revision After the First Correctness Failure

The initial full replay was stopped after HumanEval 141, generation attempt 0,
failed exact-token comparison in acceptance-only mode. The partial records in
`replay/` are an aborted diagnostic run, not a completed latency result. The
failure did inform the cache-recovery implementation; this task is therefore
a regression case, not an unseen correctness test for the revised decoder.

A focused reproduction found that suffix trimming after a rejected batched
verification left retained KV values different from ordinary single-token
execution. The largest observed absolute difference was 0.013671875 in keys
and 0.0135498046875 in values. Restoring the pre-verification snapshot and
replaying the retained prefix through ordinary forwards reduced these errors
to zero. All five methods then matched the original sequence on this request.
This is a finite-precision implementation issue, not a failure of speculative
decoding's exact-arithmetic argument.

The revised decoder restores and replays on every rejection. Recovery forwards,
cache handling and Python overhead remain inside wall-clock measurements and
target-call counts. The cost controller additionally estimates expected
recovery work as `sum((j+1) * p**j * (1-p) for j in range(k))` ordinary forwards.
Full-acceptance steps still use their batched cache; exact greedy equivalence
must therefore be checked empirically, not asserted as a numerical guarantee.

During the revised full replay, fixed k=16 methods also diverged on MBPP 260
(code and prompt history, first generation) and HumanEval 114 (prompt history,
first generation). In these traces, fully accepted batches preceded the later
difference. Rejection recovery does not erase state differences introduced by
earlier full acceptance. These are recorded as failed preservation cases, not
silently removed or repaired using the recorded target output. Their wall-time
ratios are descriptive only and cannot support an equivalent-output speedup
claim. The final counts must come from completed repetitions.

The ten separate MBPP validation tasks supplied 20 agent requests for guarded
policy calibration. All 20 matched ordinary decoding. Historical proposals
accepted 539 of 629 tokens, giving the smoothed prior `(539+1)/(629+2)`, or
0.8557844691. Costs and this prior are frozen in `calibration_guarded.json`.
No evaluation task was used to fit that prior or the 5% switching margin.

The revised full evaluation is named `guarded_replay/`. It reruns all 90 tasks,
178 fixed requests, five methods and three repetitions. Its conclusions must
be based on its own completed records, not mixed with unguarded smoke timings.

## Secondary Studies

Secondary subsets use the first ten evaluation tasks from each dataset in the
frozen manifest, with no selection by success or speedup. The small-draft and
native-model reference comparison uses fixed original SiliconLLM prompts.
Native-model token differences in that comparison are descriptive, not a
correctness failure of the custom target.

The real closed-loop comparison uses ordinary, prompt-history and cost-controlled
decoding, three repetitions per task. It measures model calls, visible-test
execution and framework time. Loading, warmup and final held-out grading are
outside agent wall time. Output sequences, call counts and final outcomes must
be compared, not just pass rates.

The native extension independently calibrates and collects trajectories from
the native mlx-lm model and KV cache. It then compares the same synchronous
decoding loop with and without historical proposals. This isolates the model
and cache implementation effect. It is not a measurement of the complete
mlx-lm `generate_step` streaming API, whose prefill and scheduling differ.
It must not be advertised as beating the production library's best throughput.

The primary ratios compare each method with ordinary decode. A separate
`controller_comparisons.json` uses fixed code-history decoding as the reference
for both controller variants. This holds the historical corpus constant.
The following ablation checks four fixed lengths, not every possible length.

An additional ablation now specifies k=2/4/8/16 before its inference. It was
added after the feedback study started, so it is a follow-up, not an original
preregistered comparison. Three validation repetitions choose the fastest
fully exact fixed length. A separate three-repeat replay on the third public
cohort compares ordinary, all four fixed lengths and unchanged feedback
control. It has its own paired timings and must not mix times from earlier
runs. All lengths and mismatches remain reported, not only the selected winner.
This reuses the third cohort; it does not add twenty more independent tasks.
It completed 300/300 exact validation decodes and 720/720 exact evaluation
decodes. Validation selected k=16. Its comparison with unchanged feedback
control did not establish an extra controller benefit. Runner:
`benches/run_fixed_length_ablation.py`.

## Measurement Environment

macOS 14.8.5 arm64, Python 3.12.13, MLX 0.29.1, mlx-lm 0.27.1,
Transformers 5.6.0 and NumPy 2.2.6. The machine reports 18 GiB memory.
AC power was connected during the primary run. Other applications are not
fully controlled, so randomized paired repetitions reduce but do not eliminate
timing noise. Confidence intervals describe task variation under this setup,
not hardware-to-hardware generalization or a guarantee for future tasks.

The cached Qwen2-1.5B checkpoint is 4-bit quantized, group size 64, revision
`42d944c83758c69dd75180f41d185ad704e7c3ad`. FP16 scale tensors and norm weights
were verified from the safetensors header. Custom projection/MLP operations
remain quantized; embedding and the tied dense head are dequantized to FP16,
and attention uses FP32 internally. The native model uses its own quantized
embedding/head and attention path. This is not a full-FP16 weight experiment.

## Fallback-Feedback Revision

The repair-only replay and closed-loop study completed before this change.
Their original source snapshots are retained. Analysis found that speculative
acceptance updates alone can lock the cost controller into ordinary decoding.
The revised mode observes whether an already retrieved first candidate matches
the ordinary next token whenever the controller selects k=0. It applies the
same EMA update, with no extra target forward or additional retrieval. CPU
update and event logging time remain inside measured wall time. This one-token
signal is a proxy for multi-token acceptance, not an unbiased estimate of it.

`benches/agent_feedback_study_20261021/` fixes ten new MBPP and ten new HumanEval
tasks, seeds 20261024, excluding all 120 earlier evaluation tasks and five pilot
tasks. The variant and manifest precede inference on this cohort. Calibration
costs, prior and margin remain unchanged. Three paired repetitions compare
ordinary, fixed history, original cost control and feedback control. Actual
closed-loop checks use the first five tasks per dataset, three repetitions,
ordinary versus feedback control. A known HumanEval 17 regression and the
original validation tasks are development checks, not unseen evaluation.

Ten separately authored controlled edits require the supplied file to pass its
old contract but fail a new visible test. Final grading checks both contracts
and additional held-out cases. Five methods, including feedback control, use
three paired replay repetitions of the resulting ordinary trajectories. The
tasks test sensitivity to meaningful edits; they are not a public benchmark
or a substitute for larger real repository tasks.

## Repair-Only Revision and Fresh Evaluation

The original cost controller also diverged on HumanEval 87, first generation,
at token index 304. Completing all three repetitions of invalid configurations
would not establish the required preservation. The run stopped after its first
complete round. All 890 first-round records, including the four failed runs,
are in `main/screening_replay/`. The interrupted raw source remains unchanged
in `main/guarded_replay/`; its few second-round records are not mixed into the
completed screening summary.

Screening exact counts were ordinary 178/178, prompt history 176/178, code
history 177/178, cost controller 177/178 and acceptance rule 178/178. Thus the
cost controller's aggregate timing ratio from this round is not a valid
equivalent-output acceleration claim. The acceptance-only result is one
screening repetition, not a three-repeat confirmation.

The MBPP 260 full-acceptance audit independently compared a batch of 16
accepted candidates plus the bonus prediction with 17 ordinary forwards.
Their local predictions agreed, but retained keys differed by up to 0.125 and
values by 0.033203125. Restoring and ordinary replay removed the difference.
This shows why rejection-only recovery cannot guarantee future equivalence.

The next design restricts speculation to calls with prior generated code.
First generation runs the actual ordinary path, without a history index.
Repairs compare the same corpus under fixed k<=16, acceptance-only k<=8 and
the cost controller's k=0/2/4/8/16. This is a phase restriction, not a general
numerical correction, and still needs empirical output checks.

`benches/agent_study_20261021_holdout/` fixes fifteen MBPP and fifteen HumanEval
tasks, excluding all ninety screening tasks and the five old pilot tasks.
Both selection seeds are 20261023. The manifest and repair-only mode were
created before inference on this cohort. Calibration reuses the original ten
separate validation tasks, not new evaluation outcomes. Repair-only calibration
had 522/524 accepted proposals. Its smoothed prior is capped at 0.99, using the
same upper cap as the earlier controller, before optimized replay starts.

The new ordinary baseline has completed: 30 tasks, 64 model calls, 34 repair
calls, 33 duplicate repairs, three length stops and 12 final passes. These are
separate workload statistics, not evidence of an optimization gain.

The suite first checks all revised methods on the separate validation prompts,
then runs four methods and three repetitions on all 64 new requests. Follow-ups
use the first ten new tasks per dataset for actual closed-loop execution and
the native-model extension. A one-repeat small-draft/native comparison on the
old cohort is a secondary diagnostic only. These studies are serialized.

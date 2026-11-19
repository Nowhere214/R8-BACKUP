# Method Scope and Related Work

> STATUS UPDATE, October 21, 2026: empirical claims below that depend on the
> earlier local agent experiments are withdrawn. Related-work references remain
> reference material, not proof of novelty. The external replacement report
> requires reconciliation. See `EXPERIMENT_STATUS_2026-10-21.md`.

Checked against primary sources on October 21, 2026.

## What This Study Tests

SiliconLLM is the implementation platform. Its standard model implementation
and KV cache are foundations, not new algorithms. The extension tests whether
cheap historical drafts can avoid the earlier small-model drafting bottleneck
on a local Apple Silicon device.

The current index is a conventional four-token n-gram lookup. Restricting its
source to earlier code is a controlled scope ablation, not a reproduction of
an existing complete agent acceleration system.

## Closest Work

- [The N-Grammys (2024)](https://proceedings.mlr.press/v262/stewart24a.html)
  studies learning-free, low-cost n-gram drafting. N-gram drafting is not new.
- [AgentSpec (August 2026)](https://arxiv.org/abs/2608.24004) uses
  structure-isolated drafting and redundancy-aware allocation for batched
  agent inference in vLLM. Our single-request experiment neither implements
  its batch scheduler nor measures its server-level throughput objective.
- [AgSpec (October 2026)](https://arxiv.org/abs/2610.01108) combines session,
  workspace and global retrieval with agent-aware adaptive draft length.
  Historical reuse and adaptive length are therefore not novel claims here.

## Candidate Contribution, Not a Proven Novelty Claim

The local study explicitly charges retrieval, decision, verification and
finite-precision cache recovery. Its controller compares predicted time per
emitted token, including ordinary-prefix replay after rejection, with ordinary
decoding. It can choose not to speculate. Measurements use one device, one
1.5B target and a restricted code-repair workload.

The relevant controller ablation is **code history with fixed length versus
the same code history with cost control**. Comparing only against ordinary
decoding would confound drafting and control. Prompt-history drafting uses a
different corpus and is a competing method, not an isolated controller ablation.

A convincing result needs both preserved outputs and measured latency gains.
It should also establish whether gains survive a faster native inference path,
and whether they depend mainly on unsuccessful agents repeating identical code.
The completed native extension uses its own greedy target as reference; native
and SiliconLLM sequences cannot be assumed numerically identical. Fixed history
reached 1.336x on that path, but this uses the shared synchronous loop, not the
complete production generation API.

If the controller does not improve on the same-source fixed policy, report
that result and refine the controller. If it only accelerates the custom engine
but not the native path, describe an implementation-level gain, not a practical
improvement over the faster engine. Neither outcome proves publication-level
novelty or guarantees a thesis outcome.

## Current Evidence

The original controller did not beat fixed history on the 30-task custom or
20-task native comparisons. The fallback-feedback revision repairs a concrete
failure to resume speculation, with no extra target forward. On another 20
public tasks it reached 1.321x versus ordinary decoding, below fixed history's
1.489x. Its improvement over the old controller was not clearly separated
from no gain by the task-bootstrap interval.

Seven of ten authored edits succeeded, but the successful-task speedup was
uncertain. The defensible claim is a measured, scoped benefit of history reuse,
not a superior dynamic algorithm. The additional fixed-length ablation selected
k=16 on validation and found 1.451x versus feedback control's 1.312x on the
same evaluation cohort in a new timing block. It did not establish an extra
controller benefit and does not create an originality claim.

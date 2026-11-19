# 01_cover

English:
Good afternoon. Today I will report the current progress of SiliconLLM.
The topic is efficient LLM inference on Apple Silicon.
Since the last major talk, I moved from finding a decode gap to explaining it.
I also built a self Stage 1 baseline and checked it against the reference.
The main line today is: measure first, explain second, validate third, and optimize fourth.

中文：
大家下午好。今天我汇报 SiliconLLM 目前的进度。
主题是在 Apple Silicon 上做高效的 LLM 推理。
上一次大组会之后，我从发现 decode gap，推进到了解释这个 gap。
同时我实现了自己的 Stage 1 baseline，并和 reference baseline 做了对齐检查。
今天的主线是：先测量，再解释，再验证 baseline，最后再做优化。

# 02_what_changed

English:
This slide shows what changed after the 5/20 talk.
At that time, the main result was that decode was much slower in the tutorial baseline.
This time, I added more evidence around that result.
The work now has three parts: Phase 0 diagnosis, component profiling, and Phase 1 baseline validation.
So the project is no longer just one benchmark plot.

中文：
这一页说明 5 月 20 日之后发生了什么变化。
当时主要结果是 tutorial baseline 的 decode 明显更慢。
这次我围绕这个结果补了更多证据。
现在工作分成三块：Phase 0 诊断、组件级 profiling、Phase 1 baseline 验证。
所以现在它不是单独一张 benchmark 图，而是一套更完整的证据链。

# 03_last_zemi_finding

English:
Let me first recall the previous message.
The largest gap was not in prefill.
It appeared mainly during autoregressive decode.
The gap was around 29 to 33 times in the previous report.
That result decided the direction of the next experiments.

中文：
先回顾一下上一次大组会的结论。
最大的差距并不是在 prefill 阶段。
它主要出现在 autoregressive decode 阶段。
上一次报告里的差距大约是 29 到 33 倍。
这个结果决定了后续实验应该集中在 decode 上。

# 04_phase0_package

English:
Phase 0 is now a reproducible diagnosis.
It includes the Stage 1 reference, mlx-lm, and llama.cpp Metal.
It also includes workload scans, KV-cache microbenchmarks, a cost model, and caveats.
The important point is that every result has a source table.
This makes the next optimization step easier to defend.

中文：
现在 Phase 0 已经整理成了可复现的诊断结果。
里面包括 Stage 1 reference、mlx-lm、以及 llama.cpp Metal。
同时也包括 workload scan、KV-cache microbenchmark、cost model 和 caveat。
重点是，每个结果都有对应的来源表格。
这样后面做优化时，依据会更清楚。

# 05_three_engine_decode

English:
Here I added a third engine, llama.cpp.
This is useful because it is not the same code path as mlx-lm or the tutorial baseline.
The result supports the same story.
Stage 1 is slow, mlx-lm is highly optimized, and llama.cpp gives another systems reference point.
This reduces the risk that the conclusion only comes from one framework.

中文：
这里我加入了第三个引擎，也就是 llama.cpp。
它有用的地方在于，它不是 mlx-lm，也不是 tutorial baseline 的同一条代码路径。
结果仍然支持同一个故事。
Stage 1 很慢，mlx-lm 高度优化，而 llama.cpp 提供了另一个系统实现的参考点。
这降低了结论只来自某一个框架的风险。

# 06_longer_prompt_larger_gap

English:
The next question is whether the gap is fixed overhead.
The prompt-length scan suggests it is not only fixed overhead.
When the prompt gets longer, the gap becomes larger.
It grows from about 14 to 15 times to about 58 times.
This points to full-context decode work as a key reason.

中文：
接下来的问题是，这个 gap 是不是只是固定开销。
prompt-length scan 表明它不只是固定开销。
当 prompt 变长时，差距也会变大。
它从大约 14 到 15 倍，增长到大约 58 倍。
这说明 full-context decode 的重复计算是关键原因之一。

# 07_prompt_length_scan

English:
This figure gives the detailed prompt-length scan.
The main thing to read is the trend, not one single point.
The Stage 1 gap grows as context length grows.
That is exactly what we would expect if each decode step keeps paying for old context.
This is why KV-cache becomes the natural next target.

中文：
这张图给出了更细的 prompt-length scan。
这里主要看趋势，而不是某一个单点。
随着 context length 增加，Stage 1 的 gap 也在增加。
如果每一步 decode 都在为旧 context 重新付出代价，这个趋势是合理的。
所以 KV-cache 就成为下一个自然目标。

# 08_cost_model_reading

English:
This slide gives a simple cost-model reading.
For Stage 1, the slope is clearly measurable.
For mlx-lm, the slope is almost flat in this view.
This means Stage 1 has a per-context-token cost during decode.
I should note that prompt-1024 projection is not a direct measurement.

中文：
这一页是一个简单的 cost-model 读法。
对 Stage 1 来说，slope 是明显可测的。
对 mlx-lm 来说，在这个读法下 slope 基本接近平。
这说明 Stage 1 在 decode 时存在 per-context-token 的代价。
需要说明的是，prompt-1024 的 projection 不是直接测量值。

# 09_stage1_decode_profile

English:
Then I used profiling to split one decode step.
The result is important for the research story.
The Python loop is only a very small part.
Most time is in model work, especially MLP, LM head, and attention.
So the problem is not just a simple Python overhead issue.

中文：
然后我用 profiling 拆分了一步 decode 的时间。
这个结果对研究故事很重要。
Python loop 只占非常小的一部分。
主要时间在模型计算里，尤其是 MLP、LM head 和 attention。
所以这个问题不是简单的 Python overhead 问题。

# 10_before_optimization

English:
This is why I did not jump directly to speedup claims.
First, I need to diagnose the slow path.
Second, I need a faithful implementation of the baseline.
Third, I need to check that my implementation matches the reference.
Only after that can I change one optimization lever and measure the effect.

中文：
这就是为什么我没有直接跳到 speedup claim。
第一步需要诊断慢在哪里。
第二步需要实现一个忠实的 baseline。
第三步需要检查自己的实现是否和 reference 一致。
只有在这之后，才适合改变一个优化变量，并测量它带来的影响。

# 11_phase1_implementation_status

English:
Phase 1 is the self Stage 1 implementation.
The implemented parts include attention, linear layers, RoPE, GQA, RMSNorm, MLP, Qwen2 blocks, and generation.
Targeted tests pass for the core path and the Qwen2 path.
Some model-dependent tests still depend on local model availability.
For this report, the important point is that the self baseline is testable.

中文：
Phase 1 是我自己的 Stage 1 实现。
已经实现的部分包括 attention、linear layer、RoPE、GQA、RMSNorm、MLP、Qwen2 block 和 generation。
核心路径和 Qwen2 路径的 targeted tests 已经通过。
有些依赖模型的测试仍然取决于本地模型是否存在。
对这次汇报来说，重点是这个 self baseline 已经可以被测试。

# 12_self_reference_parity

English:
This benchmark checks whether the self baseline matches the reference baseline.
For the 0.5B model, the decode throughput ratio is about 1.004.
For the 1.5B smoke test, the ratio is about 1.001.
So the self implementation is close enough for the next controlled experiment.
This is a parity result, not a speedup result.

中文：
这个 benchmark 检查 self baseline 是否和 reference baseline 对齐。
在 0.5B 模型上，decode throughput ratio 大约是 1.004。
在 1.5B smoke test 上，ratio 大约是 1.001。
所以这个 self implementation 已经足够接近，可以进入下一步受控实验。
但这是 parity 结果，不是 speedup 结果。

# 13_claims_now

English:
At this point, I think there are three defensible claims.
First, the decode gap has been diagnosed more clearly.
Second, the cost is linked to repeated model work in full-context decode.
Third, the self Stage 1 baseline is credible.
What I cannot claim yet is a faster custom engine.

中文：
到目前为止，我认为有三个比较稳的 claim。
第一，decode gap 已经被更清楚地诊断出来。
第二，代价和 full-context decode 中重复模型计算有关。
第三，self Stage 1 baseline 是可信的。
但我现在还不能声称已经做出了更快的自定义引擎。

# 14_evidence_package

English:
This slide summarizes where the project stands now.
Phase 0 gives the diagnosis.
Phase 1 gives a trusted self baseline.
So the next step is no longer open-ended.
The project is ready to enter Stage 2 KV-cache work.

中文：
这一页总结目前项目走到哪里。
Phase 0 给出了诊断。
Phase 1 给出了可信的 self baseline。
所以接下来不是开放式地随便优化。
项目已经准备进入 Stage 2 KV-cache。

# 15_stage2_kv_cache_next

English:
The next step should be Stage 2 KV-cache.
I should first make it correct.
Then I should benchmark Stage 1 and Stage 2 under the same workloads.
After that, I should profile again and see which cost moved.
Only after this step should I discuss quantized matmul or Flash Attention style kernels.

中文：
下一步应该是 Stage 2 KV-cache。
首先要把它做正确。
然后在相同 workload 下比较 Stage 1 和 Stage 2。
之后再重新 profile，看哪些成本真正发生了变化。
只有完成这一步之后，才适合讨论 quantized matmul 或类似 Flash Attention 的 kernel。

# 16_takeaway

English:
To close, the main message is simple.
Measure first.
Explain second.
Validate third.
Optimize fourth.
This is the path I will follow for the next stage of the project.

中文：
最后总结一下，核心信息很简单。
先测量。
再解释。
再验证。
最后优化。
接下来这个项目会按照这条路径继续推进。

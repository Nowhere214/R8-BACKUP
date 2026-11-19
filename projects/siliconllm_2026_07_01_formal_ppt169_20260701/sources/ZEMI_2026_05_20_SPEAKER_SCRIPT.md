# Speaker Script for ZEMI 2026-05-20

Target duration: about 15 to 18 minutes at normal speaking speed.

## English Script

### Slide 1: Efficient LLM Inference on Apple Silicon

Good morning. Today I will present my current research project, SiliconLLM. The topic is efficient large language model inference on Apple Silicon.

This talk is about the first stage of the project, which I call Phase 0. In this stage, I do not try to build a new optimization immediately. Instead, I first measure the performance gap between a simple reference implementation and a production inference engine. The goal is to understand where the bottleneck is before deciding what to optimize.

### Slide 2: Outline

The talk has four parts.

First, I will introduce the motivation and the research problem. Second, I will explain the benchmark method and the experimental setup. Third, I will show the current Phase 0 results. Finally, I will discuss what these results mean and what I plan to do next.

The main message is simple: on Apple Silicon, the largest performance gap appears in the decode stage, not in the prefill stage.

### Slide 3: Introduction: Why This Research?

Large language models are now used in many applications, but inference is still expensive. After a model is trained, every user request still needs computation. If the system has many users, inference can become the main cost.

Local inference is also becoming more important. Running models on a personal machine can reduce network latency, improve privacy, and make AI systems more accessible. Apple Silicon is a common consumer platform, especially for students and developers. However, most performance discussions still focus on NVIDIA GPUs. This is why I want to study LLM inference on Apple Silicon.

### Slide 4: Research Problem

Production engines such as mlx-lm can run models very fast. But they are also highly optimized systems. Many techniques are used together, such as KV cache, quantization, kernel fusion, and optimized attention.

Because of this, a single speed number is not enough. If mlx-lm is 30 times faster than a simple implementation, I want to know where this 30 times comes from. Is it from matrix multiplication? Is it from memory usage? Is it from the decode loop? Or is it from several factors together?

This is the main research problem: open the black box and locate the performance gap.

### Slide 5: Project Framing: SiliconLLM

SiliconLLM is my personal research branch based on the open-source tiny-llm teaching project. I use tiny-llm as a controlled experimental platform.

This distinction is important. The base code and teaching structure come from the open-source project. My work is to use it for measurement, comparison, and later optimization experiments on Apple Silicon.

This platform is useful because it is much smaller and easier to inspect than a full production serving system. It allows me to compare a simple, readable implementation with production engines under the same workload.

### Slide 6: Research Questions

There are three research questions.

The first question is: where is the performance gap? I want to separate prefill throughput, decode throughput, wall time, and memory usage.

The second question is: which optimization gives the largest benefit? Later, I will look at KV cache, quantized matrix multiplication, Flash Attention, batching, and chunked prefill.

The third question is more hardware-related. I want to understand whether the optimization behavior on Apple Silicon is similar to the behavior reported for CUDA GPUs, or whether the physical bottleneck is different.

### Slide 7: Method: Experimental Setup

The experiments are run on my MacBook Pro with Apple Silicon and 18 GB unified memory.

I use the Qwen2 model family, including 0.5B, 1.5B, and 7B models. These sizes are useful because they show how the system behaves as the model becomes larger.

The two engines in the current benchmark are SiliconLLM Stage 1 reference implementation and mlx-lm. The Stage 1 reference implementation is simple and mostly unoptimized. mlx-lm is a production-level inference engine for Apple Silicon.

### Slide 8: Method: Benchmark Design

The benchmark is designed as a controlled comparison. I keep the hardware, model family, prompt, and generation workload fixed as much as possible.

For each configuration, I run the benchmark three times in the same session. I report the mean and the standard deviation. This is important because one run can be misleading, especially when the code path has high overhead.

The metrics are prefill throughput, decode throughput, end-to-end wall time, and peak memory usage. Together, these metrics give a more complete view than one single tokens-per-second number.

### Slide 9: Method: Why Separate Prefill and Decode?

In LLM inference, prefill and decode are very different.

Prefill processes the input prompt. This part can use parallel computation over many tokens. It is often closer to large matrix operations.

Decode is different. It generates one token at a time. Each step depends on the previous step. This makes decode very sensitive to KV cache, attention recomputation, Python overhead, and kernel launch overhead.

So if we only report total speed, we may hide the real bottleneck. That is why I separate prefill and decode.

### Slide 10: Result: Overall Throughput

This slide shows the overall throughput result.

The blue line is mlx-lm, and the red line is the SiliconLLM Stage 1 reference implementation. The left panel is prefill throughput, and the right panel is decode throughput. Both axes use a log scale.

The first observation is that mlx-lm is faster in both stages. But the difference is not the same. In prefill, the gap is visible but moderate. In decode, the gap is much larger. This suggests that the main bottleneck of the simple implementation is in autoregressive generation.

### Slide 11: Result: The Gap Is Concentrated in Decode

This slide makes the gap more explicit.

For prefill, mlx-lm is only about 1.3 to 1.6 times faster than the Stage 1 reference implementation. But for decode, the speedup is about 29 to 33 times.

This is the most important result so far. It means the performance problem is not evenly distributed across the pipeline. The simple implementation is not just slightly slower everywhere. It is especially weak in the decode stage.

This also gives a clear direction for later optimization work.

### Slide 12: Result: End-to-End Wall Time

Throughput numbers are useful, but wall time is easier to understand from a user perspective.

For the same benchmark prompt, the 0.5B Stage 1 reference run takes about 66.8 seconds, while mlx-lm takes about 2.5 seconds. For 1.5B, the Stage 1 reference run takes about 172.7 seconds, while mlx-lm takes about 5.9 seconds.

So the decode gap is not only a technical metric. It directly appears as user-visible latency. This is important for any interactive LLM application.

### Slide 13: Result: Memory and Feasibility Boundary

Memory is another important result.

The Stage 1 reference implementation uses about 1.9 GB peak memory for 0.5B and about 4.7 GB for 1.5B. For mlx-lm, the 7B model can run with about 4.8 GB peak memory in this benchmark.

This shows that optimization affects not only speed, but also feasibility. On an 18 GB machine, an unoptimized path can become unsafe or impractical for larger models. A production engine can make the same model size much more manageable.

### Slide 14: Result: Repeatability Check

This slide is about measurement reliability.

Earlier in the project, I had some N=1 measurements. Later, when I repeated the benchmark with N=3, one of the earlier trends changed. The old single-run result suggested a different prefill trend, but the repeated result showed that it was likely noise.

This is an important lesson for the project. For high-overhead inference paths, one measurement is not enough. The benchmark method itself is part of the research contribution.

### Slide 15: Result: 7B Boundary for Stage 1 Reference

The missing 7B datapoint for the Stage 1 reference is also meaningful.

I do not report it as a measured result because the estimated memory usage is close to the 18 GB machine limit. Based on the 0.5B and 1.5B measurements, extrapolation suggests that decode would be very slow, around one token per second, and memory would be near the safe limit.

So this is not just a missing experiment. It shows a feasibility boundary. Before running larger models with the reference path, optimization is necessary.

### Slide 16: Discussion: What Explains the Gap?

The current data tells us where the gap appears. It does not yet give the exact percentage for each cause.

However, there are several likely reasons. The Stage 1 reference path does not have a full KV cache optimization. It may recompute attention work during decode. It also lacks the same level of optimized quantized matrix multiplication and kernel fusion as mlx-lm. Python-level overhead can also be important when generation happens token by token.

The next step is to use profiling to break down this 30 times decode gap more carefully.

### Slide 17: Discussion: Next Steps

There are three immediate next steps.

First, I will add llama.cpp Metal as a third production baseline. This will make the comparison stronger because it adds another mature engine.

Second, I will run workload-length scans. I want to test how prompt length and generated length affect the gap.

Third, I will use the MLX profiler to locate which kernels and code paths dominate decode time.

After that, I will start implementing the Stage 1 components in my own source directory, and later add KV cache, quantization, and attention optimizations one by one.

### Slide 18: Thank You

To summarize, this project studies LLM inference on Apple Silicon using a measurement-first approach.

The current Phase 0 result shows that the largest gap between a simple implementation and mlx-lm is in decode. Prefill differs by only about 1.3 to 1.6 times, but decode differs by about 29 to 33 times.

The results also show that memory limits are part of the problem on consumer hardware. My next step is to add more baselines and use profiling to explain the decode bottleneck in more detail.

Thank you for listening. I am happy to take questions.

## 中文翻译

### 第 1 页：Efficient LLM Inference on Apple Silicon

大家早上好。今天我会介绍我目前的研究项目 SiliconLLM。这个项目的主题是 Apple Silicon 上的大语言模型高效推理。

这次报告主要介绍项目的第一阶段，也就是 Phase 0。在这个阶段，我并不是马上实现新的优化方法，而是先测量一个简单参考实现和生产级推理引擎之间的性能差距。目标是在决定优化什么之前，先理解瓶颈在哪里。

### 第 2 页：Outline

这次报告分成四个部分。

第一部分是研究动机和研究问题。第二部分是 benchmark 方法和实验设置。第三部分是目前 Phase 0 的实验结果。最后一部分是对结果的讨论，以及接下来的计划。

这次报告的核心信息很简单：在 Apple Silicon 上，最大的性能差距主要出现在 decode 阶段，而不是 prefill 阶段。

### 第 3 页：Introduction: Why This Research?

大语言模型现在已经被用于很多应用，但推理成本仍然很高。模型训练完成之后，每一次用户请求仍然需要计算。如果系统有很多用户，推理就会成为主要成本之一。

本地推理也变得越来越重要。在个人设备上运行模型，可以降低网络延迟，提高隐私性，也让 AI 系统更容易被普通用户和开发者使用。Apple Silicon 是一个很常见的消费级平台，尤其是在学生和开发者中很常见。但目前很多性能讨论仍然主要集中在 NVIDIA GPU 上。因此，我希望研究 Apple Silicon 上的 LLM 推理问题。

### 第 4 页：Research Problem

mlx-lm 这样的生产级引擎可以很快地运行模型。但它们也是高度优化过的系统，里面同时使用了很多技术，例如 KV cache、量化、kernel fusion 和优化后的 attention。

因此，一个单独的速度数字并不够。如果 mlx-lm 比一个简单实现快 30 倍，我想知道这 30 倍到底来自哪里。是矩阵乘法更快？是内存使用更好？是 decode loop 更高效？还是多个因素共同造成的？

这就是主要研究问题：打开这个黑盒，定位性能差距的来源。

### 第 5 页：Project Framing: SiliconLLM

SiliconLLM 是我基于开源教学项目 tiny-llm 做的个人研究分支。我把 tiny-llm 作为一个可控的实验平台。

这个区分很重要。基础代码和教学结构来自开源项目。我的工作是在它的基础上做测量、对比，以及后续针对 Apple Silicon 的优化实验。

这个平台的优势是，它比完整的生产级 serving 系统小得多，也更容易检查。因此，我可以在同样的 workload 下，对比一个简单、可读的实现和生产级引擎。

### 第 6 页：Research Questions

这个项目有三个研究问题。

第一个问题是：性能差距在哪里？我希望分别观察 prefill throughput、decode throughput、wall time 和 memory usage。

第二个问题是：哪一种优化带来的收益最大？后面我会分析 KV cache、量化矩阵乘法、Flash Attention、batching 和 chunked prefill。

第三个问题更偏硬件。我想理解 Apple Silicon 上的优化行为，是否和 CUDA GPU 上的论文报告一致，还是说物理瓶颈有所不同。

### 第 7 页：Method: Experimental Setup

实验运行在我的 MacBook Pro 上，硬件是 Apple Silicon，并且有 18 GB 统一内存。

模型使用 Qwen2 系列，包括 0.5B、1.5B 和 7B。选择这些模型规模，是为了观察模型变大时系统行为如何变化。

当前 benchmark 中的两个引擎是 SiliconLLM Stage 1 reference implementation 和 mlx-lm。Stage 1 reference implementation 比较简单，基本没有生产级优化。mlx-lm 则是面向 Apple Silicon 的生产级推理引擎。

### 第 8 页：Method: Benchmark Design

benchmark 的设计目标是受控对比。我尽可能保持硬件、模型系列、prompt 和生成任务一致。

对于每一个配置，我在同一个 session 中重复运行三次，然后报告均值和标准差。这一点很重要，因为单次运行可能会误导，尤其是对于 overhead 比较高的代码路径。

测量指标包括 prefill throughput、decode throughput、端到端 wall time 和 peak memory usage。把这些指标放在一起，比只看一个 tokens-per-second 数字更完整。

### 第 9 页：Method: Why Separate Prefill and Decode?

在 LLM 推理中，prefill 和 decode 是两个很不一样的阶段。

Prefill 处理输入 prompt。这个阶段可以对很多 token 做并行计算，通常更接近大的矩阵运算。

Decode 则不同。它一次生成一个 token，每一步都依赖前一步。因此 decode 对 KV cache、attention 重算、Python overhead 和 kernel launch overhead 都更加敏感。

所以，如果只报告总速度，就可能隐藏真正的瓶颈。这就是为什么我要把 prefill 和 decode 分开测量。

### 第 10 页：Result: Overall Throughput

这一页展示整体 throughput 结果。

蓝色线是 mlx-lm，红色线是 SiliconLLM Stage 1 reference implementation。左图是 prefill throughput，右图是 decode throughput。两个图都使用 log scale。

第一个观察是，mlx-lm 在两个阶段都更快。但差距并不一样。在 prefill 阶段，差距存在但相对温和。在 decode 阶段，差距要大得多。这说明简单实现的主要瓶颈在自回归生成阶段。

### 第 11 页：Result: The Gap Is Concentrated in Decode

这一页把性能差距展示得更直接。

对于 prefill，mlx-lm 只比 Stage 1 reference implementation 快大约 1.3 到 1.6 倍。但对于 decode，速度差距大约是 29 到 33 倍。

这是目前最重要的结果。它说明性能问题不是平均分布在整个 pipeline 里。简单实现不是在所有地方都稍微慢一点，而是在 decode 阶段特别弱。

这也为后续优化提供了明确方向。

### 第 12 页：Result: End-to-End Wall Time

throughput 数字很有用，但 wall time 更容易从用户角度理解。

对于同一个 benchmark prompt，0.5B 的 Stage 1 reference 运行大约需要 66.8 秒，而 mlx-lm 大约需要 2.5 秒。对于 1.5B，Stage 1 reference 大约需要 172.7 秒，而 mlx-lm 大约需要 5.9 秒。

所以 decode 差距不仅是一个技术指标，它会直接表现为用户能感受到的延迟。这对任何交互式 LLM 应用都很重要。

### 第 13 页：Result: Memory and Feasibility Boundary

内存也是一个重要结果。

Stage 1 reference implementation 在 0.5B 上的峰值内存大约是 1.9 GB，在 1.5B 上大约是 4.7 GB。而 mlx-lm 在这个 benchmark 中运行 7B 模型时，峰值内存大约是 4.8 GB。

这说明优化影响的不只是速度，也影响模型是否能实际运行。在 18 GB 的机器上，一个未优化路径对于更大的模型可能变得不安全或不现实。而生产级引擎可以让同样规模的模型更可控。

### 第 14 页：Result: Repeatability Check

这一页讨论测量可靠性。

在项目早期，我有一些 N=1 的测量。后来当我用 N=3 重新测量时，其中一个早期趋势发生了变化。之前的单次结果暗示了一个不同的 prefill 趋势，但重复测量说明它很可能是噪声。

这是项目中的一个重要教训。对于 overhead 较高的推理路径，单次测量是不够的。benchmark 方法本身也是这个研究的一部分贡献。

### 第 15 页：Result: 7B Boundary for Stage 1 Reference

Stage 1 reference 缺少 7B 数据点，这件事本身也有意义。

我没有把它作为真实测量结果报告，是因为估计的内存使用接近 18 GB 的机器限制。根据 0.5B 和 1.5B 的测量结果进行外推，decode 可能会非常慢，大约每秒 1 个 token，而且内存也会接近安全边界。

所以这不只是少了一个实验点。它说明了一个可运行性边界。在用 reference 路径运行更大模型之前，优化是必要的。

### 第 16 页：Discussion: What Explains the Gap?

目前的数据告诉我们差距出现在哪里，但还不能精确说明每个原因分别贡献了多少。

不过，有几个原因比较可能。Stage 1 reference 路径没有完整的 KV cache 优化。它可能在 decode 时重复计算 attention。它也没有 mlx-lm 那种优化过的量化矩阵乘法和 kernel fusion。另外，因为生成是逐 token 进行的，Python 层面的 overhead 也可能很重要。

下一步需要使用 profiler，把这个约 30 倍的 decode 差距进一步拆解。

### 第 17 页：Discussion: Next Steps

接下来有三个近期任务。

第一，我会加入 llama.cpp Metal，作为第三个生产级基线。这样对比会更强，因为它增加了另一个成熟引擎。

第二，我会做 workload length scan。我想测试 prompt 长度和生成长度如何影响性能差距。

第三，我会使用 MLX profiler，定位哪些 kernel 和代码路径占据了主要 decode 时间。

之后，我会开始在自己的 source directory 里实现 Stage 1 组件，并逐步加入 KV cache、量化和 attention 优化。

### 第 18 页：Thank You

总结一下，这个项目用 measurement-first 的方式研究 Apple Silicon 上的 LLM 推理。

当前 Phase 0 的结果显示，一个简单实现和 mlx-lm 之间最大的差距在 decode 阶段。Prefill 只相差大约 1.3 到 1.6 倍，但 decode 相差大约 29 到 33 倍。

结果也说明，在消费级硬件上，内存限制也是问题的一部分。我的下一步是增加更多基线，并使用 profiling 更详细地解释 decode bottleneck。

谢谢大家，我很乐意回答问题。

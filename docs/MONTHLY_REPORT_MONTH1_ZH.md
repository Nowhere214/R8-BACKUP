# SiliconLLM 项目：第 1 个月研究进度报告

**主题**：Phase 0 — 基准方法论建立与 LLM 推理优化差距的首次定量刻画
**汇报周期**：2026 年 3 月末 — 2026 年 4 月末
**汇报人**：WANG PEIRANS

---

## 摘要

本月完成本研究的 Phase 0 工作。研究采用第三方开源教学项目 `skyzh/tiny-llm` 作为受控实验载体，在其之上建立个人研究分支 **SiliconLLM**。需要特别说明：tiny-llm 本身是一门由他人编写的 LLM 推理系统课程，其内容为注意力、KV 缓存、Flash Attention、PagedAttention 等**已发表论文技术的教学性复刻**，**本身不构成本项目的研究成果**；本项目的研究贡献是**基于该平台设计的测量框架、所得实证数据、以及由数据驱动的后续工作路径**。

本月在此框架下完成的具体产出：

1. **方法论设计**——基于导师在上月末提出的"在构建任何优化之前先对 vLLM / llama.cpp 进行基准测量以明确可测量指标"的建议，制定了一套在 Apple Silicon 上可复现的 LLM 推理基准测量方法论（指标集、基线选择、工作负载），用以回答"各系统优化对 end-to-end 吞吐的分解贡献"这一研究问题。
2. **首批实证数据**——在同一硬件上对 Qwen2-0.5B-Instruct 完成 SiliconLLM Stage 1 参考实现与生产级引擎 mlx-lm 的对照测量。
3. **核心发现**——**decode 阶段与 prefill 阶段的性能差距存在显著的非对称性**：未优化实现在 decode 吞吐上比 mlx-lm 慢 **34 倍**（7.6 vs 260 tok/s），而在 prefill 上仅慢 **2.4 倍**（1,467 vs 3,562 tok/s）。该分布指向后续工程预算应集中于 decode 链路的 KV 缓存与融合注意力，而非矩阵乘本身，为 Phase 1/2 的工作路径提供了实证依据。

---

## 1. 研究背景与问题

大语言模型（LLM）的推理系统优化是近两年系统领域的热点。从 vLLM 的 PagedAttention、FlashAttention、连续批处理，到 speculative decoding 与 MoE 路由，每一项优化都声称了显著的性能提升。然而，现有文献普遍存在两个局限：

1. 单一优化的评测通常在已经高度优化的基线（如完整版 vLLM）之上叠加，**缺乏对每项优化独立贡献的分解量化**。
2. Apple Silicon 在消费级推理场景（尤其是个人工作站 / 本地 agent）快速普及，但针对 Apple Silicon 的系统级优化研究远少于 NVIDIA GPU 生态。

本项目试图回答的研究问题是：

> **Q1**：在 Apple Silicon 平台上，一个"仅使用矩阵运算 API、无任何系统优化"的 LLM 推理实现，与生产级引擎（mlx-lm / llama.cpp / vLLM）的性能差距具体分布在推理流水线的哪些阶段？
>
> **Q2**：依次引入 KV 缓存、融合量化 matmul、Flash Attention、连续批处理、Paged Attention 等优化，每一项对 end-to-end 吞吐的边际贡献各是多少？
>
> **Q3**：这些优化在 Apple Silicon 上的收益曲线，与其原论文在 NVIDIA GPU 上报告的收益是否一致？若存在差异，差异的物理根因是什么？

需要强调的是：本项目**不以"重新发明 LLM 推理算法"为研究贡献**。上述 Q1–Q3 中所涉及的每一项底层技术（KV 缓存、Flash Attention 2、PagedAttention 等）均已在 2022–2024 年的系统顶会论文中提出并开源。本项目的研究贡献**在于测量层面与分解层面**：既有论文以"在已优化的 vLLM/TGI 上叠加自己的新优化"为主要范式，缺乏对"从零基线到生产基线"之间每个优化阶段的单独量化；本项目填补的是这一空白。

本月工作聚焦 **Q1**——通过 Phase 0 的基准测量，将"多大差距、差在何处"的问题从定性描述转为具体数字。

---

## 2. 研究方法

### 2.1 研究载体：tiny-llm 作为外部实验平台

本研究的实验代码基础来自第三方开源项目 `skyzh/tiny-llm`（作者 Alex Chi 与 Connor，GitHub 4.1k star，许可证 Apache-2.0 / CC-BY-NC-SA-4.0）。**该项目本身定位明确为一门 LLM 推理系统课程**——其 README 首句即为 "A course on LLM serving"，作者本人将其类比为 CMU Deep Learning Systems 课程的 needle 项目。项目内的每一项组件均为已发表论文或既有开源实现的 MLX 复刻，不含新的算法或系统层贡献：


| tiny-llm 中的组件                                  | 原始出处                          |
| ---------------------------------------------- | ----------------------------- |
| Scaled Dot-Product Attention、Transformer Block | Vaswani et al., NeurIPS 2017  |
| Rotary Positional Encoding (RoPE)              | Su et al., 2021               |
| Grouped Query Attention                        | Ainslie et al., 2023          |
| Flash Attention 2                              | Dao, 2023                     |
| Continuous Batching                            | Yu et al., OSDI 2022 (Orca)   |
| Paged Attention                                | Kwon et al., SOSP 2023 (vLLM) |
| Speculative Decoding                           | Leviathan et al., ICML 2023   |
| Quantized matmul (int4)                        | llama.cpp / GPTQ 系列           |


本研究选定它作为实验载体，而非自行从零搭建，是基于以下工程考量：

1. **"优化台阶"结构天然契合 Q2 的分解测量需求**——tiny-llm 的 Stage 1 提供"零系统优化"基线，Stage 2 对应 KV 缓存 + Flash Attention + 连续批处理等工业级优化，Stage 3 对应 Paged Attention / Speculative Decoding 等前沿优化。研究者可在同一代码库内**逐项打开优化并测量其增量贡献**，避免了跨代码库比较引入的大量噪声变量。
2. **减少非研究性投入的工程成本**——从零实现一套 Qwen2 MLX 后端本身需要数周工程量，但这部分工作不属于本项目的研究贡献范畴。采用已有平台可以把研究者时间集中在真正有研究价值的测量、分析与原创优化设计上。
3. **可复现性**——公开平台意味着本研究产出的任何发现都可由第三方独立验证。

### 2.2 本项目的研究贡献与 tiny-llm 的角色分工

为避免对"本项目研究成果"的认定产生混淆，在此明确列出双方分工：


| 来自 tiny-llm 的部分（不计为本项目贡献）             | 本项目独立贡献（本月及后续产出）                                      |
| ------------------------------------- | ----------------------------------------------------- |
| 注意力、RoPE、GQA、MLP、Qwen2 模型结构的 MLX 实现代码 | Phase 0 基准方法论的设计（指标集、基线选择、工作负载标准化）                    |
| Stage 1–3 的任务拆分与教学路径                   | 方法论的学术论证（为什么 vLLM-on-CPU 无意义、为什么 mlx-lm 是正确的"同框架天花板"） |
| 参考实现 `tiny_llm_ref/`，可作为正确性 oracle    | 实证测量数据及其解读（本月产出：见第 3 节表 1）                          |
| `bench.py` 测量工具的基本结构                  | **关键发现：decode–prefill 性能差距的非对称性**（见第 3.3 节）         |
| Stage 2 Metal/C++ 扩展的编译脚手架             | 可证伪的 Phase 2/3 成功判据（见第 4.2 节）                       |
| 单元测试框架                                | Phase 3 原创优化选题（TBD，预计在 Stage 3 未完成章节中选择一个进行原创设计）       |


换言之：**tiny-llm 之于本研究的角色，等同于 PyTorch / MLX 之于绝大多数深度学习论文的角色——是实验基础设施，不是研究贡献本身。**

### 2.3 Phase 0 基准方法论

**指标集**（与业界通行指标对齐）：


| 指标                         | 定义                               | 关注点                         |
| -------------------------- | -------------------------------- | --------------------------- |
| Prefill throughput (tok/s) | `prompt_tokens / prefill_time`   | 长 prompt 短回复场景（例如 RAG / 摘要） |
| Decode throughput (tok/s)  | `generated_tokens / decode_time` | 长回复场景（聊天 / 代码生成）            |
| TTFT (Time To First Token) | 请求到达至首 token 输出的延迟               | 交互式用户体验                     |
| Peak memory (GB)           | 最大 RSS                           | 消费级硬件可行性                    |


**基线选择与论证**：

- **mlx-lm**（生产级 MLX 引擎）— 作为"同框架同硬件的性能天花板"。由于其与 SiliconLLM 共享底层运行时（MLX），两者的差距纯粹来自系统优化层面的实现质量差异，是**最严格的可比性基线**。
- **llama.cpp**（GGML + Metal）— 作为"跨框架跨生态的产业基线"。代表 Apple Silicon 上用户实际使用的推理引擎，任何声称的研究成果若能逼近它才具有实用意义。
- **vLLM**（CUDA 原生）— 因其核心优化（PagedAttention 的 CUDA 实现、Triton 融合 kernel）与 Apple Silicon 架构不兼容，在 Mac 上仅能退化为 CPU 模式运行，与其设计意图严重背离。本月决定将其**降级为可选补充**（详见第 6 节讨论），可在后续阶段通过云端 GPU 时租方式补测。

**工作负载设定**（沿用 tiny-llm 上游 `bench.py` 的默认参数，以保证本研究自身不同阶段的测量与外部引擎测量使用完全一致的输入分布）：

- 合成请求，random seed = 0
- Prompt 长度：均匀分布 [64, 256] token
- 输出长度：均匀分布 [64, 256] token
- 贪婪解码（temperature = 0）
- 1 次 warm-up，后正式测量
- 每项测量 4 请求（本轮）/ 16 请求（完整轮次，视磁盘与模型规模逐步放大）

方法论的完整、可复现描述已归档于 `docs/PHASE0_BENCHMARK.md`。

---

## 3. 实验结果

### 3.1 硬件与模型


| 项   | 取值                                     |
| --- | -------------------------------------- |
| 硬件  | MacBook Pro，Apple Silicon，macOS 14.6.1 |
| 模型  | Qwen2-0.5B-Instruct，int4 量化（MLX 格式）    |
| 框架  | MLX 0.29.1 + Python 3.12.13            |


> 注：本月受磁盘容量限制，7B 规模的完整对照测量延后至 5 月初进行；0.5B 规模的结论在量级上仍具有代表性，主要节点上的非对称差距模式在更大模型下预期会继续保持或放大。

### 3.2 对照测量结果

**表 1**：SiliconLLM Stage 1 参考实现 vs mlx-lm，相同硬件、相同模型、相同 4 请求合成工作负载


| 指标                 | SiliconLLM Stage 1 ref | mlx-lm        | 比率        |
| ------------------ | --------------------- | ------------- | --------- |
| Prefill throughput | 1,467 tok/s           | 3,562 tok/s   | **2.4×**  |
| Decode throughput  | **7.6 tok/s**         | **260 tok/s** | **34.2×** |
| 4 请求总 wall time    | 89.4 s                | 2.83 s        | 31.6×     |
| 峰值内存               | 未测                    | 0.53 GB       | —         |


> 完整的每请求原始数据及复现命令已归档于 `docs/PHASE0_RESULTS.md`。

### 3.3 主要观察

结果最显著的特征是**差距的非对称性**。若仅看 end-to-end 耗时比（31.6×），容易以为"全面落后"；但分解到子阶段后可以清楚看到：

- **Prefill 仅差 2.4 倍**：这是因为 prefill 是一次单轮大矩阵乘，MLX 框架本身的原生 matmul kernel 对未优化实现已经提供了显著加速。剩余的 2.4 倍差距主要来源于 mlx-lm 的融合 int4-反量化-matmul（一次 kernel 内同时完成反量化与乘法），相当于节省了一次全尺寸激活的 memory round-trip。
- **Decode 差 34 倍**：这并非 kernel 效率问题，而是**算法复杂度差**。未优化实现在每生成一个 token 时，都会对完整的 `[prefill + 已生成上下文]` 重新计算一次注意力（O(L²) 计算/token）。引入 KV 缓存后，每 token 的注意力成本从 O(L²) 降至 O(L)，在本次实验的 L ≈ 100–400 规模下即可产生约 1–2 个数量级的加速，这与 34× 的观察量级吻合。

---

## 4. 研究含义

### 4.1 优化预算的定位

表 1 的结果直接指导后续工程预算的分配：

- 针对 **prefill**（2.4× 差距）的优化，其上限由量化融合 kernel 决定，边际改进空间有限。
- 针对 **decode**（34× 差距）的优化，其空间巨大且集中在少数几个已知机制上（KV 缓存、Flash Attention、连续批处理）。

值得一提的是，这一分布在方向上与 tiny-llm 作者选择"Stage 2 集中于 KV 缓存 + Flash Attention + 连续批处理"的教学设计一致——但此前该选择更多依赖作者的工程直觉，并无开源的定量证据。本项目的数据**独立验证**了该优先级排序在 Apple Silicon 上的正确性，并为后续研究者（或未来阅读该课程的工程师）提供了可引用的具体数字。此外，这组数字也使本项目在 Phase 1 自实现时的工作重心不再依赖第三方教程的顺序假设，而是有了自己的数据依据。

### 4.2 可证伪的下阶段目标

基于以上测量，我为后续阶段设立可证伪的成功判据，使进度变得可定量评估：

- **Phase 2（Stage 2 自实现完成）**：SiliconLLM decode 吞吐应达到 mlx-lm 同机测量的 **至少 1/3**（≈ 87 tok/s on 同硬件）。
  - 若低于 1/5（≤ 52 tok/s）：表明 Stage 2 的 KV 缓存或 Flash Attention 实现质量不足，需深入剖析瓶颈。
  - 若高于 1/1.5（≥ 173 tok/s）：可疑偏高，需排查是否意外使用了 mlx-lm 的内部融合算子。
- **Phase 3（Stage 3 某一进阶选题）**：预计在 Paged Attention 与 Speculative Decoding 之间二选一，产出该优化**相对 Stage 2 基线**的增量收益曲线，并与原论文在 NVIDIA GPU 上报告的收益进行对比（回答 Q3）。

---

## 5. 下阶段计划（Phase 0 收尾 → Phase 1 启动）

**Phase 0 收尾（第 2 个月前半）**：

1. 完成 llama.cpp Metal 基线在 Qwen2-0.5B 上的对照测量（跨框架对照点）。
2. 将 bench 脚本正规化为 `benches/phase0_compare.py`，支持 `--engine {mlx_lm, llama_cpp, silicon_llm}` 统一入口，输出 CSV 便于后续画图与文档。
3. 输出 Phase 0 小论文风格总结（2–3 页），可作为正式研究报告的第一章。

**Phase 1 启动（第 2 个月后半）**：

1. 以 tiny-llm 提供的参考实现（`tiny_llm_ref/`）为正确性 oracle，自行完成 Qwen2 推理栈在 Apple Silicon 上的"零优化"重实现（注意力、RoPE、GQA、RMSNorm、MLP、采样）。**此阶段的研究价值不在于代码本身（它会与参考实现在功能上等价），而在于后续步骤 2**。
2. 在自实现完成后重跑 Phase 0 基准，验证自实现与参考实现的 tok/s 在 ±5% 以内。若偏差 > 5%，则差异本身构成一个需独立调查的研究问题（例如"MLX graph compilation 是否对同功能等价的不同实现产生不同程度的融合"），可能形成一篇独立的 short note。

**已识别的阻塞项**：

- 完成 Stage 2 参考实现的 GPU 路径测量需要完整版 Xcode.app（约 15 GB），已规划至 Phase 2 启动前安装。不影响本月与下月工作。
- 7B 模型三引擎完整对照需磁盘余量 ≥ 15 GB；已规划先完成 0.5B 全对照，7B 延后。

---

## 6. 讨论

### 6.1 Phase 0 的实际研究价值（超出"建立标尺"本身）

导师上月提出的"先测量再构建"建议，在本月实施中表现出三重价值：

1. **方向定位**：若未进行 Phase 0 直接开始自实现 Stage 1，我不会知道"prefill 已经只剩 2.4× 空间，decode 才是真正的战场"。Phase 0 把"应把时间花在哪里"的问题从猜测变成了数据驱动的决策。
2. **可证伪进度**：有了 mlx-lm 的 260 tok/s 作为硬上限，Phase 1/2 的任何实现都不再是"能跑就行"，而是对照一个明确的定量目标。
3. **避免伪优化陷阱**：若无 mlx-lm 基线，当自实现 Stage 2 版本达到 100 tok/s 时很容易产生"已经很快了"的错觉。有了 260 tok/s 这个上限，才能清楚看到仍有 2.6× 差距有待改进。

### 6.2 关于是否纳入 vLLM 的思考

vLLM 在本月的方法论讨论中被降级为"可选补充"。理由如下：

- vLLM 的核心技术贡献（PagedAttention CUDA 实现、Triton 融合 kernel、CUDA graph capture）均依赖 NVIDIA GPU 生态。
- 在 Mac 上运行 vLLM 必须退化至 CPU 后端。此形态下的 vLLM 既不代表它在生产中的实际性能（差数十倍），也不能用于评估其技术设计的有效性，产生的数字对任何研究问题都**无信息量**。
- 若后续确需将 vLLM 纳入对照，建议通过云端 GPU 按小时租用（单次约 5 美元即可完成一组完整测量），以得到能反映其设计价值的真实数字。此项作为"Phase 0.5 可选扩展"暂存。

建议导师意见：**将 mlx-lm + llama.cpp 作为 Phase 0 的主基线对，vLLM 作为可选的云端对照**。

### 6.3 7B vs 0.5B 的外推有效性

本月的测量集中在 0.5B 规模，主要出于磁盘与迭代速度考虑。我预期将规模扩到 7B 后的定性结论保持不变（decode 差距仍显著大于 prefill 差距），但二者的**绝对差距比**可能变化：

- **假设一**：在 7B 上两个差距都变小。因为模型变大后，每步计算时间被 kernel 效率主导而非 overhead 主导，Python 开销占比下降。
- **假设二**：decode 差距进一步扩大。因为无 KV 缓存的 O(L²) 成本随上下文长度超线性增长，而 L ≈ 200 在 0.5B 上已经凸显，7B 上更凸显。

下月的 7B 实测将直接验证这两个假设。

---

## 7. 参考资源

**本研究使用的外部实验平台**：

- Alex Chi (skyzh), Connor. *Tiny LLM — LLM Serving in a Week*. 开源教学项目，GitHub: [https://github.com/skyzh/tiny-llm](https://github.com/skyzh/tiny-llm), 课程页: [https://skyzh.github.io/tiny-llm/](https://skyzh.github.io/tiny-llm/) （accessed 2026-04）。本研究的实验代码、Qwen2 MLX 模型加载、测试框架、基础 bench.py 结构均来自该项目。许可证 Apache-2.0（代码）/ CC BY-NC-SA 4.0（教材）。

**本研究使用的开源软件与模型**：

- Apple. *MLX: An array framework for Apple Silicon*. [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx).
- Apple. *mlx-lm: LM serving on MLX*. [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm).
- G. Gerganov et al. *llama.cpp: Inference of LLaMA in pure C/C++*. [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).
- Alibaba (Qwen Team). *Qwen2-0.5B/7B-Instruct-MLX*. Hugging Face.

**本研究引用的学术工作**（为正文中提及的技术标注原始出处）：

- Vaswani et al. *Attention Is All You Need*. NeurIPS 2017.
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021.
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. 2023.
- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022（连续批处理原论文）。
- T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv:2307.08691.
- W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023（vLLM 原论文）。
- Leviathan et al. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
- A. Yang et al. *Qwen2 Technical Report*. arXiv:2407.10671.

---

## 附录 A：原始测量数据与复现命令

详见 `docs/PHASE0_RESULTS.md`（随此报告一并归档）。

## 附录 B：方法论完整描述

详见 `docs/PHASE0_BENCHMARK.md`（随此报告一并归档）。
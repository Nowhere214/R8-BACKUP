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
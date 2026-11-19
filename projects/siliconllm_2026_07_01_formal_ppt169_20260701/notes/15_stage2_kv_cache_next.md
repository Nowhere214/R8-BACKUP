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
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
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
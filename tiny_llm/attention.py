import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = 1.0 / (query.shape[-1] ** 0.5) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / (self.head_dim**0.5)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch, seq_len, _ = query.shape
        projection_q = (
            linear(query, self.wq)
            .reshape(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        x = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )
        x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_size)
        return linear(x, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=S - L)
    return mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = 1.0 / (query.shape[-1] ** 0.5) if scale is None else scale
    factor = mx.array(factor).astype(query.dtype)
    expected_shape = query.shape

    num_query_heads, query_len, head_dim = query.shape[-3:]
    num_kv_heads, key_len, _ = key.shape[-3:]
    batch_shape = query.shape[:-3]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    repeats = num_query_heads // num_kv_heads

    query = query.reshape(*batch_shape, -1, num_kv_heads, repeats, query_len, head_dim)
    key = key.reshape(*batch_shape, -1, num_kv_heads, 1, key_len, head_dim)
    value = value.reshape(*batch_shape, -1, num_kv_heads, 1, key_len, head_dim)

    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            scores = scores + causal_mask(query_len, key_len, scores.dtype)
        else:
            mask = mx.broadcast_to(mask, (*batch_shape, num_query_heads, query_len, key_len))
            mask = mask.reshape(*batch_shape, 1, num_kv_heads, repeats, query_len, key_len)
            scores = scores + mask
    result = mx.matmul(softmax(scores, axis=-1), value)
    return result.reshape(expected_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = 1.0 / (query.shape[-1] ** 0.5) if scale is None else scale
    if isinstance(mask, mx.array):
        batch_shape = query.shape[:-3]
        num_query_heads, query_len, _ = query.shape[-3:]
        key_len = key.shape[-2]
        mask = mx.broadcast_to(mask, (*batch_shape, num_query_heads, query_len, key_len))
    return mx.fast.scaled_dot_product_attention(
        q=mx.contiguous(query),
        k=mx.contiguous(key),
        v=mx.contiguous(value),
        scale=factor,
        mask=mask,
    )

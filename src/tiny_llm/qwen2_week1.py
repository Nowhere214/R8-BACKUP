import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / (self.head_dim**0.5)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        q = linear(x, self.wq, self.bq).reshape(
            batch, seq_len, self.num_heads, self.head_dim
        )
        k = linear(x, self.wk, self.bk).reshape(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        v = linear(x, self.wv, self.bv).reshape(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        q = self.rope(q, offset=slice(0, seq_len))
        k = self.rope(k, offset=slice(0, seq_len))
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        y = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)
        y = y.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_size)
        return linear(y, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        self.precision = mx.float16

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(self.precision),
        )
        self.layers_inner = []
        for i in range(self.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj)

            self.layers_inner.append(
                Qwen2TransformerBlock(
                    num_attention_heads=mlx_model.args.num_attention_heads,
                    num_kv_heads=mlx_model.args.num_key_value_heads,
                    hidden_size=mlx_model.args.hidden_size,
                    intermediate_size=mlx_model.args.intermediate_size,
                    rms_norm_eps=mlx_model.args.rms_norm_eps,
                    wq=wq.astype(self.precision),
                    wk=wk.astype(self.precision),
                    wv=wv.astype(self.precision),
                    wo=wo.astype(self.precision),
                    bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(
                        self.precision
                    ),
                    bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(
                        self.precision
                    ),
                    bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(
                        self.precision
                    ),
                    w_gate=w_gate.astype(self.precision),
                    w_up=w_up.astype(self.precision),
                    w_down=w_down.astype(self.precision),
                    w_input_layernorm=mlx_model.model.layers[
                        i
                    ].input_layernorm.weight.astype(self.precision),
                    w_post_attention_layernorm=mlx_model.model.layers[
                        i
                    ].post_attention_layernorm.weight.astype(self.precision),
                    max_seq_len=mlx_model.args.max_position_embeddings,
                    theta=mlx_model.args.rope_theta,
                )
            )
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(self.precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if mlx_model.args.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head).astype(self.precision)
        self.mlx_model = mlx_model

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in self.layers_inner:
            h = layer(h, mask="causal")
        h = self.norm(h)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        return self.embedding.as_linear(h)

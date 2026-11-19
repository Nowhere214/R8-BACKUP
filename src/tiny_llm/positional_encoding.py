import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        if dims % 2 != 0:
            raise ValueError("dims must be even")
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.half_dims = dims // 2

        dims_idx = mx.arange(0, self.half_dims, dtype=mx.float32) / self.half_dims
        inv_freq = mx.power(base, -dims_idx)
        positions = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, inv_freq)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        batch, seq_len, num_heads, dims = x.shape
        if dims != self.dims:
            raise ValueError(f"expected last dimension {self.dims}, got {dims}")

        if offset is None:
            cos_basis = self.cos_freqs[:seq_len, :]
            sin_basis = self.sin_freqs[:seq_len, :]
        elif isinstance(offset, slice):
            if offset.start is None or offset.stop is None:
                raise ValueError("offset slice must have start and stop")
            if offset.stop - offset.start != seq_len:
                raise ValueError(f"offset must be of length {seq_len}")
            cos_basis = self.cos_freqs[offset, :]
            sin_basis = self.sin_freqs[offset, :]
        else:
            if len(offset) != batch:
                raise ValueError("offsets must match batch size")
            indices = []
            for item in offset:
                if item.start is None or item.stop is None:
                    raise ValueError("offset slices must have start and stop")
                if item.stop - item.start != seq_len:
                    raise ValueError(f"offset must be of length {seq_len}")
                indices.append(list(range(item.start, item.stop)))
            offset_idx = mx.array(indices)
            cos_basis = self.cos_freqs[offset_idx, :]
            sin_basis = self.sin_freqs[offset_idx, :]

        if self.traditional:
            x_pairs = x.reshape(batch, seq_len, num_heads, self.half_dims, 2)
            x1 = x_pairs[..., 0]
            x2 = x_pairs[..., 1]
        else:
            x1 = x[..., : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]

        cos_basis = cos_basis.reshape(-1, seq_len, 1, self.half_dims)
        sin_basis = sin_basis.reshape(-1, seq_len, 1, self.half_dims)
        real = x1 * cos_basis - x2 * sin_basis
        imag = x1 * sin_basis + x2 * cos_basis

        if self.traditional:
            y = mx.stack([real, imag], axis=-1).reshape(batch, seq_len, num_heads, dims)
        else:
            y = mx.concat([real, imag], axis=-1).reshape(batch, seq_len, num_heads, dims)
        return y.astype(x.dtype)

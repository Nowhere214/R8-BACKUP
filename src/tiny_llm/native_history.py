"""Snapshot adapter for a native mlx-lm non-rotating KV cache."""

from copy import copy

from .adaptive_decode import ModelEngine


class NativeHistoryEngine(ModelEngine):
    def __init__(self, model):
        super().__init__(model, native=True)

    def snapshot(self, cache):
        from mlx_lm.models.cache import KVCache

        if any(type(layer) is not KVCache for layer in cache):
            raise TypeError("Only the native non-rotating KVCache is supported")
        # MLX array wrappers are mutable under indexed assignment. Separate
        # wrappers retain the pre-update graph without copying GPU buffers.
        return [(copy(layer.keys), copy(layer.values), layer.offset) for layer in cache]

    def restore(self, cache, snapshot):
        for layer, (keys, values, offset) in zip(cache, snapshot):
            layer.keys, layer.values, layer.offset = copy(keys), copy(values), offset

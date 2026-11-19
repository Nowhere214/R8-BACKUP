from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx
from .attention import causal_mask


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[TinyKvCache | None] = [None] * max_active_requests
        self.HD = None

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        B, H, S, D = keys.shape
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        if S > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if B != self.max_active_requests:
            raise ValueError("batch size must match max_active_requests")
        if self.HD is None:
            self.HD = (H, D)
        elif self.HD != (H, D):
            raise ValueError(f"expected cached head shape {self.HD}, got {(H, D)}")

        data = []
        for batch_idx in range(B):
            cache = self.kv_caches[batch_idx]
            if cache is None:
                data.append(None)
                continue
            new_key = keys[batch_idx : batch_idx + 1]
            new_value = values[batch_idx : batch_idx + 1]
            data.append(cache.update_and_fetch(new_key, new_value))

        seq_len = max((item[2] if item is not None else 0 for item in data), default=0)
        batched_keys = mx.zeros((B, H, seq_len, D), dtype=keys.dtype)
        batched_values = mx.zeros((B, H, seq_len, D), dtype=values.dtype)

        if mask_length is None:
            mask_length = S
        batched_mask = mx.full((B, mask_length, seq_len), -mx.inf, dtype=keys.dtype)

        for batch_idx, item in enumerate(data):
            if item is None:
                continue
            cached_key, cached_value, item_seq_len, item_mask = item
            batched_keys[batch_idx, :, seq_len - item_seq_len : seq_len, :] = cached_key[0]
            batched_values[batch_idx, :, seq_len - item_seq_len : seq_len, :] = cached_value[0]
            if item_mask is None or item_mask == "causal":
                batched_mask[batch_idx, :, seq_len - item_seq_len : seq_len] = causal_mask(
                    mask_length, item_seq_len, dtype=keys.dtype
                )
            elif isinstance(item_mask, mx.array):
                batched_mask[batch_idx, :, seq_len - item_seq_len : seq_len] = item_mask
            else:
                raise NotImplementedError(f"unsupported mask: {item_mask}")

        return batched_keys, batched_values, None, batched_mask.reshape(
            B, 1, mask_length, seq_len
        )

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"request id {id} is out of range")
        if getattr(prefilled, "key_values", None) is not None:
            keys, _ = prefilled.key_values
            B, H, _, D = keys.shape
            if B != 1:
                raise ValueError("prefilled cache must contain exactly one request")
            if self.HD is None:
                self.HD = (H, D)
            elif self.HD != (H, D):
                raise ValueError(f"expected cached head shape {self.HD}, got {(H, D)}")
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"request id {id} is out of range")
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        B, H, S, D = key.shape
        if self.key_values is None:
            if self.offset != 0:
                raise ValueError("empty cache must have offset 0")
            self.key_values = (key, value)
            self.offset = S
            return key, value, self.offset, mask

        prev_key, prev_value = self.key_values
        if prev_key.shape != (B, H, self.offset, D):
            raise ValueError("cached key shape does not match current offset")
        if prev_value.shape != (B, H, self.offset, D):
            raise ValueError("cached value shape does not match current offset")

        new_key = mx.concat([prev_key, key], axis=2)
        new_value = mx.concat([prev_value, value], axis=2)
        self.key_values = (new_key, new_value)
        self.offset += S
        return new_key, new_value, self.offset, mask

    def rewind(self, n: int):
        if self.key_values is None:
            raise ValueError("cannot rewind an empty cache")
        if n < 0 or n > self.offset:
            raise ValueError("invalid rewind length")
        self.offset -= n
        key, value = self.key_values
        self.key_values = (key[:, :, : self.offset, :], value[:, :, : self.offset, :])

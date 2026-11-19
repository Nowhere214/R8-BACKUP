import mlx.core as mx
from typing import Any


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight
        self._dequantized: mx.array | None = None

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )

    def dequantized(self) -> mx.array:
        if self._dequantized is None:
            self._dequantized = mx.dequantize(
                self.weight,
                self.scales,
                self.biases,
                self.group_size,
                self.bits,
            )
        return self._dequantized


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    *batch_shape, in_dim = a.shape
    a_2d = a.reshape(-1, in_dim)
    out = mx.quantized_matmul(
        a_2d,
        b,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        transpose=transpose_b,
    )
    return out.reshape(*batch_shape, out.shape[-1])


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    out = quantized_matmul(w.scales, w.biases, w.group_size, w.bits, x, w.weight, True)
    if bias is not None:
        out = out + bias
    return out

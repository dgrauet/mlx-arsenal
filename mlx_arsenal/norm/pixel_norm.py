"""Miscellaneous normalization layers."""

import mlx.core as mx
import mlx.nn as nn


class PixelNorm(nn.Module):
    """Pixel-wise feature vector normalization (ProGAN / StyleGAN).

    Normalizes each feature vector to unit length along the channel dimension.

    Args:
        eps: Epsilon for numerical stability.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)


class ScaleNorm(nn.Module):
    """Scale normalization (L2 normalize then scale by a learned parameter).

    Args:
        dim: Feature dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / norm) * self.scale

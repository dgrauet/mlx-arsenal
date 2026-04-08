"""Multi-dimensional Rotary Position Embeddings (RoPE).

Extends standard 1D RoPE to 2D (images) and 3D (video) by factorizing
the embedding dimensions across spatial/temporal axes.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _compute_freqs(dim: int, max_len: int, theta: float = 10000.0) -> mx.array:
    """Compute frequency bands for RoPE.

    Returns shape (max_len, dim//2) of angles.
    """
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(max_len, dtype=mx.float32)
    return mx.expand_dims(t, 1) * mx.expand_dims(freqs, 0)  # (max_len, dim//2)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor (..., seq_len, dim). dim must be even.
        cos: Cosine frequencies (..., seq_len, dim//2).
        sin: Sine frequencies (..., seq_len, dim//2).

    Returns:
        Rotated tensor of the same shape as x.
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return mx.concatenate([out1, out2], axis=-1)


class RoPE2d(nn.Module):
    """2D Rotary Position Embeddings for image/spatial data.

    Splits the embedding dimension in half: one portion encodes the
    row position, the other encodes the column position.

    Args:
        dim: Total embedding dimension (must be divisible by 4).
        max_h: Maximum height.
        max_w: Maximum width.
        theta: Base frequency.
    """

    def __init__(self, dim: int, max_h: int = 256, max_w: int = 256, theta: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4 for 2D RoPE, got {dim}"
        self.dim = dim
        half = dim // 2  # each axis gets half the dimensions
        self._freqs_h = _compute_freqs(half, max_h, theta)  # (max_h, half//2)
        self._freqs_w = _compute_freqs(half, max_w, theta)  # (max_w, half//2)

    def __call__(self, x: mx.array, h: int, w: int) -> mx.array:
        """Apply 2D RoPE to x.

        Args:
            x: Shape (B, N, H_heads, D) or (B, N, D) where N = h * w.
            h: Height of the 2D grid.
            w: Width of the 2D grid.

        Returns:
            Tensor with rotary embeddings applied.
        """
        squeeze = x.ndim == 3
        if squeeze:
            x = mx.expand_dims(x, axis=2)

        half = self.dim // 2
        quarter = half // 2

        # Get freqs for each axis
        fh = self._freqs_h[:h]  # (h, quarter)
        fw = self._freqs_w[:w]  # (w, quarter)

        # Expand to 2D grid: (h*w, quarter)
        cos_h = mx.cos(mx.repeat(fh, w, axis=0))
        sin_h = mx.sin(mx.repeat(fh, w, axis=0))
        cos_w = mx.cos(mx.tile(fw, (h, 1)))
        sin_w = mx.sin(mx.tile(fw, (h, 1)))

        # Concatenate h and w frequencies
        cos = mx.concatenate([cos_h, cos_w], axis=-1)  # (h*w, half)
        sin = mx.concatenate([sin_h, sin_w], axis=-1)

        # Reshape for broadcasting: (1, h*w, 1, half)
        cos = cos.reshape(1, h * w, 1, half)
        sin = sin.reshape(1, h * w, 1, half)

        x = apply_rope(x, cos, sin)

        if squeeze:
            x = x.squeeze(axis=2)
        return x


class RoPE3d(nn.Module):
    """3D Rotary Position Embeddings for video/spatiotemporal data.

    Splits the embedding dimension into three portions: temporal,
    height, and width, with configurable allocation ratios.

    Args:
        dim: Total embedding dimension (must be divisible by 2).
        max_t: Maximum temporal length.
        max_h: Maximum height.
        max_w: Maximum width.
        theta: Base frequency.
        dim_ratios: Fraction of dimensions for (temporal, height, width).
            Must sum to 1.0. Default splits evenly.
    """

    def __init__(
        self,
        dim: int,
        max_t: int = 128,
        max_h: int = 128,
        max_w: int = 128,
        theta: float = 10000.0,
        dim_ratios: tuple = (1 / 3, 1 / 3, 1 / 3),
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.dim = dim
        half = dim // 2

        # Allocate dimensions per axis (must be even each)
        raw = [int(r * half) for r in dim_ratios]
        # Make each even
        for i in range(len(raw)):
            if raw[i] % 2 != 0:
                raw[i] += 1
        # Adjust last to match total
        raw[-1] = half - sum(raw[:-1])
        if raw[-1] <= 0:
            raw[-1] = 2
            raw[0] = half - raw[1] - raw[2]

        self.dims_t, self.dims_h, self.dims_w = raw

        self._freqs_t = _compute_freqs(self.dims_t * 2, max_t, theta)
        self._freqs_h = _compute_freqs(self.dims_h * 2, max_h, theta)
        self._freqs_w = _compute_freqs(self.dims_w * 2, max_w, theta)

    def __call__(self, x: mx.array, t: int, h: int, w: int) -> mx.array:
        """Apply 3D RoPE to x.

        Args:
            x: Shape (B, N, H_heads, D) or (B, N, D) where N = t * h * w.
            t: Temporal length.
            h: Height.
            w: Width.

        Returns:
            Tensor with 3D rotary embeddings applied.
        """
        squeeze = x.ndim == 3
        if squeeze:
            x = mx.expand_dims(x, axis=2)

        n = t * h * w

        # Temporal freqs: repeat for each spatial position
        ft = self._freqs_t[:t]  # (t, dims_t)
        ft = mx.repeat(ft, h * w, axis=0)  # (t*h*w, dims_t)

        # Height freqs: tile within each temporal frame
        fh = self._freqs_h[:h]  # (h, dims_h)
        fh = mx.repeat(fh, w, axis=0)  # (h*w, dims_h)
        fh = mx.tile(fh, (t, 1))  # (t*h*w, dims_h)

        # Width freqs: tile across both temporal and height
        fw = self._freqs_w[:w]  # (w, dims_w)
        fw = mx.tile(fw, (t * h, 1))  # (t*h*w, dims_w)

        # Concatenate all axes
        freqs = mx.concatenate([ft, fh, fw], axis=-1)  # (n, dim//2)
        cos = mx.cos(freqs).reshape(1, n, 1, -1)
        sin = mx.sin(freqs).reshape(1, n, 1, -1)

        x = apply_rope(x, cos, sin)

        if squeeze:
            x = x.squeeze(axis=2)
        return x

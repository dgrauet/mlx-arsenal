"""Timestep embeddings for diffusion models."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: float = 10000.0,
) -> mx.array:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D array of timestep values.
        embedding_dim: Dimension of the output embeddings.
        flip_sin_to_cos: If True, output ``[cos, sin]``; if False, ``[sin, cos]``.
        downscale_freq_shift: Controls delta between frequencies.
        scale: Scaling factor applied before sin/cos.
        max_period: Controls the minimum frequency.

    Returns:
        Array of shape ``(len(timesteps), embedding_dim)``.
    """
    half = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half).astype(mx.float32)
    exponent = exponent / (half - downscale_freq_shift)
    freqs = mx.exp(exponent)
    args = timesteps[:, None].astype(mx.float32) * freqs[None, :]
    args = scale * args
    if flip_sin_to_cos:
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    else:
        embedding = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if embedding_dim % 2:
        embedding = mx.pad(embedding, [(0, 0), (0, 1)])
    return embedding


class TimestepEmbedding(nn.Module):
    """MLP that projects sinusoidal timestep embeddings into a feature space.

    Weight keys: ``linear1.{weight,bias}``, ``linear2.{weight,bias}``.
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, time_embed_dim)
        self.linear2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = nn.silu(self.linear1(sample))
        return self.linear2(sample)

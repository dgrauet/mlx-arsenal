"""Timestep embeddings for diffusion models."""

import math

import mlx.core as mx
import mlx.nn as nn


def sinusoidal_embedding(
    timesteps: mx.array,
    dim: int,
    max_period: float = 10000.0,
) -> mx.array:
    """Compute sinusoidal positional embeddings for diffusion timesteps.

    Args:
        timesteps: 1D tensor of timestep indices (B,).
        dim: Embedding dimension (must be even).
        max_period: Controls the frequency range.

    Returns:
        Embeddings of shape (B, dim).
    """
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = mx.expand_dims(timesteps.astype(mx.float32), 1) * mx.expand_dims(freqs, 0)
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class TimestepEmbedding(nn.Module):
    """Timestep embedding: sinusoidal encoding + 2-layer MLP projection.

    Standard pattern used in DDPM, Stable Diffusion, etc.

    Args:
        dim: Output embedding dimension.
        hidden_dim: Hidden dimension of the MLP. Defaults to 4 * dim.
        max_period: Base frequency for sinusoidal encoding.
        freq_dim: Dimension of the sinusoidal encoding. Defaults to dim.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 0,
        max_period: float = 10000.0,
        freq_dim: int = 0,
    ):
        super().__init__()
        self.max_period = max_period
        self.freq_dim = freq_dim or dim
        hidden_dim = hidden_dim or 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def __call__(self, timesteps: mx.array) -> mx.array:
        """Embed timesteps.

        Args:
            timesteps: (B,) tensor of timestep values.

        Returns:
            (B, dim) embeddings.
        """
        emb = sinusoidal_embedding(timesteps, self.freq_dim, self.max_period)
        return self.mlp(emb)

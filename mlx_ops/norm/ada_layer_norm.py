"""Adaptive Layer Normalization variants for diffusion models.

These layers modulate LayerNorm output with learned shift/scale/gate
parameters conditioned on a timestep or other embedding.
"""

import mlx.core as mx
import mlx.nn as nn


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Norm with zero-initialized gating (DiT-style).

    Produces shift, scale, and gate for both attention and MLP branches.
    Output: 6 modulation vectors from a single conditioning input.

    Args:
        dim: Model dimension.
        cond_dim: Conditioning embedding dimension. Defaults to dim.
    """

    def __init__(self, dim: int, cond_dim: int = 0):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm = nn.LayerNorm(dim, affine=False)
        self.linear = nn.Linear(cond_dim, 6 * dim)

    def __call__(self, x: mx.array, cond: mx.array):
        """Apply adaptive normalization.

        Args:
            x: Input tensor (B, L, D).
            cond: Conditioning tensor (B, cond_dim).

        Returns:
            Tuple of (normed_x, shift_attn, scale_attn, gate_attn,
                       shift_mlp, scale_mlp, gate_mlp).
        """
        modulation = nn.silu(cond)
        modulation = self.linear(modulation)
        if modulation.ndim == 2:
            modulation = mx.expand_dims(modulation, axis=1)

        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            mx.split(modulation, 6, axis=-1)
        )

        normed = self.norm(x)
        return normed, shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Module):
    """Adaptive Layer Norm producing a single set of shift/scale/gate.

    Used in PixArt-alpha and similar architectures where attention and MLP
    share modulation or only one branch is modulated.

    Args:
        dim: Model dimension.
        cond_dim: Conditioning dimension. Defaults to dim.
    """

    def __init__(self, dim: int, cond_dim: int = 0):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm = nn.LayerNorm(dim, affine=False)
        self.linear = nn.Linear(cond_dim, 3 * dim)

    def __call__(self, x: mx.array, cond: mx.array):
        """Apply adaptive normalization.

        Args:
            x: Input tensor (B, L, D).
            cond: Conditioning tensor (B, cond_dim).

        Returns:
            Tuple of (normed_x, shift, scale, gate).
        """
        modulation = nn.silu(cond)
        modulation = self.linear(modulation)
        if modulation.ndim == 2:
            modulation = mx.expand_dims(modulation, axis=1)

        shift, scale, gate = mx.split(modulation, 3, axis=-1)
        normed = self.norm(x)
        return normed, shift, scale, gate


class AdaLayerNormContinuous(nn.Module):
    """Adaptive Layer Norm with continuous conditioning (SD3-style).

    Unlike Zero/Single variants, this applies the modulation directly
    and returns a single modulated output.

    Args:
        dim: Model dimension.
        cond_dim: Conditioning dimension.
    """

    def __init__(self, dim: int, cond_dim: int = 0):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm = nn.LayerNorm(dim, affine=False)
        self.linear = nn.Linear(cond_dim, 2 * dim)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        """Apply adaptive normalization and return modulated output.

        Args:
            x: Input tensor (B, L, D).
            cond: Conditioning tensor (B, cond_dim).

        Returns:
            Modulated tensor (B, L, D).
        """
        modulation = nn.silu(cond)
        modulation = self.linear(modulation)
        if modulation.ndim == 2:
            modulation = mx.expand_dims(modulation, axis=1)

        shift, scale = mx.split(modulation, 2, axis=-1)
        return self.norm(x) * (1 + scale) + shift

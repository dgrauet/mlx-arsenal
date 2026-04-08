"""ResNet blocks for diffusion models."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class ResnetBlock2d(nn.Module):
    """2D ResNet block with optional timestep conditioning.

    GroupNorm -> SiLU -> Conv -> (+ time_emb) -> GroupNorm -> SiLU -> Conv -> (+ skip)

    Args:
        in_channels: Input channels.
        out_channels: Output channels. Defaults to in_channels.
        time_emb_dim: Timestep embedding dimension. If 0, no conditioning.
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 0,
        time_emb_dim: int = 0,
        num_groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array, time_emb: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: (B, H, W, C) input tensor.
            time_emb: Optional (B, time_emb_dim) timestep embedding.

        Returns:
            (B, H, W, out_channels) output tensor.
        """
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)

        if self.time_proj is not None and time_emb is not None:
            t = nn.silu(time_emb)
            t = self.time_proj(t)
            # Reshape for broadcasting: (B, 1, 1, C)
            h = h + t.reshape(t.shape[0], 1, 1, t.shape[-1])

        h = nn.silu(self.norm2(h))
        h = self.conv2(h)

        if self.skip is not None:
            x = self.skip(x)

        return x + h


class ResnetBlock3d(nn.Module):
    """3D ResNet block with optional timestep conditioning.

    Same structure as ResnetBlock2d but uses Conv3d for video data.

    Args:
        in_channels: Input channels.
        out_channels: Output channels. Defaults to in_channels.
        time_emb_dim: Timestep embedding dimension. If 0, no conditioning.
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 0,
        time_emb_dim: int = 0,
        num_groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array, time_emb: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: (B, D, H, W, C) input tensor (channels-last).
            time_emb: Optional (B, time_emb_dim) timestep embedding.

        Returns:
            (B, D, H, W, out_channels) output tensor.
        """
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)

        if self.time_proj is not None and time_emb is not None:
            t = nn.silu(time_emb)
            t = self.time_proj(t)
            h = h + t.reshape(t.shape[0], 1, 1, 1, t.shape[-1])

        h = nn.silu(self.norm2(h))
        h = self.conv2(h)

        if self.skip is not None:
            x = self.skip(x)

        return x + h

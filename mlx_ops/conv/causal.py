"""Causal convolutions for temporal/sequential data.

Causal convolutions pad only on the left (past) side of the temporal dimension,
ensuring that output at time t depends only on inputs at time <= t.
"""

from typing import Union

import mlx.core as mx
import mlx.nn as nn


def _normalize_tuple(value, n: int, name: str) -> tuple:
    if isinstance(value, int):
        return (value,) * n
    if len(value) != n:
        raise ValueError(f"{name} must have {n} elements, got {len(value)}")
    return tuple(value)


def _causal_pad(x: mx.array, kernel_size: tuple, dims: int, mode: str) -> mx.array:
    """Apply causal padding: full padding on temporal dim (left), symmetric on spatial dims.

    Args:
        x: Input tensor in channels-last format.
            1D: (B, L, C), 2D: (B, H, W, C), 3D: (B, D, H, W, C)
        kernel_size: Kernel size per dimension.
        dims: Number of spatial dimensions (1, 2, or 3).
        mode: Padding mode - "zero", "replicate", or "reflect".
    """
    # Temporal dimension: causal = pad only on left
    temporal_pad = kernel_size[0] - 1

    # Spatial dimensions: symmetric padding
    spatial_pads = [(k // 2, k // 2) for k in kernel_size[1:]]

    # Build pad widths: (batch, temporal, *spatial, channels)
    pad_widths = [(0, 0), (temporal_pad, 0)] + spatial_pads + [(0, 0)]

    if mode == "zero":
        return mx.pad(x, pad_widths)
    elif mode == "replicate":
        # MLX doesn't have replicate padding natively - implement via slicing
        result = x
        # Pad temporal dimension by replicating the first frame
        if temporal_pad > 0:
            first = mx.expand_dims(result[:, 0], axis=1)
            first = mx.repeat(first, temporal_pad, axis=1)
            result = mx.concatenate([first, result], axis=1)
        # Pad spatial dimensions by replicating edges
        for i, (pl, pr) in enumerate(spatial_pads):
            dim = i + 2  # skip batch and temporal
            if pl > 0:
                left = mx.expand_dims(
                    result.swapaxes(0, dim)[0].swapaxes(0, dim - (dim > 0) * 0),
                    axis=dim,
                )
                # Simpler: just use slice and repeat
                slices = [slice(None)] * result.ndim
                slices[dim] = slice(0, 1)
                edge = result[tuple(slices)]
                edge = mx.repeat(edge, pl, axis=dim)
                result = mx.concatenate([edge, result], axis=dim)
            if pr > 0:
                slices = [slice(None)] * result.ndim
                slices[dim] = slice(-1, None)
                edge = result[tuple(slices)]
                edge = mx.repeat(edge, pr, axis=dim)
                result = mx.concatenate([result, edge], axis=dim)
        return result
    elif mode == "reflect":
        return mx.pad(x, pad_widths, mode="reflect")
    else:
        raise ValueError(f"Unknown padding mode: {mode}. Use 'zero', 'replicate', or 'reflect'.")


class CausalConv1d(nn.Module):
    """1D causal convolution.

    Pads the input on the left (past) side only so that the output at
    position t depends only on inputs at positions <= t.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding_mode: "zero", "replicate", or "reflect".
        bias: If True, adds a learnable bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding_mode: str = "zero",
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = (kernel_size,)
        self.stride = stride
        self.padding_mode = padding_mode
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Input shape: (B, L, C)."""
        x = _causal_pad(x, self.kernel_size, dims=1, mode=self.padding_mode)
        return self.conv(x)


class CausalConv2d(nn.Module):
    """2D causal convolution.

    Causal along the first spatial dimension (height/time), symmetric
    padding along the second (width).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel (int or (h, w)).
        stride: Stride of the convolution (int or (h, w)).
        padding_mode: "zero", "replicate", or "reflect".
        bias: If True, adds a learnable bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding_mode: str = "zero",
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = _normalize_tuple(kernel_size, 2, "kernel_size")
        self.stride = _normalize_tuple(stride, 2, "stride")
        self.padding_mode = padding_mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, self.kernel_size,
            stride=self.stride, padding=0, bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Input shape: (B, H, W, C)."""
        x = _causal_pad(x, self.kernel_size, dims=2, mode=self.padding_mode)
        return self.conv(x)


class CausalConv3d(nn.Module):
    """3D causal convolution.

    Causal along the first dimension (temporal/depth), symmetric padding
    along the spatial dimensions (height, width).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel (int or (d, h, w)).
        stride: Stride of the convolution (int or (d, h, w)).
        padding_mode: "zero", "replicate", or "reflect".
        bias: If True, adds a learnable bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding_mode: str = "zero",
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = _normalize_tuple(kernel_size, 3, "kernel_size")
        self.stride = _normalize_tuple(stride, 3, "stride")
        self.padding_mode = padding_mode
        self.conv = nn.Conv3d(
            in_channels, out_channels, self.kernel_size,
            stride=self.stride, padding=0, bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. Input shape: (B, D, H, W, C)."""
        x = _causal_pad(x, self.kernel_size, dims=3, mode=self.padding_mode)
        return self.conv(x)

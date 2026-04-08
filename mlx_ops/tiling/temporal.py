"""Temporal slicing for processing long video sequences.

Processes video tensors in overlapping temporal windows, useful for
VAE encoding/decoding of long videos that don't fit in memory at once.
"""

from typing import Callable

import mlx.core as mx


def _blend_weight_1d(size: int, blend_left: int, blend_right: int) -> mx.array:
    """Create a 1D blending weight: ramp up at left, flat in middle, ramp down at right."""
    w = mx.ones((size,), dtype=mx.float32)
    if blend_left > 0:
        ramp = mx.linspace(0, 1, blend_left + 2)[1:-1]
        w = mx.concatenate([ramp, w[blend_left:]])
    if blend_right > 0:
        ramp = mx.linspace(1, 0, blend_right + 2)[1:-1]
        w = mx.concatenate([w[: size - blend_right], ramp])
    return w


def temporal_slice_process(
    x: mx.array,
    fn: Callable[[mx.array], mx.array],
    window_size: int = 16,
    overlap: int = 4,
    temporal_dim: int = 1,
) -> mx.array:
    """Process a video tensor in overlapping temporal windows with linear blending.

    Args:
        x: Input tensor with a temporal dimension (e.g., (B, T, H, W, C)).
        fn: Processing function applied to each temporal window.
        window_size: Number of frames per window.
        overlap: Number of overlapping frames between windows.
        temporal_dim: Which dimension is temporal.

    Returns:
        Processed tensor with the same temporal length as fn(x) would produce.
    """
    T = x.shape[temporal_dim]

    if T <= window_size:
        return fn(x)

    stride = window_size - overlap

    starts = list(range(0, max(T - window_size, 0) + 1, stride))
    if starts[-1] + window_size < T:
        starts.append(T - window_size)

    # Process first window to get output info
    slices_0 = [slice(None)] * x.ndim
    slices_0[temporal_dim] = slice(starts[0], starts[0] + window_size)
    sample_out = fn(x[tuple(slices_0)])

    scale_t = sample_out.shape[temporal_dim] / window_size
    out_T = int(T * scale_t)

    out_shape = list(sample_out.shape)
    out_shape[temporal_dim] = out_T

    output = mx.zeros(out_shape, dtype=sample_out.dtype)
    weight_sum = mx.zeros(out_shape, dtype=mx.float32)

    for ts in starts:
        slices = [slice(None)] * x.ndim
        slices[temporal_dim] = slice(ts, ts + window_size)
        window_out = fn(x[tuple(slices)])

        wt = window_out.shape[temporal_dim]
        ot = int(ts * scale_t)
        blend = int(overlap * scale_t)

        t_weight = _blend_weight_1d(
            wt,
            blend_left=blend if ts > 0 else 0,
            blend_right=blend if ts + window_size < T else 0,
        )

        # Reshape for broadcasting
        w_shape = [1] * x.ndim
        w_shape[temporal_dim] = wt
        t_weight = t_weight.reshape(w_shape)

        # Pad to full output size and accumulate
        pad_before = ot
        pad_after = out_T - ot - wt

        pad_widths = [(0, 0)] * x.ndim
        pad_widths[temporal_dim] = (pad_before, pad_after)

        padded_out = mx.pad(window_out * t_weight, pad_widths)
        padded_w = mx.pad(t_weight * mx.ones_like(window_out), pad_widths)

        output = output + padded_out
        weight_sum = weight_sum + padded_w

    return output / mx.maximum(weight_sum, 1e-8)

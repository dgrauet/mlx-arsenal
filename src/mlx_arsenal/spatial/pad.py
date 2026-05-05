"""Padding operations missing from MLX core.

MLX's mx.pad only supports constant and reflect modes.
This module adds replicate padding (edge padding).
"""

import mlx.core as mx


def replicate_pad(x: mx.array, pad_widths: list) -> mx.array:
    """Pad a tensor by replicating edge values.

    Equivalent to torch.nn.functional.pad(x, ..., mode="replicate").

    Args:
        x: Input tensor of any shape.
        pad_widths: List of (before, after) padding per dimension.
            Length must match x.ndim.

    Returns:
        Padded tensor.
    """
    result = x
    for dim in range(result.ndim):
        before, after = pad_widths[dim]
        if before > 0:
            slices = [slice(None)] * result.ndim
            slices[dim] = slice(0, 1)
            edge = mx.repeat(result[tuple(slices)], before, axis=dim)
            result = mx.concatenate([edge, result], axis=dim)
        if after > 0:
            slices = [slice(None)] * result.ndim
            slices[dim] = slice(-1, None)
            edge = mx.repeat(result[tuple(slices)], after, axis=dim)
            result = mx.concatenate([result, edge], axis=dim)
    return result

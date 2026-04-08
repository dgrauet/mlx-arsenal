"""Channel layout conversion utilities.

MLX uses channels-last (NHWC/NDHWC) while PyTorch uses channels-first
(NCHW/NCDHW). These utilities handle the conversion.
"""

from contextlib import contextmanager

import mlx.core as mx


def to_channels_last(x: mx.array) -> mx.array:
    """Convert from channels-first to channels-last format.

    Args:
        x: Tensor in channels-first format.
            3D: (B, C, L) -> (B, L, C)
            4D: (B, C, H, W) -> (B, H, W, C)
            5D: (B, C, D, H, W) -> (B, D, H, W, C)

    Returns:
        Tensor in channels-last format.
    """
    if x.ndim == 3:
        return x.transpose(0, 2, 1)
    elif x.ndim == 4:
        return x.transpose(0, 2, 3, 1)
    elif x.ndim == 5:
        return x.transpose(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"Expected 3D-5D tensor, got {x.ndim}D")


def to_channels_first(x: mx.array) -> mx.array:
    """Convert from channels-last to channels-first format.

    Args:
        x: Tensor in channels-last format.
            3D: (B, L, C) -> (B, C, L)
            4D: (B, H, W, C) -> (B, C, H, W)
            5D: (B, D, H, W, C) -> (B, C, D, H, W)

    Returns:
        Tensor in channels-first format.
    """
    if x.ndim == 3:
        return x.transpose(0, 2, 1)
    elif x.ndim == 4:
        return x.transpose(0, 3, 1, 2)
    elif x.ndim == 5:
        return x.transpose(0, 4, 1, 2, 3)
    else:
        raise ValueError(f"Expected 3D-5D tensor, got {x.ndim}D")


@contextmanager
def channels_last(x_ref: list):
    """Context manager that converts a tensor to channels-last on entry
    and back to channels-first on exit.

    Usage::

        ref = [tensor_nchw]
        with channels_last(ref):
            # ref[0] is now in NHWC format
            result = some_mlx_op(ref[0])
            ref[0] = result
        # ref[0] is back in NCHW format

    Args:
        x_ref: Single-element list containing the tensor. Modified in-place.
    """
    x_ref[0] = to_channels_last(x_ref[0])
    try:
        yield
    finally:
        x_ref[0] = to_channels_first(x_ref[0])

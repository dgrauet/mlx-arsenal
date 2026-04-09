"""Interpolation operations missing from MLX core.

Equivalent to torch.nn.functional.interpolate for various modes.
"""

import mlx.core as mx


def interpolate_nearest(
    x: mx.array,
    size: tuple = None,
    scale_factor: float = None,
) -> mx.array:
    """Nearest-neighbor interpolation for N-dimensional spatial tensors.

    Supports 3D (B,L,C), 4D (B,H,W,C), and 5D (B,D,H,W,C) inputs.
    Operates on all spatial dims (all except batch and channel).

    Args:
        x: Input tensor in channels-last format.
        size: Target spatial size. Mutually exclusive with scale_factor.
        scale_factor: Scale factor for all spatial dims.

    Returns:
        Interpolated tensor.
    """
    ndim = x.ndim
    spatial_dims = list(range(1, ndim - 1))

    if size is not None and scale_factor is not None:
        raise ValueError("Only one of size or scale_factor can be specified")

    if size is None and scale_factor is None:
        return x

    # Compute target sizes
    if size is not None:
        if isinstance(size, int):
            target = [size] * len(spatial_dims)
        else:
            target = list(size)
    else:
        target = [int(x.shape[d] * scale_factor) for d in spatial_dims]

    # Apply nearest-neighbor resize per dimension
    result = x
    for i, dim in enumerate(spatial_dims):
        src_size = result.shape[dim]
        tgt_size = target[i]
        if src_size == tgt_size:
            continue
        indices = mx.clip(
            (mx.arange(tgt_size) * src_size) // tgt_size,
            0, src_size - 1,
        )
        # Gather along this dimension
        slices = [slice(None)] * result.ndim
        slices[dim] = indices
        result = result[tuple(slices)]

    return result


def interpolate_3d(
    x: mx.array,
    size: tuple,
) -> mx.array:
    """Nearest-neighbor interpolation for 5D (B,D,H,W,C) tensors.

    Convenience wrapper around interpolate_nearest for video tensors.

    Args:
        x: (B, D, H, W, C) input tensor.
        size: Target (D, H, W).

    Returns:
        (B, target_D, target_H, target_W, C) interpolated tensor.
    """
    return interpolate_nearest(x, size=size)


def avg_pool1d(
    x: mx.array,
    kernel_size: int,
    stride: int = None,
) -> mx.array:
    """1D average pooling.

    Equivalent to torch.nn.functional.avg_pool1d.

    Args:
        x: (B, L, C) input tensor.
        kernel_size: Pooling window size.
        stride: Pooling stride. Defaults to kernel_size.

    Returns:
        (B, L_out, C) pooled tensor where L_out = (L - kernel_size) // stride + 1.
    """
    stride = stride or kernel_size
    B, L, C = x.shape
    L_out = (L - kernel_size) // stride + 1

    # Reshape to windows and average
    windows = []
    for i in range(L_out):
        start = i * stride
        windows.append(x[:, start:start + kernel_size])

    # Stack and mean: (B, L_out, kernel_size, C) -> (B, L_out, C)
    stacked = mx.stack(windows, axis=1)
    return mx.mean(stacked, axis=2)

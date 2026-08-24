"""Upsampling operations."""

import mlx.core as mx

from mlx_arsenal.spatial.interpolate import interpolate_nearest


def upsample_nearest(x: mx.array, scale_factor: int = 2) -> mx.array:
    """Nearest-neighbor upsampling for spatial tensors.

    Thin wrapper over :func:`mlx_arsenal.spatial.interpolate_nearest` that
    additionally rejects non-4D/5D inputs.

    Args:
        x: Input tensor (B, H, W, C) or (B, D, H, W, C).
        scale_factor: Integer upsampling factor.

    Returns:
        Upsampled tensor.

    Raises:
        ValueError: if ``x`` is not 4D or 5D.
    """
    if x.ndim not in (4, 5):
        raise ValueError(f"Expected 4D or 5D input, got {x.ndim}D")
    return interpolate_nearest(x, scale_factor=float(scale_factor))


def upsample_bilinear(x: mx.array, scale_factor: int = 2) -> mx.array:
    """Bilinear upsampling for 2D spatial tensors (B, H, W, C).

    Uses the formula: output[i,j] = weighted average of 4 nearest input pixels.

    Args:
        x: (B, H, W, C) input tensor.
        scale_factor: Integer upsampling factor.

    Returns:
        (B, H*scale_factor, W*scale_factor, C) upsampled tensor.
    """
    if x.ndim != 4:
        raise ValueError(f"Bilinear upsampling requires 4D input, got {x.ndim}D")

    B, H, W, C = x.shape
    new_h = H * scale_factor
    new_w = W * scale_factor

    # Compute source coordinates
    # Map output pixel centers to input coordinate space
    row_coords = (mx.arange(new_h, dtype=mx.float32) + 0.5) / scale_factor - 0.5
    col_coords = (mx.arange(new_w, dtype=mx.float32) + 0.5) / scale_factor - 0.5

    row_coords = mx.clip(row_coords, 0, H - 1)
    col_coords = mx.clip(col_coords, 0, W - 1)

    r0 = mx.floor(row_coords).astype(mx.int32)
    c0 = mx.floor(col_coords).astype(mx.int32)
    r1 = mx.minimum(r0 + 1, H - 1)
    c1 = mx.minimum(c0 + 1, W - 1)

    dr = mx.expand_dims(row_coords - r0.astype(mx.float32), 1)  # (new_h, 1)
    dc = mx.expand_dims(col_coords - c0.astype(mx.float32), 0)  # (1, new_w)

    # Gather corners: x[:, r, c, :] for each combination
    # Use advanced indexing
    r0 = mx.expand_dims(r0, 1)  # (new_h, 1)
    r1 = mx.expand_dims(r1, 1)
    c0 = mx.expand_dims(c0, 0)  # (1, new_w)
    c1 = mx.expand_dims(c1, 0)

    top_left = x[:, r0, c0, :]  # (B, new_h, new_w, C)
    top_right = x[:, r0, c1, :]
    bottom_left = x[:, r1, c0, :]
    bottom_right = x[:, r1, c1, :]

    dr = dr.reshape(1, new_h, 1, 1)
    dc = dc.reshape(1, 1, new_w, 1)

    top = top_left * (1 - dc) + top_right * dc
    bottom = bottom_left * (1 - dc) + bottom_right * dc
    return top * (1 - dr) + bottom * dr

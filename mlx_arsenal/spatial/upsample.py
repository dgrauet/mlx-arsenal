"""Upsampling operations."""

import mlx.core as mx


def upsample_nearest(x: mx.array, scale_factor: int = 2) -> mx.array:
    """Nearest-neighbor upsampling for spatial tensors.

    Args:
        x: Input tensor (B, H, W, C) or (B, D, H, W, C).
        scale_factor: Integer upsampling factor.

    Returns:
        Upsampled tensor.
    """
    if x.ndim == 4:
        B, H, W, C = x.shape
        x = mx.expand_dims(x, axis=2)  # (B, H, 1, W, C)
        x = mx.repeat(x, scale_factor, axis=2)  # (B, H, s, W, C)
        x = x.reshape(B, H * scale_factor, W, C)
        x = mx.expand_dims(x, axis=3)  # (B, H*s, W, 1, C)
        x = mx.repeat(x, scale_factor, axis=3)  # (B, H*s, W, s, C)
        return x.reshape(B, H * scale_factor, W * scale_factor, C)

    elif x.ndim == 5:
        B, D, H, W, C = x.shape
        # Upsample depth
        x = mx.expand_dims(x, axis=2)
        x = mx.repeat(x, scale_factor, axis=2)
        x = x.reshape(B, D * scale_factor, H, W, C)
        # Upsample height
        x = mx.expand_dims(x, axis=3)
        x = mx.repeat(x, scale_factor, axis=3)
        x = x.reshape(B, D * scale_factor, H * scale_factor, W, C)
        # Upsample width
        x = mx.expand_dims(x, axis=4)
        x = mx.repeat(x, scale_factor, axis=4)
        return x.reshape(B, D * scale_factor, H * scale_factor, W * scale_factor, C)

    else:
        raise ValueError(f"Expected 4D or 5D input, got {x.ndim}D")


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

    top_left = x[:, r0, c0, :]       # (B, new_h, new_w, C)
    top_right = x[:, r0, c1, :]
    bottom_left = x[:, r1, c0, :]
    bottom_right = x[:, r1, c1, :]

    dr = dr.reshape(1, new_h, 1, 1)
    dc = dc.reshape(1, 1, new_w, 1)

    top = top_left * (1 - dc) + top_right * dc
    bottom = bottom_left * (1 - dc) + bottom_right * dc
    return top * (1 - dr) + bottom * dr

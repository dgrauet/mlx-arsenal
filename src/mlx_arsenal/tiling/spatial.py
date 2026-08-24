"""Tiled processing for large spatial tensors.

Splits tensors into overlapping tiles, processes each tile independently,
and blends them back together. Useful for VAE encoding/decoding of
high-resolution images/video that exceed memory limits.
"""

from collections.abc import Callable

import mlx.core as mx

from mlx_arsenal.tiling._blend import blend_weight_1d, window_starts


def tiled_process(
    x: mx.array,
    fn: Callable[[mx.array], mx.array],
    tile_size: int = 512,
    overlap: int = 64,
    spatial_dims: tuple[int, int] = (1, 2),
) -> mx.array:
    """Process a tensor in overlapping spatial tiles with linear blending.

    Args:
        x: Input tensor. Spatial dimensions specified by spatial_dims.
        fn: Processing function applied to each tile.
        tile_size: Size of each tile along spatial dimensions.
        overlap: Overlap between adjacent tiles for blending.
        spatial_dims: Which dimensions are spatial (default: H, W for BHWC).

    Returns:
        Processed tensor, same shape as fn(x) would produce if memory
        were unlimited.
    """
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

    shape = x.shape
    dim_h, dim_w = spatial_dims
    H, W = shape[dim_h], shape[dim_w]

    if H <= tile_size and W <= tile_size:
        return fn(x)

    stride = tile_size - overlap

    h_starts = window_starts(H, tile_size, stride)
    w_starts = window_starts(W, tile_size, stride)

    # Process first tile to determine output shape/dtype
    slices_0 = [slice(None)] * x.ndim
    slices_0[dim_h] = slice(h_starts[0], h_starts[0] + tile_size)
    slices_0[dim_w] = slice(w_starts[0], w_starts[0] + tile_size)
    sample_out = fn(x[tuple(slices_0)])

    # The first tile's input slice is min(tile_size, dim) long — scaling by
    # tile_size alone breaks any axis that fits inside a single tile.
    scale_h = sample_out.shape[dim_h] / min(tile_size, H)
    scale_w = sample_out.shape[dim_w] / min(tile_size, W)

    out_H = int(H * scale_h)
    out_W = int(W * scale_w)
    out_shape = list(sample_out.shape)
    out_shape[dim_h] = out_H
    out_shape[dim_w] = out_W

    output = mx.zeros(out_shape, dtype=sample_out.dtype)
    weight_sum = mx.zeros(out_shape, dtype=mx.float32)

    for hs in h_starts:
        for ws in w_starts:
            slices = [slice(None)] * x.ndim
            slices[dim_h] = slice(hs, hs + tile_size)
            slices[dim_w] = slice(ws, ws + tile_size)
            tile_out = fn(x[tuple(slices)])

            oh = int(hs * scale_h)
            ow = int(ws * scale_w)
            th = tile_out.shape[dim_h]
            tw = tile_out.shape[dim_w]

            blend_h = int(overlap * scale_h)
            blend_w = int(overlap * scale_w)

            h_w = blend_weight_1d(
                th,
                blend_left=blend_h if hs > 0 else 0,
                blend_right=blend_h if hs + tile_size < H else 0,
            )
            w_w = blend_weight_1d(
                tw,
                blend_left=blend_w if ws > 0 else 0,
                blend_right=blend_w if ws + tile_size < W else 0,
            )

            # 2D weight via outer product, reshaped for broadcasting
            tile_w = mx.expand_dims(h_w, 1) * mx.expand_dims(w_w, 0)
            bcast_shape = [1] * x.ndim
            bcast_shape[dim_h] = th
            bcast_shape[dim_w] = tw
            tile_w = tile_w.reshape(bcast_shape)

            # Build full-size arrays with zeros and add
            pad_before_h = oh
            pad_after_h = out_H - oh - th
            pad_before_w = ow
            pad_after_w = out_W - ow - tw

            pad_widths = [(0, 0)] * x.ndim
            pad_widths[dim_h] = (pad_before_h, pad_after_h)
            pad_widths[dim_w] = (pad_before_w, pad_after_w)

            padded_tile = mx.pad(tile_out * tile_w, pad_widths)
            padded_weight = mx.pad(tile_w * mx.ones_like(tile_out), pad_widths)

            output = output + padded_tile
            weight_sum = weight_sum + padded_weight

    return output / mx.maximum(weight_sum, 1e-8)

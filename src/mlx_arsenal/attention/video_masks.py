"""Spatiotemporal attention masks for video diffusion transformers.

All functions in this module assume **T-major** token flattening: a video
tensor of shape `(T, H, W)` is flattened to `S = T*H*W` tokens in the order
`[t0(h0w0..hHwW), t1(...), ..., tT(...)]`. This matches LTX-Video,
CogVideoX, and the convention used by `mlx_arsenal.spatial.patchify`.

Each function returns a mask of shape `(1, 1, S, S)` with float values:
`0.0` means the query is allowed to attend to the key, `-inf` means it is
blocked. The shape broadcasts over batch and head axes expected by
`mx.fast.scaled_dot_product_attention`.

For typical LTX latents (`T=8, H=32, W=32 → S=8192`) the mask is
`S² ≈ 67M` entries. Use `dtype=mx.float16` to halve memory.
"""

import mlx.core as mx

from mlx_arsenal.attention._thw import thw_coords as _thw_coords
from mlx_arsenal.attention._thw import validate_thw as _validate_thw


def _additive(valid: mx.array, dtype: mx.Dtype) -> mx.array:
    """Convert a bool (S, S) mask to additive float (1, 1, S, S)."""
    S = valid.shape[0]
    out = mx.where(
        valid,
        mx.zeros((S, S), dtype=dtype),
        mx.full((S, S), float("-inf"), dtype=dtype),
    )
    return out.reshape(1, 1, S, S)


def spatial_only_mask(T: int, H: int, W: int, *, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Mask that restricts attention to tokens in the same frame.

    Each token at frame `t` attends only to other tokens whose frame index
    equals `t`. Captures the "spatial-locality" head pattern from Sparse
    VideoGen.

    Args:
        T: Number of frames.
        H: Latent height.
        W: Latent width.
        dtype: Output dtype.

    Returns:
        Additive mask of shape `(1, 1, T*H*W, T*H*W)`.
    """
    _validate_thw(T, H, W)
    t_flat, _, _ = _thw_coords(T, H, W)
    valid = mx.equal(mx.expand_dims(t_flat, 0), mx.expand_dims(t_flat, 1))
    return _additive(valid, dtype)


def temporal_only_mask(T: int, H: int, W: int, *, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Mask that restricts attention to tokens at the same spatial position.

    Each token at `(h, w)` attends only to tokens whose `(h, w)` matches,
    across all frames. Captures the "temporal-locality" head pattern from
    Sparse VideoGen.

    Args:
        T: Number of frames.
        H: Latent height.
        W: Latent width.
        dtype: Output dtype.

    Returns:
        Additive mask of shape `(1, 1, T*H*W, T*H*W)`.
    """
    _validate_thw(T, H, W)
    _, h_flat, w_flat = _thw_coords(T, H, W)
    same_h = mx.equal(mx.expand_dims(h_flat, 0), mx.expand_dims(h_flat, 1))
    same_w = mx.equal(mx.expand_dims(w_flat, 0), mx.expand_dims(w_flat, 1))
    valid = mx.logical_and(same_h, same_w)
    return _additive(valid, dtype)


def sliding_tile_centered_mask(
    T: int,
    H: int,
    W: int,
    *,
    window: tuple[int, int, int],
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Per-query centered spatiotemporal window mask.

    Token `(t, h, w)` attends to `(t', h', w')` iff
    `|t-t'| <= window[0]` AND `|h-h'| <= window[1]` AND `|w-w'| <= window[2]`.

    Args:
        T: Number of frames.
        H: Latent height.
        W: Latent width.
        window: `(dt, dh, dw)` non-negative inclusive radii.
        dtype: Output dtype.

    Returns:
        Additive mask of shape `(1, 1, T*H*W, T*H*W)`.
    """
    _validate_thw(T, H, W)
    if any(w < 0 for w in window):
        raise ValueError(f"window radii must be non-negative, got {window}")
    t_flat, h_flat, w_flat = _thw_coords(T, H, W)
    dt = mx.abs(mx.expand_dims(t_flat, 0) - mx.expand_dims(t_flat, 1))
    dh = mx.abs(mx.expand_dims(h_flat, 0) - mx.expand_dims(h_flat, 1))
    dw = mx.abs(mx.expand_dims(w_flat, 0) - mx.expand_dims(w_flat, 1))
    in_t = mx.less_equal(dt, window[0])
    in_h = mx.less_equal(dh, window[1])
    in_w = mx.less_equal(dw, window[2])
    valid = mx.logical_and(mx.logical_and(in_t, in_h), in_w)
    return _additive(valid, dtype)


def sliding_tile_block_mask(
    T: int,
    H: int,
    W: int,
    *,
    tile: tuple[int, int, int],
    window: tuple[int, int, int] = (1, 1, 1),
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Tile-block sliding attention (STA, ICML 2025).

    Tokens are grouped into non-overlapping tiles of shape
    `tile = (tt, th, tw)`. Every query in a tile attends to all keys in the
    `±window` neighboring tiles (window in tile units, inclusive).

    Args:
        T: Number of frames. Must be divisible by `tile[0]`.
        H: Latent height. Must be divisible by `tile[1]`.
        W: Latent width. Must be divisible by `tile[2]`.
        tile: `(tt, th, tw)` tile dims, all positive.
        window: `(wt, wh, ww)` non-negative tile-unit radii.
        dtype: Output dtype.

    Returns:
        Additive mask of shape `(1, 1, T*H*W, T*H*W)`.
    """
    _validate_thw(T, H, W)
    tt, th, tw = tile
    if tt <= 0 or th <= 0 or tw <= 0:
        raise ValueError(f"tile dims must be positive, got {tile}")
    if T % tt or H % th or W % tw:
        raise ValueError(f"(T, H, W)={(T, H, W)} not divisible by tile={tile}")
    if any(w < 0 for w in window):
        raise ValueError(f"window radii must be non-negative, got {window}")
    t_flat, h_flat, w_flat = _thw_coords(T, H, W)
    t_tile = mx.floor_divide(t_flat, tt)
    h_tile = mx.floor_divide(h_flat, th)
    w_tile = mx.floor_divide(w_flat, tw)
    dt = mx.abs(mx.expand_dims(t_tile, 0) - mx.expand_dims(t_tile, 1))
    dh = mx.abs(mx.expand_dims(h_tile, 0) - mx.expand_dims(h_tile, 1))
    dw = mx.abs(mx.expand_dims(w_tile, 0) - mx.expand_dims(w_tile, 1))
    in_t = mx.less_equal(dt, window[0])
    in_h = mx.less_equal(dh, window[1])
    in_w = mx.less_equal(dw, window[2])
    valid = mx.logical_and(mx.logical_and(in_t, in_h), in_w)
    return _additive(valid, dtype)


def radial_box_mask(
    T: int,
    H: int,
    W: int,
    *,
    radius_t: int,
    radius_s: float,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Hard-cutoff radial spatiotemporal mask.

    Query `(t, h, w)` attends to `(t', h', w')` iff
    `|t-t'| <= radius_t` AND `sqrt((h-h')**2 + (w-w')**2) <= radius_s`.

    Args:
        T: Number of frames.
        H: Latent height.
        W: Latent width.
        radius_t: Non-negative temporal radius (frames, inclusive).
        radius_s: Non-negative Euclidean spatial radius (latent units).
        dtype: Output dtype.

    Returns:
        Additive mask of shape `(1, 1, T*H*W, T*H*W)`.
    """
    _validate_thw(T, H, W)
    if radius_t < 0:
        raise ValueError(f"radius_t must be non-negative, got {radius_t}")
    if radius_s < 0:
        raise ValueError(f"radius_s must be non-negative, got {radius_s}")
    t_flat, h_flat, w_flat = _thw_coords(T, H, W)
    dt = mx.abs(mx.expand_dims(t_flat, 0) - mx.expand_dims(t_flat, 1))
    dh = mx.expand_dims(h_flat, 0) - mx.expand_dims(h_flat, 1)
    dw = mx.expand_dims(w_flat, 0) - mx.expand_dims(w_flat, 1)
    ds_sq = (dh * dh + dw * dw).astype(mx.float32)
    in_t = mx.less_equal(dt, radius_t)
    in_s = mx.less_equal(ds_sq, radius_s * radius_s)
    valid = mx.logical_and(in_t, in_s)
    return _additive(valid, dtype)


def frame_stride_diagonal_mask(
    T: int,
    H: int,
    W: int,
    *,
    num_diagonals: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Multi-diagonal mask at frame-stride offsets (Sparse-vDiT M3).

    Token at flat index `i` attends to token at flat index `j` iff
    `(j - i)` is a multiple of the per-frame stride `H*W` in
    `{-(k-1)*HW, ..., -HW, 0, HW, ..., (k-1)*HW}` where `k = num_diagonals`.
    Captures the "multi-diagonal" head pattern from Sparse-vDiT
    (Chen et al. 2025): same `(h, w)` across nearby frames.

    Setting `num_diagonals=1` reduces to the main diagonal (self-attention
    only). Setting `num_diagonals=T` is equivalent to ``temporal_only_mask``.

    Args:
        T: Number of frames.
        H: Latent height.
        W: Latent width.
        num_diagonals: Strictly positive number of diagonal bands
            (counting the main diagonal once).
        dtype: Output dtype.

    Returns:
        Additive mask of shape `(1, 1, T*H*W, T*H*W)`.
    """
    _validate_thw(T, H, W)
    if num_diagonals <= 0:
        raise ValueError(f"num_diagonals must be positive, got {num_diagonals}")
    S = T * H * W
    HW = H * W
    i = mx.arange(S).reshape(S, 1)
    j = mx.arange(S).reshape(1, S)
    offset = j - i
    # Same (h, w) within a frame iff |offset| % HW == 0.
    abs_off = mx.abs(offset)
    on_band = mx.equal(mx.remainder(abs_off, HW), 0)
    within_range = mx.less_equal(abs_off, (num_diagonals - 1) * HW)
    valid = mx.logical_and(on_band, within_range)
    return _additive(valid, dtype)


def vertical_stripe_mask(
    T: int,
    H: int,
    W: int,
    *,
    key_indices: mx.array,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Anchor-column mask (Sparse-vDiT M4).

    Every query attends only to a fixed set of "sink" key tokens identified
    by ``key_indices`` (flat indices into the T-major sequence). Captures
    the "vertical-stripe" head pattern from Sparse-vDiT, where a small set
    of anchor positions act as global memory.

    The set must be non-empty and contain unique in-range indices. The
    main diagonal is *not* added automatically — include it in
    ``key_indices`` if self-attention is desired.

    Args:
        T: Number of frames.
        H: Latent height.
        W: Latent width.
        key_indices: 1-D ``mx.array`` of int indices in `[0, T*H*W)`.
        dtype: Output dtype.

    Returns:
        Additive mask of shape `(1, 1, T*H*W, T*H*W)`.
    """
    _validate_thw(T, H, W)
    if key_indices.ndim != 1:
        raise ValueError(f"key_indices must be 1-D, got shape {key_indices.shape}")
    if key_indices.size == 0:
        raise ValueError("key_indices must be non-empty")
    S = T * H * W
    keys = key_indices.astype(mx.int32)
    if mx.any(mx.less(keys, 0)).item() or mx.any(mx.greater_equal(keys, S)).item():
        raise ValueError(f"key_indices out of range [0, {S})")
    valid_row = mx.zeros((S,), dtype=mx.bool_)
    valid_row[keys] = mx.array(True)
    valid = mx.broadcast_to(valid_row.reshape(1, S), (S, S))
    return _additive(valid, dtype)


def radial_gaussian_mask(
    T: int,
    H: int,
    W: int,
    *,
    sigma_t: float,
    sigma_s: float,
    cutoff: float = -6.0,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Exponential-decay radial mask (dense log-weights).

    Value at `(i, j)` is `-(dt**2 / (2 sigma_t**2) + ds**2 / (2 sigma_s**2))`
    where `ds**2 = (h-h')**2 + (w-w')**2`. Values below `cutoff` are clamped
    to `-inf` so the mask is usable in fp16 without underflow.

    Args:
        T: Number of frames.
        H: Latent height.
        W: Latent width.
        sigma_t: Temporal scale, strictly positive.
        sigma_s: Spatial scale, strictly positive.
        cutoff: Strictly negative log-weight floor; values below are replaced
            by `-inf`. Default `-6.0` ≈ exp(-6) ≈ 0.0025.
        dtype: Output dtype.

    Returns:
        Additive mask of shape `(1, 1, T*H*W, T*H*W)`.
    """
    _validate_thw(T, H, W)
    if sigma_t <= 0:
        raise ValueError(f"sigma_t must be positive, got {sigma_t}")
    if sigma_s <= 0:
        raise ValueError(f"sigma_s must be positive, got {sigma_s}")
    if cutoff >= 0:
        raise ValueError(f"cutoff must be negative, got {cutoff}")
    t_flat, h_flat, w_flat = _thw_coords(T, H, W)
    dt = (mx.expand_dims(t_flat, 0) - mx.expand_dims(t_flat, 1)).astype(mx.float32)
    dh = (mx.expand_dims(h_flat, 0) - mx.expand_dims(h_flat, 1)).astype(mx.float32)
    dw = (mx.expand_dims(w_flat, 0) - mx.expand_dims(w_flat, 1)).astype(mx.float32)
    log_w = -(dt * dt / (2 * sigma_t * sigma_t) + (dh * dh + dw * dw) / (2 * sigma_s * sigma_s))
    log_w = log_w.astype(dtype)
    S = T * H * W
    neg_inf = mx.full((S, S), float("-inf"), dtype=dtype)
    out = mx.where(mx.less(log_w, cutoff), neg_inf, log_w)
    return out.reshape(1, 1, S, S)

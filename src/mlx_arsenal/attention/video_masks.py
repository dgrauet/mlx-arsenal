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


def _validate_thw(T: int, H: int, W: int) -> None:
    if T <= 0 or H <= 0 or W <= 0:
        raise ValueError(f"T, H, W must be positive, got T={T}, H={H}, W={W}")


def _thw_coords(T: int, H: int, W: int) -> tuple[mx.array, mx.array, mx.array]:
    """Per-token (t, h, w) coordinate vectors of length S=T*H*W (T-major)."""
    t_idx = mx.arange(T).reshape(T, 1, 1)
    h_idx = mx.arange(H).reshape(1, H, 1)
    w_idx = mx.arange(W).reshape(1, 1, W)
    t_flat = mx.broadcast_to(t_idx, (T, H, W)).reshape(-1)
    h_flat = mx.broadcast_to(h_idx, (T, H, W)).reshape(-1)
    w_flat = mx.broadcast_to(w_idx, (T, H, W)).reshape(-1)
    return t_flat, h_flat, w_flat


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

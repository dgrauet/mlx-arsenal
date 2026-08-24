"""Shared (T, H, W) token-grid helpers for video attention utilities."""

from __future__ import annotations

import mlx.core as mx


def validate_thw(T: int, H: int, W: int) -> None:
    if T <= 0 or H <= 0 or W <= 0:
        raise ValueError(f"T, H, W must be positive, got T={T}, H={H}, W={W}")


def thw_coords(T: int, H: int, W: int) -> tuple[mx.array, mx.array, mx.array]:
    """Per-token (t, h, w) coordinate vectors of length S=T*H*W (T-major)."""
    t_idx = mx.arange(T).reshape(T, 1, 1)
    h_idx = mx.arange(H).reshape(1, H, 1)
    w_idx = mx.arange(W).reshape(1, 1, W)
    t_flat = mx.broadcast_to(t_idx, (T, H, W)).reshape(-1)
    h_flat = mx.broadcast_to(h_idx, (T, H, W)).reshape(-1)
    w_flat = mx.broadcast_to(w_idx, (T, H, W)).reshape(-1)
    return t_flat, h_flat, w_flat

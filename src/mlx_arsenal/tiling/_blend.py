"""Shared helpers for overlapped-window processing with linear blending."""

from __future__ import annotations

import mlx.core as mx


def blend_weight_1d(size: int, blend_left: int, blend_right: int) -> mx.array:
    """Create a 1D blending weight: ramp up at left, flat in middle, ramp down at right."""
    w = mx.ones((size,), dtype=mx.float32)
    if blend_left > 0:
        ramp = mx.linspace(0, 1, blend_left + 2)[1:-1]  # exclude 0 and 1
        w = mx.concatenate([ramp, w[blend_left:]])
    if blend_right > 0:
        ramp = mx.linspace(1, 0, blend_right + 2)[1:-1]
        w = mx.concatenate([w[: size - blend_right], ramp])
    return w


def window_starts(total: int, window: int, stride: int) -> list[int]:
    """Start offsets of ``window``-sized windows covering ``[0, total)``.

    Strided from 0; a final window ending exactly at ``total`` is appended
    when the strided grid does not already reach it.
    """
    starts = list(range(0, max(total - window, 0) + 1, stride))
    if starts[-1] + window < total:
        starts.append(total - window)
    return starts

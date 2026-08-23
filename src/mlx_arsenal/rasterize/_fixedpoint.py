"""Fixed-point screen-space setup for the tiled rasterizer.

Screen coordinates are snapped to a 1/16-pixel grid and held as int32 so that
edge functions and signed areas can be evaluated exactly in int64. Exactness is
what makes coverage watertight: two triangles sharing an edge evaluate that
edge from identical integer endpoints, so they cannot disagree about it.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_arsenal._typing import item_int

SUBPIXEL_BITS = 4
SUBPIXEL_SCALE = 1 << SUBPIXEL_BITS
MAX_DIM = 16384

__all__ = ["MAX_DIM", "SUBPIXEL_BITS", "SUBPIXEL_SCALE", "to_screen"]


def to_screen(vertices: mx.array, width: int, height: int) -> tuple[mx.array, mx.array]:
    """Project clip-space vertices to fixed-point screen space.

    Args:
        vertices: (N, 4) clip-space homogeneous coordinates.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        verts_fx: (N, 2) int32 screen x/y on the 1/16-pixel grid.
        verts_zw: (N, 2) float32 depth and clip w, kept for the depth test and
            the perspective-correct barycentric division.

    Raises:
        ValueError: if width or height exceeds the fixed-point range.
    """
    if width > MAX_DIM or height > MAX_DIM:
        raise ValueError(
            f"rasterize supports images up to {MAX_DIM}px per axis "
            f"(fixed-point range with {SUBPIXEL_BITS} sub-pixel bits); "
            f"got {width}x{height}"
        )

    v = vertices.astype(mx.float32)
    w_clip = v[:, 3:4]
    x = (v[:, 0:1] / w_clip * 0.5 + 0.5) * (width - 1) + 0.5
    y = (0.5 + 0.5 * v[:, 1:2] / w_clip) * (height - 1) + 0.5
    z = v[:, 2:3] / w_clip * 0.49999 + 0.5

    # A vertex with w == 0 (or any NaN clip-space input) gives x/w = NaN. Inf
    # and int32-saturated coordinates are caught downstream by the magnitude
    # guard in rasterize_triangles, but a NaN rounds and casts to fixed-point
    # 0, landing inside every bound: it would rasterize a bogus triangle with
    # no error, and NaN comparisons are false so the depth test would not
    # reject it either. z needs its own term: a NaN z with finite x/y (a NaN
    # fed straight into the z component) rasterizes at the right pixels but
    # wins every depth comparison, silently overwriting closer geometry.
    # Catch all three here, before the coordinates disappear into the cast.
    if item_int(mx.any(mx.isnan(x)) | mx.any(mx.isnan(y)) | mx.any(mx.isnan(z))):
        raise ValueError(
            "projected vertex coordinates are not finite (NaN); this most "
            "likely means a vertex has w == 0 in clip space. Clip triangles "
            "against the near plane before rasterizing."
        )

    fx = mx.round(x * SUBPIXEL_SCALE).astype(mx.int32)
    fy = mx.round(y * SUBPIXEL_SCALE).astype(mx.int32)
    return mx.concatenate([fx, fy], axis=1), mx.concatenate([z, w_clip], axis=1)

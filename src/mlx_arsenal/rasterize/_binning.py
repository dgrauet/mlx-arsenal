"""Tile binning for the rasterizer: which faces can touch which tiles.

Stages 1-3 of the pipeline. Everything here is exact integer arithmetic, so
culling and degeneracy are decided by sign and nullity rather than by epsilon.
"""

from __future__ import annotations

import mlx.core as mx

from ._fixedpoint import SUBPIXEL_SCALE

__all__ = ["face_spans", "signed_area"]


def signed_area(verts_fx: mx.array, faces: mx.array) -> mx.array:
    """Exact doubled signed area per face, in fixed-point units.

    Args:
        verts_fx: (N, 2) int32 fixed-point screen coordinates.
        faces: (F, 3) int32 vertex indices.

    Returns:
        (F,) int64. Sign gives orientation, zero means degenerate.
    """
    p = verts_fx.astype(mx.int64)[faces]  # (F, 3, 2)
    ax, ay = p[:, 0, 0], p[:, 0, 1]
    bx, by = p[:, 1, 0], p[:, 1, 1]
    cx, cy = p[:, 2, 0], p[:, 2, 1]
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def face_spans(
    verts_fx: mx.array,
    faces: mx.array,
    width: int,
    height: int,
    tile_size: int,
    cull: str,
) -> tuple[mx.array, mx.array]:
    """Tile range and tile count for each face.

    Args:
        verts_fx: (N, 2) int32 fixed-point screen coordinates.
        faces: (F, 3) int32 vertex indices.
        width: Image width in pixels.
        height: Image height in pixels.
        tile_size: Binning tile edge in pixels.
        cull: One of ``"none"``, ``"back"``, ``"front"``.

    Returns:
        tile_bounds: (F, 4) int32 ``[tx0, ty0, tx1, ty1]``, inclusive.
        n_tiles: (F,) int32, zero for faces that cannot contribute.
    """
    p = verts_fx[faces]  # (F, 3, 2)
    lo = mx.min(p, axis=1)  # (F, 2)
    hi = mx.max(p, axis=1)

    # Fixed-point bounds -> pixel indices whose centres could be covered.
    half = SUBPIXEL_SCALE // 2
    px0 = mx.floor_divide(lo - half + SUBPIXEL_SCALE - 1, SUBPIXEL_SCALE)
    px1 = mx.floor_divide(hi - half, SUBPIXEL_SCALE)

    max_xy = mx.array([width - 1, height - 1], dtype=mx.int32)
    px0 = mx.clip(px0, mx.array(0, dtype=mx.int32), max_xy)
    px1 = mx.clip(px1, mx.array(0, dtype=mx.int32), max_xy)

    onscreen = mx.all(mx.logical_and(hi >= half, lo <= max_xy * SUBPIXEL_SCALE + half), axis=1)

    t0 = mx.floor_divide(px0, tile_size)
    t1 = mx.floor_divide(px1, tile_size)
    tile_bounds = mx.concatenate([t0, t1], axis=1).astype(mx.int32)  # [tx0,ty0,tx1,ty1]

    area = signed_area(verts_fx, faces)
    # mx.array.__ne__ types as `array | bool`; mx.not_equal keeps `keep` a plain array.
    keep = mx.not_equal(area, 0)
    if cull == "back":
        keep = mx.logical_and(keep, area > 0)
    elif cull == "front":
        keep = mx.logical_and(keep, area < 0)
    keep = mx.logical_and(mx.logical_and(keep, onscreen), mx.all(px1 >= px0, axis=1))

    span = t1 - t0 + 1  # (F, 2)
    n_tiles = (span[:, 0] * span[:, 1] * keep.astype(mx.int32)).astype(mx.int32)
    return tile_bounds, n_tiles

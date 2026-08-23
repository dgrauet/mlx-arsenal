"""Tile binning for the rasterizer: which faces can touch which tiles.

Stages 1-3 of the pipeline. Everything here is exact integer arithmetic, so
culling and degeneracy are decided by sign and nullity rather than by epsilon.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_arsenal._typing import item_int

from ._fixedpoint import SUBPIXEL_BITS, SUBPIXEL_SCALE

__all__ = ["MAX_PAIRS", "TILE_SIZES", "choose_tiling", "face_spans", "signed_area"]

MAX_PAIRS = 32 * 1024 * 1024  # 256 MB of int64 keys
TILE_SIZES = (16, 32, 64)


def _shift_amount(n: int) -> int:
    """Bit count ``k`` such that ``n == 1 << k``, for power-of-two ``n``.

    ``mx.floor_divide`` truncates toward zero for negative operands rather
    than flooring, so any division by a power of two that might see a
    negative numerator must go through ``mx.right_shift`` instead, which is
    an arithmetic (floor) shift and agrees with Python's ``//`` for all
    signs.

    Raises:
        ValueError: if ``n`` is not a positive power of two.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"expected a power-of-two tile size, got {n}")
    return n.bit_length() - 1


def _floor_div_pow2(x: mx.array, bits: int) -> mx.array:
    """``floor(x / 2**bits)`` for possibly-negative ``x``, exact for all signs.

    Exposed (private) so the floor behaviour on negative fixed-point
    coordinates can be pinned directly by a test, independent of the
    downstream clamp in :func:`face_spans` which otherwise hides a regression
    to truncating division: any negative numerator clamps to the same ``0``
    whether it was floored correctly or merely truncated toward zero.
    """
    return mx.right_shift(x, bits)


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
    px0 = _floor_div_pow2(lo - half + SUBPIXEL_SCALE - 1, SUBPIXEL_BITS)
    px1 = _floor_div_pow2(hi - half, SUBPIXEL_BITS)

    max_xy = mx.array([width - 1, height - 1], dtype=mx.int32)
    px0 = mx.clip(px0, mx.array(0, dtype=mx.int32), max_xy)
    px1 = mx.clip(px1, mx.array(0, dtype=mx.int32), max_xy)

    onscreen = mx.all(mx.logical_and(hi >= half, lo <= max_xy * SUBPIXEL_SCALE + half), axis=1)

    tile_bits = _shift_amount(tile_size)
    t0 = _floor_div_pow2(px0, tile_bits)
    t1 = _floor_div_pow2(px1, tile_bits)
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


def choose_tiling(
    verts_fx: mx.array,
    faces: mx.array,
    width: int,
    height: int,
    cull: str,
    tile_size: int | None = None,
) -> tuple[int, mx.array, mx.array, int]:
    """Pick the finest tiling whose (tile, face) pair list fits the budget.

    Reading ``total_pairs`` back to the host is what sizes the pair buffer, so
    this is the pipeline's mandatory synchronization point.

    Args:
        verts_fx: (N, 2) int32 fixed-point screen coordinates.
        faces: (F, 3) int32 vertex indices.
        width: Image width in pixels.
        height: Image height in pixels.
        cull: One of ``"none"``, ``"back"``, ``"front"``.
        tile_size: Force a tile size instead of choosing one. Test hook.

    Returns:
        ``(tile_size, tile_bounds, n_tiles, total_pairs)``.

    Raises:
        MemoryError: if even the coarsest tiling exceeds the budget.
    """
    candidates = (tile_size,) if tile_size is not None else TILE_SIZES
    last = 0
    for ts in candidates:
        bounds, n = face_spans(verts_fx, faces, width, height, ts, cull)
        total = item_int(mx.sum(n))
        if tile_size is not None or total <= MAX_PAIRS:
            return ts, bounds, n, total
        last = total
    raise MemoryError(
        f"rasterize: {last} (tile, face) pairs at tile size {TILE_SIZES[-1]} "
        f"exceeds the {MAX_PAIRS} pair budget. The mesh has faces spanning a "
        f"large share of the image; reduce the face count or the resolution."
    )

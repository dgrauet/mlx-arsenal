"""Independent float64/int64 reference rasterizer, for testing only.

Implements the same rules as the MLX kernel — exact integer edge functions on
1/16-pixel fixed-point coordinates, top-left fill rule — in plain numpy. It is
slow and meant for small images; its value is being a genuinely separate
implementation rather than a recording of the one under test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SUBPIXEL_SCALE = 16


def _edge(
    ax: int, ay: int, bx: int, by: int, px: int | NDArray, py: int | NDArray
) -> int | NDArray:
    """Exact integer edge function for edge a->b evaluated at p."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _is_top_left(ax: int, ay: int, bx: int, by: int) -> bool:
    """True if edge a->b is a top or left edge (y grows downward)."""
    dx, dy = bx - ax, by - ay
    return (dy == 0 and dx > 0) or dy > 0


def _pixel_centres(width: int, height: int) -> tuple[NDArray, NDArray]:
    xs = np.arange(width, dtype=np.int64) * SUBPIXEL_SCALE + SUBPIXEL_SCALE // 2
    ys = np.arange(height, dtype=np.int64) * SUBPIXEL_SCALE + SUBPIXEL_SCALE // 2
    px, py = np.meshgrid(xs, ys)  # (H, W) each
    return px, py


def _face_coverage(
    verts_fx: NDArray, face: NDArray, width: int, height: int
) -> tuple[NDArray, NDArray | int, NDArray | int, NDArray | int, NDArray | int | float]:
    """(inside mask (H,W) bool, w0, w1, w2 int64 edge values, area int64)."""
    i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
    (x0, y0), (x1, y1), (x2, y2) = (
        verts_fx[i0].tolist(),
        verts_fx[i1].tolist(),
        verts_fx[i2].tolist(),
    )
    area = _edge(x0, y0, x1, y1, x2, y2)
    if area == 0:
        h, w = height, width
        z = np.zeros((h, w), dtype=np.int64)
        return np.zeros((h, w), dtype=bool), z, z, z, 0
    swapped = area < 0
    if swapped:  # normalise winding so `area` is positive
        (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        area = -area

    px, py = _pixel_centres(width, height)
    e0 = _edge(x1, y1, x2, y2, px, py)  # opposite v0
    e1 = _edge(x2, y2, x0, y0, px, py)  # opposite v1
    e2 = _edge(x0, y0, x1, y1, px, py)  # opposite v2

    inside = (
        np.where(_is_top_left(x1, y1, x2, y2), e0 >= 0, e0 > 0)
        & np.where(_is_top_left(x2, y2, x0, y0), e1 >= 0, e1 > 0)
        & np.where(_is_top_left(x0, y0, x1, y1), e2 >= 0, e2 > 0)
    )
    if swapped:  # undo the vertex swap for the weights
        e1, e2 = e2, e1
    return inside, e0, e1, e2, area


def coverage_count(verts_fx: NDArray, faces: NDArray, width: int, height: int) -> NDArray:
    """How many faces claim each pixel under the top-left rule."""
    counts = np.zeros((height, width), dtype=np.int32)
    for face in faces:
        inside, *_ = _face_coverage(verts_fx, face, width, height)
        counts += inside.astype(np.int32)
    return counts


def rasterize_reference(
    verts_fx: NDArray,
    verts_zw: NDArray,
    faces: NDArray,
    width: int,
    height: int,
    cull: str = "none",
) -> tuple[NDArray, NDArray, NDArray]:
    """Reference rasterization. Returns (face_indices, barycentric, depth)."""
    fi = np.zeros((height, width), dtype=np.int32)
    bary = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.full((height, width), np.inf, dtype=np.float64)

    for f, face in enumerate(faces):
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        raw_area = _edge(*verts_fx[i0].tolist(), *verts_fx[i1].tolist(), *verts_fx[i2].tolist())
        if raw_area == 0:
            continue
        if cull == "back" and raw_area < 0:
            continue
        if cull == "front" and raw_area > 0:
            continue

        inside, e0, e1, e2, area = _face_coverage(verts_fx, face, width, height)
        if not inside.any():
            continue

        w0 = e0 / area
        w1 = e1 / area
        w2 = e2 / area
        z = w0 * verts_zw[i0, 0] + w1 * verts_zw[i1, 0] + w2 * verts_zw[i2, 0]

        win = inside & (z < depth)
        if not win.any():
            continue
        depth = np.where(win, z, depth)
        fi = np.where(win, f + 1, fi)

        # perspective-correct barycentrics
        p0 = w0 / verts_zw[i0, 1]
        p1 = w1 / verts_zw[i1, 1]
        p2 = w2 / verts_zw[i2, 1]
        norm = p0 + p1 + p2
        for c, pc in enumerate((p0, p1, p2)):
            bary[..., c] = np.where(win, pc / norm, bary[..., c])

    return fi, bary, depth


def quad_mesh(n: int, seed: int = 0):
    """Triangulated grid in clip space with many shared edges (test mesh)."""
    import numpy as np

    from mlx_arsenal._typing import array_from_any

    rng = np.random.default_rng(seed)
    g = np.linspace(-0.9, 0.9, n + 1)
    xs, ys = np.meshgrid(g, g)
    zs = rng.uniform(0.1, 0.9, size=xs.shape)
    verts = np.stack([xs.ravel(), ys.ravel(), zs.ravel(), np.ones(xs.size)], axis=1).astype(
        np.float32
    )
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate(
        [np.stack([tl, bl, tr], axis=1), np.stack([tr, bl, br], axis=1)], axis=0
    ).astype(np.int32)
    return array_from_any(verts), array_from_any(faces)

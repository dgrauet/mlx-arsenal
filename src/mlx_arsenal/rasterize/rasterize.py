"""Metal triangle rasterizer with z-buffering for Apple Silicon.

Tile-binned: faces are bucketed into screen tiles so each tile tests only the
faces that can touch it, instead of every thread scanning every face. Coverage
uses exact int64 edge functions over 1/16-pixel fixed-point coordinates and the
top-left fill rule, so adjacent triangles cannot both claim a shared edge, nor
leave a crack between them.

Note that this function synchronizes: the number of (tile, face) pairs has to
reach the host to size the pair buffer, so it cannot stay lazy in an MLX graph.
"""

from __future__ import annotations

from typing import Literal, overload

import mlx.core as mx

from mlx_arsenal._typing import item_int

from ._binning import build_tile_lists, choose_tiling
from ._fixedpoint import MAX_DIM, to_screen
from ._tile_raster import raster_tiles

__all__ = ["rasterize_triangles"]

_CULL_MODES = ("none", "back", "front")

# The tile kernel's edge function (and the binning stage's signed-area test)
# forms two differences of fixed-point vertex coordinates and multiplies them
# together, then subtracts a second such product:
#   edge = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
# If every fixed-point coordinate is bounded by B in absolute value, a
# difference of two coordinates is bounded by 2B, a product of two
# differences by (2B)*(2B) = 4*B**2, and the subtraction of two such products
# by 8*B**2. This is evaluated in int64, whose range tops out at 2**63 - 1, so
# overflow-free evaluation requires 8*B**2 < 2**63, i.e. B < 2**30. A vertex
# whose projected fixed-point coordinate approaches this bound is nowhere
# near representable screen space (a 16384px image only spans 16384*16 =
# 262144 fixed-point units), so this only fires when a vertex crosses or
# nears the camera's near plane and its projected position blows up.
_MAX_FX_MAGNITUDE = 1 << 30


@overload
def rasterize_triangles(
    vertices: mx.array,
    faces: mx.array,
    width: int,
    height: int,
    depth_prior: mx.array | None = ...,
    occlusion_truncation: float = ...,
    cull: Literal["none", "back", "front"] = ...,
    return_depth: Literal[False] = ...,
    _tile_size: int | None = ...,
) -> tuple[mx.array, mx.array]: ...


@overload
def rasterize_triangles(
    vertices: mx.array,
    faces: mx.array,
    width: int,
    height: int,
    depth_prior: mx.array | None = ...,
    occlusion_truncation: float = ...,
    cull: Literal["none", "back", "front"] = ...,
    *,
    return_depth: Literal[True],
    _tile_size: int | None = ...,
) -> tuple[mx.array, mx.array, mx.array]: ...


def rasterize_triangles(
    vertices: mx.array,
    faces: mx.array,
    width: int,
    height: int,
    depth_prior: mx.array | None = None,
    occlusion_truncation: float = 1e-6,
    cull: Literal["none", "back", "front"] = "none",
    return_depth: bool = False,
    _tile_size: int | None = None,
) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array]:
    """Rasterize projected triangles with depth-aware z-buffering.

    Args:
        vertices: (N, 4) float32, clip-space homogeneous (x, y, z, w).
        faces: (F, 3) int32, triangle vertex indices.
        width: Image width in pixels, at most 16384.
        height: Image height in pixels, at most 16384.
        depth_prior: Optional (H, W) float32 depth map for occlusion culling,
            expected in **NDC z** (pre-mapping): it is mapped internally the
            same way vertex z is mapped for rasterization. The ``depth``
            returned by ``return_depth=True`` is already post-mapping, so
            feeding a previously returned depth map straight back in as a
            prior double-maps it and culls the wrong geometry — convert it
            back to NDC first.
        occlusion_truncation: Depth threshold for occlusion.
        cull: Discard faces by orientation. ``"none"`` keeps everything, which
            is the default and matches earlier releases. A face is
            front-facing when its vertices wind counter-clockwise in NDC
            (equivalently, in the returned image's coordinate system, since
            ``to_screen`` maps both axes monotonically increasing and does
            not flip the winding) — the OpenGL convention. ``"back"``
            discards clockwise faces and keeps counter-clockwise ones;
            ``"front"`` does the reverse.
        return_depth: Also return the winning face's interpolated depth.
        _tile_size: Force the binning tile size. Private test hook — results are
            invariant to it.

    Returns:
        face_indices: (H, W) int32 — 1-indexed face ID per pixel (0 = background).
        barycentric: (H, W, 3) float32 — perspective-correct barycentrics.
        depth: (H, W) float32, only when ``return_depth`` — ``+inf`` on
            background pixels.

    Raises:
        ValueError: on an unknown ``cull`` mode, an out-of-range vertex index,
            an image larger than 16384px per axis, or projected vertex
            coordinates outside the representable range (most likely
            geometry crossing or approaching the near plane).
        MemoryError: if the mesh needs more (tile, face) pairs than the budget
            allows even at the coarsest tiling.
    """
    if cull not in _CULL_MODES:
        raise ValueError(f"cull must be one of {_CULL_MODES}, got {cull!r}")
    if width > MAX_DIM or height > MAX_DIM:
        raise ValueError(
            f"rasterize supports images up to {MAX_DIM}px per axis; got {width}x{height}"
        )
    if depth_prior is not None and tuple(depth_prior.shape) != (height, width):
        raise ValueError(
            f"depth_prior must have shape (height, width) = ({height}, {width}), "
            f"got {tuple(depth_prior.shape)}"
        )

    num_faces = faces.shape[0]
    if num_faces > 0:
        max_index = item_int(mx.max(faces))
        min_index = item_int(mx.min(faces))
        if max_index >= vertices.shape[0] or min_index < 0:
            raise ValueError(
                f"faces contains a vertex index outside [0, {vertices.shape[0] - 1}]: "
                f"saw [{min_index}, {max_index}]"
            )

    if num_faces == 0:
        bg_faces = mx.zeros((height, width), dtype=mx.int32)
        bg_bary = mx.zeros((height, width, 3), dtype=mx.float32)
        if return_depth:
            bg_depth = mx.full((height, width), float("inf"), dtype=mx.float32)
            return bg_faces, bg_bary, bg_depth
        return bg_faces, bg_bary

    verts_fx, verts_zw = to_screen(vertices, width, height)

    # Bound both signs directly rather than via `mx.abs`: `int32` cannot
    # represent `abs(INT32_MIN)` (it saturates and stays negative), so an
    # abs-then-compare check silently misses a coordinate that saturated
    # negative.
    max_fx = item_int(mx.max(verts_fx))
    min_fx = item_int(mx.min(verts_fx))
    if max_fx >= _MAX_FX_MAGNITUDE or min_fx <= -_MAX_FX_MAGNITUDE:
        offending = max_fx if max_fx >= _MAX_FX_MAGNITUDE else min_fx
        raise ValueError(
            f"projected vertex coordinates are outside the representable range "
            f"(coord = {offending}, allowed magnitude < {_MAX_FX_MAGNITUDE} "
            f"fixed-point units); this most likely means geometry crosses or "
            f"approaches the near plane. Clip triangles against the near plane "
            f"before rasterizing."
        )

    tile_size, bounds, n_tiles, total_pairs = choose_tiling(
        verts_fx, faces, width, height, cull, tile_size=_tile_size
    )

    if total_pairs == 0:
        # Every face is degenerate, culled, or offscreen: nothing can be
        # covered. The spec's failure-modes table asks for no dispatch here,
        # matching the num_faces == 0 short-circuit above.
        bg_faces = mx.zeros((height, width), dtype=mx.int32)
        bg_bary = mx.zeros((height, width, 3), dtype=mx.float32)
        if return_depth:
            bg_depth = mx.full((height, width), float("inf"), dtype=mx.float32)
            return bg_faces, bg_bary, bg_depth
        return bg_faces, bg_bary

    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    sorted_faces, tile_starts = build_tile_lists(
        bounds, n_tiles, total_pairs, tiles_x, tiles_y, num_faces
    )

    findices, bary, depth = raster_tiles(
        verts_fx,
        verts_zw,
        faces,
        sorted_faces,
        tile_starts,
        width,
        height,
        tile_size,
        depth_prior,
        occlusion_truncation,
    )
    return (findices, bary, depth) if return_depth else (findices, bary)

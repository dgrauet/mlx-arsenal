"""Per-tile rasterization kernel.

One threadgroup per 16x16 sub-tile, one thread per pixel. The sub-tile looks up
its parent binning tile's face list and walks it in chunks of 256, loading each
chunk cooperatively into threadgroup memory before testing.

Coverage uses exact int64 edge functions on fixed-point coordinates plus the
top-left fill rule, so a pixel on a shared edge belongs to exactly one triangle.
"""

from __future__ import annotations

import mlx.core as mx

RASTER_TILE = 16
CHUNK = 256

# The Metal source is generated from _TEMPLATE below with CHUNK and
# RASTER_TILE interpolated, so the constants and the kernel cannot desync.
# The one real constraint: the chunk is loaded cooperatively by the
# RASTER_TILE * RASTER_TILE threads of a threadgroup, so CHUNK must not
# exceed the threadgroup size.
if CHUNK > RASTER_TILE * RASTER_TILE:
    raise RuntimeError(
        f"CHUNK ({CHUNK}) must be <= RASTER_TILE**2 ({RASTER_TILE * RASTER_TILE}): "
        "each chunk is loaded cooperatively by one threadgroup"
    )

__all__ = ["RASTER_TILE", "raster_tiles"]

_HEADER = """
inline long edge_fn(long ax, long ay, long bx, long by, long px, long py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

inline bool is_top_left(long ax, long ay, long bx, long by) {
    long dx = bx - ax;
    long dy = by - ay;
    return (dy == 0 && dx > 0) || (dy > 0);
}

inline bool edge_inside(long e, bool top_left) {
    return top_left ? (e >= 0) : (e > 0);
}
"""

_TEMPLATE = """
    threadgroup int   sh_xy[__CHUNK__ * 6];
    threadgroup float sh_zw[__CHUNK__ * 6];
    threadgroup int   sh_fid[__CHUNK__];

    int width      = dims[0];
    int height     = dims[1];
    int tile_size  = dims[2];
    int sub_x      = dims[3];   // sub-tiles across the image
    int tiles_x    = dims[4];   // binning tiles across the image

    uint sub = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    int sx = (int)(sub % (uint)sub_x);
    int sy = (int)(sub / (uint)sub_x);
    int px = sx * __TILE__ + (int)(tid % __TILE__u);
    int py = sy * __TILE__ + (int)(tid / __TILE__u);

    // Parent binning tile for this sub-tile.
    int bt = (py / tile_size) * tiles_x + (px / tile_size);
    int lo = tile_starts[bt];
    int hi = tile_starts[bt + 1];

    bool active = (px < width) && (py < height);
    long fx = (long)px * 16 + 8;   // pixel centre, fixed point
    long fy = (long)py * 16 + 8;

    float best_depth = INFINITY;
    int   best_face  = 0;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
    float prior = active ? depth_prior[py * width + px] : 0.0f;
    float occl = fparams[0];

    for (int base = lo; base < hi; base += __CHUNK__) {
        int n = min(__CHUNK__, hi - base);
        if ((int)tid < n) {
            int f = sorted_faces[base + (int)tid];
            sh_fid[tid] = f;
            for (int k = 0; k < 3; k++) {
                int vi = faces[f * 3 + k];
                sh_xy[tid * 6 + k * 2]     = verts_fx[vi * 2];
                sh_xy[tid * 6 + k * 2 + 1] = verts_fx[vi * 2 + 1];
                sh_zw[tid * 6 + k * 2]     = verts_zw[vi * 2];
                sh_zw[tid * 6 + k * 2 + 1] = verts_zw[vi * 2 + 1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (int j = 0; j < n; j++) {
                long x0 = sh_xy[j * 6],     y0 = sh_xy[j * 6 + 1];
                long x1 = sh_xy[j * 6 + 2], y1 = sh_xy[j * 6 + 3];
                long x2 = sh_xy[j * 6 + 4], y2 = sh_xy[j * 6 + 5];

                long area = edge_fn(x0, y0, x1, y1, x2, y2);
                if (area == 0) continue;
                bool swapped = area < 0;
                if (swapped) {
                    long tx = x1, ty = y1;
                    x1 = x2; y1 = y2; x2 = tx; y2 = ty;
                    area = -area;
                }

                long e0 = edge_fn(x1, y1, x2, y2, fx, fy);
                long e1 = edge_fn(x2, y2, x0, y0, fx, fy);
                long e2 = edge_fn(x0, y0, x1, y1, fx, fy);
                if (!edge_inside(e0, is_top_left(x1, y1, x2, y2))) continue;
                if (!edge_inside(e1, is_top_left(x2, y2, x0, y0))) continue;
                if (!edge_inside(e2, is_top_left(x0, y0, x1, y1))) continue;

                float inv_area = 1.0f / (float)area;
                float w0 = (float)e0 * inv_area;
                float w1 = (float)e1 * inv_area;
                float w2 = (float)e2 * inv_area;
                if (swapped) { float t = w1; w1 = w2; w2 = t; }

                // sh_zw is in original face order, and w0/w1/w2 were just put
                // back into it, so these must NOT be swapped as well.
                float z0 = sh_zw[j * 6],     cw0 = sh_zw[j * 6 + 1];
                float z1 = sh_zw[j * 6 + 2], cw1 = sh_zw[j * 6 + 3];
                float z2 = sh_zw[j * 6 + 4], cw2 = sh_zw[j * 6 + 5];

                float depth = w0 * z0 + w1 * z1 + w2 * z2;
                if (depth < prior * 0.49999f + 0.5f + occl) continue;
                if (depth >= best_depth) continue;

                best_depth = depth;
                best_face  = sh_fid[j] + 1;
                float p0 = w0 / cw0, p1 = w1 / cw1, p2 = w2 / cw2;
                float inv = 1.0f / (p0 + p1 + p2);
                b0 = p0 * inv; b1 = p1 * inv; b2 = p2 * inv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        int o = py * width + px;
        findices[o] = best_face;
        depth_out[o] = best_face > 0 ? best_depth : INFINITY;
        barycentric_out[o * 3]     = b0;
        barycentric_out[o * 3 + 1] = b1;
        barycentric_out[o * 3 + 2] = b2;
    }
"""

_SOURCE = _TEMPLATE.replace("__CHUNK__", str(CHUNK)).replace("__TILE__", str(RASTER_TILE))

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="rasterize_tile_kernel",
            input_names=[
                "verts_fx",
                "verts_zw",
                "faces",
                "sorted_faces",
                "tile_starts",
                "depth_prior",
                "dims",
                "fparams",
            ],
            output_names=["findices", "barycentric_out", "depth_out"],
            source=_SOURCE,
            header=_HEADER,
            ensure_row_contiguous=True,
            atomic_outputs=False,
        )
    return _kernel


def raster_tiles(
    verts_fx: mx.array,
    verts_zw: mx.array,
    faces: mx.array,
    sorted_faces: mx.array,
    tile_starts: mx.array,
    width: int,
    height: int,
    tile_size: int,
    depth_prior: mx.array | None,
    occlusion_truncation: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Rasterize the binned faces.

    Args:
        verts_fx: (N, 2) int32 fixed-point screen coordinates.
        verts_zw: (N, 2) float32 depth and clip w.
        faces: (F, 3) int32 vertex indices.
        sorted_faces: (P,) int32 tile-major face list.
        tile_starts: (T + 1,) int32 offsets into ``sorted_faces``.
        width: Image width in pixels.
        height: Image height in pixels.
        tile_size: Binning tile edge in pixels. Must be a multiple of
            :data:`RASTER_TILE`, and must be the same value ``tile_starts`` was
            built with.
        depth_prior: Optional (H, W) float32 depth map for occlusion culling.
            Expected in **NDC z** (pre-mapping): it is mapped internally the
            same way :func:`~mlx_arsenal.rasterize._fixedpoint.to_screen` maps
            vertex z. The returned ``depth`` is already **post-mapping**, so the
            two are not interchangeable — feeding a returned depth map straight
            back in as a prior double-maps it and culls the wrong geometry.
            Convert it back to NDC first.
        occlusion_truncation: Depth threshold for occlusion.

    Returns:
        ``(face_indices, barycentric, depth)`` shaped (H, W), (H, W, 3), (H, W).
        ``depth`` is in mapped (post-``to_screen``) space; see ``depth_prior``.

    Raises:
        ValueError: if ``tile_size`` is not a multiple of :data:`RASTER_TILE`,
            or if ``tile_starts`` does not match ``tile_size`` and the image
            dimensions.
    """
    # A threadgroup covers one RASTER_TILE-square sub-tile and reads a single
    # binning tile's face list, so every pixel in it must land in the same
    # binning tile: `lo`/`hi` have to be threadgroup-uniform or the
    # `threadgroup_barrier` calls sit in divergent control flow.
    if tile_size % RASTER_TILE != 0:
        raise ValueError(f"tile_size must be a multiple of {RASTER_TILE}, got {tile_size}")

    sub_x = (width + RASTER_TILE - 1) // RASTER_TILE
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    if tile_starts.size != tiles_x * tiles_y + 1:
        raise ValueError(
            f"tile_starts does not match tile_size / image dimensions: got "
            f"{tile_starts.size} entries, expected {tiles_x * tiles_y + 1} for a "
            f"{width}x{height} image at tile_size {tile_size}"
        )
    num_sub = sub_x * ((height + RASTER_TILE - 1) // RASTER_TILE)

    prior = (
        mx.full((height * width,), -1e30, dtype=mx.float32)
        if depth_prior is None
        else depth_prior.reshape(-1).astype(mx.float32)
    )
    dims = mx.array([width, height, tile_size, sub_x, tiles_x], dtype=mx.int32)
    fparams = mx.array([occlusion_truncation], dtype=mx.float32)

    findices, bary, depth = _get_kernel()(
        inputs=[
            verts_fx.reshape(-1),
            verts_zw.reshape(-1),
            faces.reshape(-1).astype(mx.int32),
            sorted_faces,
            tile_starts,
            prior,
            dims,
            fparams,
        ],
        template=[("T", mx.float32)],
        grid=(num_sub * CHUNK, 1, 1),
        threadgroup=(CHUNK, 1, 1),
        output_shapes=[(height * width,), (height * width * 3,), (height * width,)],
        output_dtypes=[mx.int32, mx.float32, mx.float32],
    )
    return (
        findices.reshape(height, width),
        bary.reshape(height, width, 3),
        depth.reshape(height, width),
    )

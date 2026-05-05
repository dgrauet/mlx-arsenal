"""Metal triangle rasterizer with z-buffering for Apple Silicon.

Port of Hunyuan3D-2.1 CUDA rasterizer to Metal via mx.fast.metal_kernel.
Uses a per-pixel kernel that loops over all faces with bounding-box culling.
"""

import mlx.core as mx

METAL_HEADER = """
inline float calculateSignedArea2(float2 a, float2 b, float2 c) {
    return ((c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y));
}

inline float3 calculateBarycentricCoordinate(float2 a, float2 b, float2 c, float2 p) {
    float beta_tri = calculateSignedArea2(a, p, c);
    float gamma_tri = calculateSignedArea2(a, b, p);
    float area = calculateSignedArea2(a, b, c);
    if (area == 0.0f) return float3(-1.0f, -1.0f, -1.0f);
    float tri_inv = 1.0f / area;
    float beta = beta_tri * tri_inv;
    float gamma = gamma_tri * tri_inv;
    return float3(1.0f - beta - gamma, beta, gamma);
}

inline bool isBarycentricCoordInBounds(float3 bary) {
    return bary.x >= 0.0f && bary.x <= 1.0f &&
           bary.y >= 0.0f && bary.y <= 1.0f &&
           bary.z >= 0.0f && bary.z <= 1.0f;
}
"""

# ---------------------------------------------------------------------------
# Per-pixel rasterize + barycentric kernel  (T = float32)
#
# Grid = (total_pixels, 1, 1)
# One thread per pixel. Each thread loops over all faces, tracks the closest
# via depth comparison (no atomics needed).
#
# Inputs:
#   screen_verts – (N*4,) float32 [sx, sy, sz, w] per vertex (precomputed)
#   faces        – (F*3,) int32 data → as_type<int>()
#   depth_prior  – (H*W,) float32
#   dims         – (4,) int32 [width, height, num_faces, total_pixels]
#   fparams      – (1,) float32 [occlusion_truncation]
#
# Outputs:
#   findices       – float (MLX converts to int32 via output_dtypes)
#   barycentric_out– float32
# ---------------------------------------------------------------------------
RASTERIZE_SOURCE = """
    uint pix = thread_position_in_grid.x;

    int width      = as_type<int>(dims[0]);
    int height     = as_type<int>(dims[1]);
    uint num_faces = as_type<uint>(dims[2]);
    uint total_px  = as_type<uint>(dims[3]);
    if (pix >= total_px) return;

    float occlusion_trunc = fparams[0];
    float px = (float)(pix % (uint)width) + 0.5f;
    float py = (float)(pix / (uint)width) + 0.5f;

    uint  best_face  = 0;         // 0 = background
    float best_depth = 1e30f;
    float3 best_bary = float3(0.0f);
    float  best_w0 = 1.0f, best_w1 = 1.0f, best_w2 = 1.0f;

    for (uint f = 0; f < num_faces; f++) {
        int v0i = as_type<int>(faces[f * 3]);
        int v1i = as_type<int>(faces[f * 3 + 1]);
        int v2i = as_type<int>(faces[f * 3 + 2]);

        // Read precomputed screen-space verts: (sx, sy, sz, w)
        float sx0 = screen_verts[v0i*4],   sy0 = screen_verts[v0i*4+1];
        float sz0 = screen_verts[v0i*4+2], sw0 = screen_verts[v0i*4+3];
        float sx1 = screen_verts[v1i*4],   sy1 = screen_verts[v1i*4+1];
        float sz1 = screen_verts[v1i*4+2], sw1 = screen_verts[v1i*4+3];
        float sx2 = screen_verts[v2i*4],   sy2 = screen_verts[v2i*4+1];
        float sz2 = screen_verts[v2i*4+2], sw2 = screen_verts[v2i*4+3];

        // Bounding box culling
        float x_min = min(sx0, min(sx1, sx2));
        float x_max = max(sx0, max(sx1, sx2));
        float y_min = min(sy0, min(sy1, sy2));
        float y_max = max(sy0, max(sy1, sy2));

        if (px < x_min || px > x_max + 1.0f ||
            py < y_min || py > y_max + 1.0f) continue;

        float3 bary = calculateBarycentricCoordinate(
            float2(sx0, sy0), float2(sx1, sy1), float2(sx2, sy2), float2(px, py));
        if (!isBarycentricCoordInBounds(bary)) continue;

        float depth = bary.x * sz0 + bary.y * sz1 + bary.z * sz2;

        // Depth prior occlusion culling
        float depth_thres = depth_prior[pix] * 0.49999f + 0.5f + occlusion_trunc;
        if (depth < depth_thres) continue;

        if (depth < best_depth) {
            best_depth = depth;
            best_face  = f + 1;   // 1-indexed
            best_bary  = bary;
            best_w0 = sw0;  best_w1 = sw1;  best_w2 = sw2;
        }
    }

    // Perspective-correct barycentric
    if (best_face > 0) {
        best_bary.x /= best_w0;
        best_bary.y /= best_w1;
        best_bary.z /= best_w2;
        float w_inv = 1.0f / (best_bary.x + best_bary.y + best_bary.z);
        best_bary *= w_inv;
    }

    findices[pix] = (float)best_face;
    barycentric_out[pix * 3]     = best_bary.x;
    barycentric_out[pix * 3 + 1] = best_bary.y;
    barycentric_out[pix * 3 + 2] = best_bary.z;
"""

_rasterize_kernel = None


def _get_rasterize_kernel():
    global _rasterize_kernel
    if _rasterize_kernel is None:
        _rasterize_kernel = mx.fast.metal_kernel(
            name="rasterize_perpixel_kernel",
            input_names=[
                "screen_verts",
                "faces",
                "depth_prior",
                "dims",
                "fparams",
            ],
            output_names=["findices", "barycentric_out"],
            source=RASTERIZE_SOURCE,
            header=METAL_HEADER,
            ensure_row_contiguous=True,
            atomic_outputs=False,
        )
    return _rasterize_kernel


def rasterize_triangles(
    vertices: mx.array,
    faces: mx.array,
    width: int,
    height: int,
    depth_prior: mx.array | None = None,
    occlusion_truncation: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """Rasterize projected triangles with depth-aware z-buffering.

    Args:
        vertices: (N, 4) float32, clip-space homogeneous (x, y, z, w).
        faces: (F, 3) int32, triangle vertex indices.
        width: Image width in pixels.
        height: Image height in pixels.
        depth_prior: Optional (H, W) float32 depth map for occlusion culling.
        occlusion_truncation: Depth threshold for occlusion.

    Returns:
        face_indices: (H, W) int32 -- 1-indexed face ID per pixel (0 = background).
        barycentric: (H, W, 3) float32 -- barycentric coordinates per pixel.
    """
    num_faces = faces.shape[0]
    total_pixels = height * width

    # Precompute screen-space vertices in pure MLX (avoids redundant
    # per-pixel transforms in the Metal kernel)
    v = vertices.astype(mx.float32)
    w_clip = v[:, 3:4]  # (N, 1)
    screen_x = (v[:, 0:1] / w_clip * 0.5 + 0.5) * (width - 1) + 0.5
    screen_y = (0.5 + 0.5 * v[:, 1:2] / w_clip) * (height - 1) + 0.5
    screen_z = v[:, 2:3] / w_clip * 0.49999 + 0.5
    screen_verts = mx.concatenate([screen_x, screen_y, screen_z, w_clip], axis=1).reshape(
        -1
    )  # (N*4,) float32

    faces_flat = faces.reshape(-1).astype(mx.int32)

    if depth_prior is not None:
        depth_prior_flat = depth_prior.reshape(-1).astype(mx.float32)
    else:
        depth_prior_flat = mx.full((total_pixels,), -1e30, dtype=mx.float32)

    dims = mx.array([width, height, num_faces, total_pixels], dtype=mx.int32)
    fparams = mx.array([occlusion_truncation], dtype=mx.float32)

    tg_size = min(256, total_pixels)

    findices, bary_flat = _get_rasterize_kernel()(
        inputs=[screen_verts, faces_flat, depth_prior_flat, dims, fparams],
        template=[("T", mx.float32)],
        grid=(total_pixels, 1, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(total_pixels,), (total_pixels * 3,)],
        output_dtypes=[mx.int32, mx.float32],
    )

    return findices.reshape(height, width), bary_flat.reshape(height, width, 3)

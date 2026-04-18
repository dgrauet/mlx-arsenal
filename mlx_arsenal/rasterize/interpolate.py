import mlx.core as mx


def interpolate(
    attributes: mx.array,
    face_indices: mx.array,
    barycentric: mx.array,
    faces: mx.array,
) -> mx.array:
    """Interpolate per-vertex attributes at rasterized pixels.

    Args:
        attributes: (N, C) float32 per-vertex data (normals, UVs, colors, ...).
        face_indices: (H, W) int32 from rasterize_triangles (1-indexed, 0=background).
        barycentric: (H, W, 3) float32 from rasterize_triangles.
        faces: (F, 3) int32 triangle vertex indices.

    Returns:
        (H, W, C) float32 interpolated attributes per pixel.
    """
    # Map background (0) to face 0 to avoid out-of-bounds; barycentric is zero
    # there so the result is zero regardless.
    f = face_indices - 1 + (face_indices == 0).astype(mx.int32)
    tri_verts = faces[f]  # (H, W, 3)

    v0 = attributes[tri_verts[..., 0]]  # (H, W, C)
    v1 = attributes[tri_verts[..., 1]]
    v2 = attributes[tri_verts[..., 2]]

    result = (
        barycentric[..., 0:1] * v0
        + barycentric[..., 1:2] * v1
        + barycentric[..., 2:3] * v2
    )
    return result

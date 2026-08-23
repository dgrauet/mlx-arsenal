"""Tests for the Metal triangle rasterizer."""

import mlx.core as mx
import pytest

from mlx_arsenal._typing import item_float, item_int
from mlx_arsenal.rasterize import interpolate, rasterize_triangles


def _clip_vertex(x, y, z, w=1.0):
    return [x, y, z, w]


def _make_triangle_mesh(verts_clip, face_list):
    vertices = mx.array(verts_clip, dtype=mx.float32)
    faces = mx.array(face_list, dtype=mx.int32)
    return vertices, faces


class TestSingleTriangle:
    def setup_method(self):
        self.width, self.height = 16, 16
        self.vertices, self.faces = _make_triangle_mesh(
            [
                _clip_vertex(-0.5, -0.5, 0.5),
                _clip_vertex(0.5, -0.5, 0.5),
                _clip_vertex(0.0, 0.5, 0.5),
            ],
            [[0, 1, 2]],
        )

    def test_depth_prior_provided(self):
        # Passing a depth_prior takes the explicit branch (non-default path).
        # Use a prior matching the default sentinel (very small) so the z-test
        # is effectively a no-op and we still see the triangle.
        prior = mx.full((self.height, self.width), -1e30, dtype=mx.float32)
        fi, bary = rasterize_triangles(
            self.vertices, self.faces, self.width, self.height, depth_prior=prior
        )
        mx.synchronize()
        assert fi.shape == (self.height, self.width)
        assert bary.shape == (self.height, self.width, 3)
        assert item_int((fi > 0).astype(mx.int32).sum()) > 0

    def test_face_indices_nonzero(self):
        fi, bary = rasterize_triangles(self.vertices, self.faces, self.width, self.height)
        mx.synchronize()
        assert fi.shape == (self.height, self.width)
        assert bary.shape == (self.height, self.width, 3)

        covered = item_int((fi > 0).astype(mx.int32).sum())
        assert covered > 0, "No pixels covered by the triangle"

    def test_barycentric_sum_to_one(self):
        fi, bary = rasterize_triangles(self.vertices, self.faces, self.width, self.height)
        mx.synchronize()

        covered_mask = (fi > 0).astype(mx.float32)  # (H, W) 0/1
        if covered_mask.sum().item() == 0:
            pytest.skip("No covered pixels")

        bary_sum = bary.sum(axis=-1)  # (H, W)
        # Check that all covered pixels have bary sum ~1
        # For covered pixels: |bary_sum - 1| < atol
        # For background: bary_sum = 0 (don't care)
        error = mx.abs(bary_sum - 1.0) * covered_mask
        max_error = item_float(error.max())
        assert max_error < 1e-4, f"Max bary sum error on covered pixels: {max_error}"

    def test_background_is_zero(self):
        fi, bary = rasterize_triangles(self.vertices, self.faces, self.width, self.height)
        mx.synchronize()

        bg_mask = (fi == 0).astype(mx.float32)  # ty: ignore[unresolved-attribute]  # (H, W)
        if bg_mask.sum().item() == 0:
            pytest.skip("Entire image covered")

        # Background pixels should have all-zero barycentric
        bg_bary_abs = (mx.abs(bary).sum(axis=-1) * bg_mask).sum().item()
        assert bg_bary_abs == 0.0, "Background pixels should have zero barycentric"


class TestTwoOverlappingTriangles:
    def test_closer_wins(self):
        w, h = 16, 16
        vertices, faces = _make_triangle_mesh(
            [
                # Face 0 (closer, z=0.3)
                _clip_vertex(-0.5, -0.5, 0.3),
                _clip_vertex(0.5, -0.5, 0.3),
                _clip_vertex(0.0, 0.5, 0.3),
                # Face 1 (further, z=0.7)
                _clip_vertex(-0.5, -0.5, 0.7),
                _clip_vertex(0.5, -0.5, 0.7),
                _clip_vertex(0.0, 0.5, 0.7),
            ],
            [[0, 1, 2], [3, 4, 5]],
        )

        fi, bary = rasterize_triangles(vertices, faces, w, h)
        mx.synchronize()

        covered_mask = fi > 0
        num_covered = item_int(covered_mask.astype(mx.int32).sum())
        assert num_covered > 0, "No covered pixels"

        # All covered pixels should show face 1 (1-indexed for face 0)
        # Check: covered pixels where fi != 1
        wrong = ((fi != 1) * covered_mask.astype(mx.int32)).sum().item()
        assert wrong == 0, f"Closer triangle (face 1) should win everywhere, {wrong} wrong pixels"


class TestCubeMesh:
    def test_cube_silhouette(self):
        w, h = 32, 32
        s = 0.4
        z_front = 0.3
        z_back = 0.7
        verts = [
            _clip_vertex(-s, -s, z_front),
            _clip_vertex(s, -s, z_front),
            _clip_vertex(s, s, z_front),
            _clip_vertex(-s, s, z_front),
            _clip_vertex(-s, -s, z_back),
            _clip_vertex(s, -s, z_back),
            _clip_vertex(s, s, z_back),
            _clip_vertex(-s, s, z_back),
        ]
        face_list = [
            [0, 1, 2],
            [0, 2, 3],
            [5, 4, 7],
            [5, 7, 6],
            [4, 0, 3],
            [4, 3, 7],
            [1, 5, 6],
            [1, 6, 2],
            [3, 2, 6],
            [3, 6, 7],
            [4, 5, 1],
            [4, 1, 0],
        ]

        vertices, faces = _make_triangle_mesh(verts, face_list)
        fi, bary = rasterize_triangles(vertices, faces, w, h)
        mx.synchronize()

        covered = item_int((fi > 0).astype(mx.int32).sum())
        total = w * h
        assert covered > total * 0.1, f"Too few covered pixels: {covered}/{total}"
        assert covered < total * 0.95, f"Too many covered pixels: {covered}/{total}"


class TestInterpolation:
    def test_color_gradient(self):
        w, h = 16, 16
        vertices, faces = _make_triangle_mesh(
            [
                _clip_vertex(-0.8, -0.8, 0.5),
                _clip_vertex(0.8, -0.8, 0.5),
                _clip_vertex(0.0, 0.8, 0.5),
            ],
            [[0, 1, 2]],
        )

        fi, bary = rasterize_triangles(vertices, faces, w, h)
        mx.synchronize()

        colors = mx.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=mx.float32,
        )
        result = interpolate(colors, fi, bary, faces)
        mx.synchronize()
        assert result.shape == (h, w, 3)

        covered_mask = (fi > 0).astype(mx.float32)
        if covered_mask.sum().item() == 0:
            pytest.skip("No coverage")

        # For covered pixels, interpolated RGB should sum to ~1
        color_sums = result.sum(axis=-1)  # (H, W)
        error = mx.abs(color_sums - 1.0) * covered_mask
        max_error = item_float(error.max())
        assert max_error < 1e-3, f"Interpolated colors max error: {max_error}"


class TestPerspectiveCorrection:
    def test_nonuniform_w(self):
        w, h = 16, 16
        vertices, faces = _make_triangle_mesh(
            [
                _clip_vertex(-0.5, -0.5, 0.5, 1.0),
                _clip_vertex(0.5, -0.5, 0.5, 2.0),
                _clip_vertex(0.0, 0.5, 0.5, 1.5),
            ],
            [[0, 1, 2]],
        )

        fi, bary = rasterize_triangles(vertices, faces, w, h)
        mx.synchronize()

        covered_mask = (fi > 0).astype(mx.float32)
        if covered_mask.sum().item() == 0:
            pytest.skip("No coverage")

        bary_sum = bary.sum(axis=-1)
        error = mx.abs(bary_sum - 1.0) * covered_mask
        max_error = item_float(error.max())
        assert max_error < 1e-3, f"Perspective bary sum error: {max_error}"


class TestEdgeCases:
    def test_degenerate_triangle(self):
        w, h = 8, 8
        vertices, faces = _make_triangle_mesh(
            [
                _clip_vertex(0.0, 0.0, 0.5),
                _clip_vertex(0.0, 0.0, 0.5),
                _clip_vertex(0.0, 0.0, 0.5),
            ],
            [[0, 1, 2]],
        )

        fi, bary = rasterize_triangles(vertices, faces, w, h)
        mx.synchronize()

        covered = item_int((fi > 0).astype(mx.int32).sum())
        assert covered == 0, "Degenerate triangle should not cover any pixels"

    def test_triangle_outside_viewport(self):
        w, h = 8, 8
        vertices, faces = _make_triangle_mesh(
            [
                _clip_vertex(2.0, 2.0, 0.5),
                _clip_vertex(3.0, 2.0, 0.5),
                _clip_vertex(2.5, 3.0, 0.5),
            ],
            [[0, 1, 2]],
        )

        fi, bary = rasterize_triangles(vertices, faces, w, h)
        mx.synchronize()

        covered = item_int((fi > 0).astype(mx.int32).sum())
        assert covered == 0, "Out-of-viewport triangle should not cover any pixels"


def _tri(z=0.5):
    v = mx.array(
        [[-0.5, -0.5, z, 1.0], [0.5, -0.5, z, 1.0], [-0.5, 0.5, z, 1.0]],
        dtype=mx.float32,
    )
    f = mx.array([[0, 1, 2]], dtype=mx.int32)
    return v, f


class TestPublicApi:
    def test_default_returns_two_arrays(self):
        v, f = _tri()
        out = rasterize_triangles(v, f, 32, 32)
        assert len(out) == 2
        fi, bary = out
        assert fi.shape == (32, 32)
        assert bary.shape == (32, 32, 3)
        assert fi.dtype == mx.int32

    def test_return_depth_adds_third_array(self):
        v, f = _tri()
        fi, bary, depth = rasterize_triangles(v, f, 32, 32, return_depth=True)
        assert depth.shape == (32, 32)
        assert depth.dtype == mx.float32
        covered = fi > 0
        assert bool(mx.any(covered).item())
        assert item_float(mx.max(mx.where(covered, depth, mx.zeros_like(depth)))) < 1.0

    def test_cull_none_is_the_default(self):
        v, f = _tri()
        a = rasterize_triangles(v, f, 32, 32)[0]
        b = rasterize_triangles(v, f, 32, 32, cull="none")[0]
        assert bool(mx.array_equal(a, b).item())

    def test_cull_removes_one_orientation(self):
        v, f = _tri()
        back = rasterize_triangles(v, f, 32, 32, cull="back")[0]
        front = rasterize_triangles(v, f, 32, 32, cull="front")[0]
        assert (item_int(mx.sum(back)) == 0) != (item_int(mx.sum(front)) == 0)

    def test_rejects_bad_cull_value(self):
        v, f = _tri()
        with pytest.raises(ValueError, match="cull"):
            rasterize_triangles(v, f, 32, 32, cull="sideways")  # ty: ignore[invalid-argument-type]

    def test_rejects_out_of_range_vertex_index(self):
        v, _ = _tri()
        bad = mx.array([[0, 1, 99]], dtype=mx.int32)
        with pytest.raises(ValueError, match="vertex index"):
            rasterize_triangles(v, bad, 32, 32)

    def test_rejects_oversized_image(self):
        v, f = _tri()
        with pytest.raises(ValueError, match="16384"):
            rasterize_triangles(v, f, 20000, 32)

    def test_zero_faces_returns_background(self):
        v, _ = _tri()
        empty = mx.zeros((0, 3), dtype=mx.int32)
        fi, bary = rasterize_triangles(v, empty, 16, 16)
        assert item_int(mx.sum(fi)) == 0
        assert item_float(mx.max(mx.abs(bary))) == 0.0

    def test_rejects_near_plane_vertex(self):
        # w = 1e-7 blows the projected fixed-point coordinate up to
        # int32-saturation magnitude, which is far past what int64 edge
        # functions can multiply without overflow.
        v = mx.array(
            [
                [-0.5, -0.5, 0.5, 1e-7],
                [0.5, -0.5, 0.5, 1.0],
                [-0.5, 0.5, 0.5, 1.0],
            ],
            dtype=mx.float32,
        )
        f = mx.array([[0, 1, 2]], dtype=mx.int32)
        with pytest.raises(ValueError, match="representable range"):
            rasterize_triangles(v, f, 32, 32)

    def test_rejects_near_plane_vertex_positive_saturation(self):
        # (x, y) = (1, 1) with w = 1e-7 projects to +INT32_MAX after
        # int32 saturation.
        v = mx.array(
            [
                [1.0, 1.0, 0.5, 1e-7],
                [0.5, -0.5, 0.5, 1.0],
                [-0.5, 0.5, 0.5, 1.0],
            ],
            dtype=mx.float32,
        )
        f = mx.array([[0, 1, 2]], dtype=mx.int32)
        with pytest.raises(ValueError, match="representable range"):
            rasterize_triangles(v, f, 32, 32)

    def test_rejects_near_plane_vertex_negative_saturation(self):
        # (x, y) = (-1, -1) with w = 1e-7 projects to -INT32_MAX-1 after
        # int32 saturation. `mx.abs` cannot represent `abs(INT32_MIN)` (it
        # saturates and stays negative), so a check built on `mx.abs` would
        # silently miss exactly this case.
        v = mx.array(
            [
                [-1.0, -1.0, 0.5, 1e-7],
                [0.5, -0.5, 0.5, 1.0],
                [-0.5, 0.5, 0.5, 1.0],
            ],
            dtype=mx.float32,
        )
        f = mx.array([[0, 1, 2]], dtype=mx.int32)
        with pytest.raises(ValueError, match="representable range"):
            rasterize_triangles(v, f, 32, 32)

    def test_ordinary_mesh_does_not_trigger_near_plane_check(self):
        v, f = _tri()
        # Should not raise.
        rasterize_triangles(v, f, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

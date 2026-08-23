import numpy as np

from tests.rasterize_oracle import coverage_count, rasterize_reference

S = 16  # sub-pixel scale, mirrors SUBPIXEL_SCALE


def _tri(pts_px, zs=(0.5, 0.5, 0.5)):
    """Build oracle inputs for one triangle given pixel-space float points."""
    verts_fx = np.round(np.array(pts_px, dtype=np.float64) * S).astype(np.int64)
    verts_zw = np.stack([np.array(zs), np.ones(3)], axis=1).astype(np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return verts_fx, verts_zw, faces


class TestReference:
    def test_covers_expected_pixels(self):
        """A large right triangle covers the pixels whose centres are inside."""
        verts_fx, verts_zw, faces = _tri([(0.0, 0.0), (8.0, 0.0), (0.0, 8.0)])
        fi, _, _ = rasterize_reference(verts_fx, verts_zw, faces, 8, 8)
        assert fi[0, 0] == 1  # centre (0.5, 0.5) is inside
        assert fi[7, 7] == 0  # centre (7.5, 7.5) is outside
        assert fi.dtype == np.int32

    def test_barycentrics_sum_to_one_on_covered(self):
        verts_fx, verts_zw, faces = _tri([(0.0, 0.0), (8.0, 0.0), (0.0, 8.0)])
        fi, bary, _ = rasterize_reference(verts_fx, verts_zw, faces, 8, 8)
        covered = fi > 0
        assert covered.any()
        assert np.allclose(bary[covered].sum(axis=-1), 1.0, atol=1e-12)

    def test_background_is_zero(self):
        verts_fx, verts_zw, faces = _tri([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        fi, bary, depth = rasterize_reference(verts_fx, verts_zw, faces, 8, 8)
        bg = fi == 0
        assert np.all(bary[bg] == 0.0)
        assert np.all(np.isinf(depth[bg]))

    def test_winding_independence(self):
        """Reversing winding must not change coverage."""
        verts_fx, verts_zw, faces = _tri([(0.0, 0.0), (8.0, 0.0), (0.0, 8.0)])
        fi_a, _, _ = rasterize_reference(verts_fx, verts_zw, faces, 8, 8)
        fi_b, _, _ = rasterize_reference(verts_fx, verts_zw, faces[:, ::-1].copy(), 8, 8)
        np.testing.assert_array_equal(fi_a, fi_b)

    def test_closer_face_wins(self):
        verts_fx = np.round(
            np.array([(0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (0.0, 0.0), (8.0, 0.0), (0.0, 8.0)]) * S
        ).astype(np.int64)
        verts_zw = np.stack([np.array([0.9, 0.9, 0.9, 0.2, 0.2, 0.2]), np.ones(6)], axis=1)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        fi, _, _ = rasterize_reference(verts_fx, verts_zw, faces, 8, 8)
        assert fi[0, 0] == 2  # the nearer (smaller z) triangle

    def test_cull_back_removes_one_winding(self):
        verts_fx, verts_zw, faces = _tri([(0.0, 0.0), (8.0, 0.0), (0.0, 8.0)])
        fi_none, _, _ = rasterize_reference(verts_fx, verts_zw, faces, 8, 8, cull="none")
        fi_back, _, _ = rasterize_reference(verts_fx, verts_zw, faces, 8, 8, cull="back")
        fi_front, _, _ = rasterize_reference(verts_fx, verts_zw, faces, 8, 8, cull="front")
        assert fi_none.any()
        # exactly one of back/front culling removes this triangle
        assert (not fi_back.any()) != (not fi_front.any())


class TestCoverageCount:
    def test_shared_edge_claimed_exactly_once(self):
        """Two triangles splitting a square claim every interior pixel once."""
        pts = [(0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (8.0, 8.0)]
        verts_fx = np.round(np.array(pts) * S).astype(np.int64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        counts = coverage_count(verts_fx, faces, 8, 8)
        assert counts.max() == 1
        assert counts.sum() > 0

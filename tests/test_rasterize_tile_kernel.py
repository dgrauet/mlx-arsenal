import mlx.core as mx
import numpy as np

from mlx_arsenal._typing import array_from_any, item_float, item_int
from mlx_arsenal.rasterize._binning import build_tile_lists, choose_tiling
from mlx_arsenal.rasterize._fixedpoint import to_screen
from mlx_arsenal.rasterize._tile_raster import raster_tiles
from tests.rasterize_oracle import rasterize_reference


def _run(vertices, faces, width, height, tile_size=16, cull="none"):
    verts_fx, verts_zw = to_screen(vertices, width, height)
    ts, bounds, n, total = choose_tiling(verts_fx, faces, width, height, cull, tile_size=tile_size)
    tiles_x = (width + ts - 1) // ts
    tiles_y = (height + ts - 1) // ts
    sf, starts = build_tile_lists(bounds, n, total, tiles_x, tiles_y, faces.shape[0])
    return raster_tiles(verts_fx, verts_zw, faces, sf, starts, width, height, ts, None, 1e-6)


def _oracle(vertices, faces, width, height, cull="none"):
    verts_fx, verts_zw = to_screen(vertices, width, height)
    return rasterize_reference(
        np.array(verts_fx.tolist(), dtype=np.int64),
        np.array(verts_zw.tolist(), dtype=np.float64),
        np.array(faces.tolist(), dtype=np.int64),
        width,
        height,
        cull,
    )


def _quad_mesh(n, seed=0):
    """Triangulated grid in clip space: many shared edges."""
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


class TestAgainstOracle:
    def test_single_triangle_matches(self):
        v = mx.array(
            [[-0.5, -0.5, 0.5, 1.0], [0.5, -0.5, 0.5, 1.0], [-0.5, 0.5, 0.5, 1.0]],
            dtype=mx.float32,
        )
        f = mx.array([[0, 1, 2]], dtype=mx.int32)
        fi, bary, _ = _run(v, f, 32, 32)
        ref_fi, ref_bary, _ = _oracle(v, f, 32, 32)
        np.testing.assert_array_equal(np.array(fi.tolist()), ref_fi)
        np.testing.assert_allclose(np.array(bary.tolist()), ref_bary, atol=1e-5)

    def test_grid_mesh_matches(self):
        v, f = _quad_mesh(6)
        fi, bary, _ = _run(v, f, 64, 64)
        ref_fi, ref_bary, _ = _oracle(v, f, 64, 64)
        np.testing.assert_array_equal(np.array(fi.tolist()), ref_fi)
        np.testing.assert_allclose(np.array(bary.tolist()), ref_bary, atol=1e-5)

    def test_depth_matches_oracle(self):
        v, f = _quad_mesh(4)
        _, _, depth = _run(v, f, 32, 32)
        _, _, ref_depth = _oracle(v, f, 32, 32)
        got = np.array(depth.tolist())
        finite = np.isfinite(ref_depth)
        np.testing.assert_allclose(got[finite], ref_depth[finite], atol=1e-5)
        assert np.all(np.isinf(got[~finite]))


class TestBackground:
    def test_empty_face_list_is_all_background(self):
        v = mx.array([[0.0, 0.0, 0.5, 1.0]] * 3, dtype=mx.float32)  # degenerate
        f = mx.array([[0, 1, 2]], dtype=mx.int32)
        fi, bary, depth = _run(v, f, 16, 16)
        assert item_int(mx.sum(fi)) == 0
        assert item_float(mx.max(mx.abs(bary))) == 0.0
        assert bool(mx.all(mx.isinf(depth)).item())


class TestDepthPrior:
    def test_prior_culls_near_geometry(self):
        v = mx.array(
            [[-0.5, -0.5, 0.5, 1.0], [0.5, -0.5, 0.5, 1.0], [-0.5, 0.5, 0.5, 1.0]],
            dtype=mx.float32,
        )
        f = mx.array([[0, 1, 2]], dtype=mx.int32)
        verts_fx, verts_zw = to_screen(v, 32, 32)
        ts, bounds, n, total = choose_tiling(verts_fx, f, 32, 32, "none", tile_size=16)
        sf, starts = build_tile_lists(bounds, n, total, 2, 2, 1)
        prior = mx.full((32, 32), 1.0, dtype=mx.float32)
        fi, _, _ = raster_tiles(verts_fx, verts_zw, f, sf, starts, 32, 32, ts, prior, 1e-6)
        assert item_int(mx.sum(fi)) == 0

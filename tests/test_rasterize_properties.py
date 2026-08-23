import mlx.core as mx
import numpy as np

from mlx_arsenal._typing import array_from_any
from mlx_arsenal.rasterize import rasterize_triangles
from mlx_arsenal.rasterize._fixedpoint import to_screen
from tests.rasterize_oracle import coverage_count


def _quad_mesh(n, seed=0):
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


class TestWatertightness:
    def test_no_holes_inside_the_mesh(self):
        """A closed triangulated grid leaves no background pixel in its interior."""
        v, f = _quad_mesh(8)
        fi, _ = rasterize_triangles(v, f, 128, 128)
        got = np.array(fi.tolist())
        covered = got > 0
        rows = np.where(covered.any(axis=1))[0]
        cols = np.where(covered.any(axis=0))[0]
        interior = covered[rows[1] : rows[-1], cols[1] : cols[-1]]
        assert interior.all(), f"{(~interior).sum()} hole pixels inside the mesh"

    def test_every_pixel_claimed_exactly_once(self):
        v, f = _quad_mesh(8)
        verts_fx, _ = to_screen(v, 128, 128)
        counts = coverage_count(
            np.array(verts_fx.tolist(), dtype=np.int64),
            np.array(f.tolist(), dtype=np.int64),
            128,
            128,
        )
        assert counts.max() <= 1, "a pixel is claimed by more than one face"


class TestTileSizeInvariance:
    def test_identical_across_tile_sizes(self):
        v, f = _quad_mesh(10)
        ref_fi, ref_bary = rasterize_triangles(v, f, 128, 128, _tile_size=16)
        for ts in (32, 64):
            fi, bary = rasterize_triangles(v, f, 128, 128, _tile_size=ts)
            assert bool(mx.array_equal(fi, ref_fi).item()), f"tile size {ts}"
            assert bool(mx.array_equal(bary, ref_bary).item()), f"tile size {ts}"


class TestDeterminism:
    def test_repeated_runs_are_identical(self):
        v, f = _quad_mesh(6)
        a_fi, a_bary = rasterize_triangles(v, f, 64, 64)
        for _ in range(3):
            fi, bary = rasterize_triangles(v, f, 64, 64)
            assert bool(mx.array_equal(fi, a_fi).item())
            assert bool(mx.array_equal(bary, a_bary).item())

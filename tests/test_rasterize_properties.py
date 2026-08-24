import mlx.core as mx
import numpy as np

from mlx_arsenal._typing import item_int
from mlx_arsenal.rasterize import rasterize_triangles
from mlx_arsenal.rasterize._fixedpoint import to_screen
from tests.rasterize_oracle import quad_mesh, rasterize_reference

# Deliberately not a multiple of the RASTER_TILE (16) sub-tile size or of the
# 32/64 binning tile sizes exercised below, so the right and bottom edges
# always fall inside a partial tile — the exact place a binning or mapping
# bug hides from dimensions that divide evenly.
PARTIAL_TILE_WIDTH, PARTIAL_TILE_HEIGHT = 100, 76


class TestWatertightness:
    def test_no_holes_inside_the_mesh(self):
        """A closed triangulated grid leaves no background pixel in its interior."""
        v, f = quad_mesh(8)
        fi, _ = rasterize_triangles(v, f, 128, 128)
        got = np.array(fi.tolist())
        covered = got > 0
        rows = np.where(covered.any(axis=1))[0]
        cols = np.where(covered.any(axis=0))[0]
        interior = covered[rows[1] : rows[-1], cols[1] : cols[-1]]
        assert interior.size > 0, "test proves nothing over an empty interior slice"
        assert interior.all(), f"{(~interior).sum()} hole pixels inside the mesh"

    def test_matches_oracle_on_partial_tiles(self):
        """The shipped pipeline's face indices must equal the independent oracle's,
        exactly, on a shared-edge grid at dimensions that leave partial tiles at
        the right and bottom edges.

        This is a property of `rasterize_triangles` itself (not of the oracle in
        isolation): any binning/claiming bug — including one confined to a
        partial tile — shows up as a face-index mismatch here.
        """
        v, f = quad_mesh(8)
        width, height = PARTIAL_TILE_WIDTH, PARTIAL_TILE_HEIGHT
        fi, _ = rasterize_triangles(v, f, width, height)
        verts_fx, verts_zw = to_screen(v, width, height)
        ref_fi, _, _ = rasterize_reference(
            np.array(verts_fx.tolist(), dtype=np.int64),
            np.array(verts_zw.tolist(), dtype=np.float64),
            np.array(f.tolist(), dtype=np.int64),
            width,
            height,
        )
        covered = int((ref_fi > 0).sum())
        assert covered > 0, "test proves nothing if no pixel is covered"
        np.testing.assert_array_equal(np.array(fi.tolist()), ref_fi)


class TestTileSizeInvariance:
    def test_identical_across_tile_sizes(self):
        v, f = quad_mesh(10)
        width, height = PARTIAL_TILE_WIDTH, PARTIAL_TILE_HEIGHT
        ref_fi, ref_bary = rasterize_triangles(v, f, width, height, _tile_size=16)
        covered = int((np.array(ref_fi.tolist()) > 0).sum())
        assert covered > 0, "test proves nothing if no pixel is covered"
        for ts in (32, 64):
            fi, bary = rasterize_triangles(v, f, width, height, _tile_size=ts)
            assert bool(mx.array_equal(fi, ref_fi).item()), f"tile size {ts}"
            assert bool(mx.array_equal(bary, ref_bary).item()), f"tile size {ts}"


class TestDeterminism:
    def test_repeated_runs_are_identical(self):
        """Proves determinism across repeated calls within one process — catches
        uninitialised memory and unordered reductions — but not process-level
        state such as first-call compile-cache population.
        """
        v, f = quad_mesh(6)
        a_fi, a_bary = rasterize_triangles(v, f, 64, 64)
        assert item_int(mx.sum(a_fi > 0)) > 0, "test proves nothing if no pixel is covered"
        for _ in range(3):
            fi, bary = rasterize_triangles(v, f, 64, 64)
            assert bool(mx.array_equal(fi, a_fi).item())
            assert bool(mx.array_equal(bary, a_bary).item())

import mlx.core as mx
import pytest

from mlx_arsenal._typing import item_int
from mlx_arsenal.rasterize._binning import (
    MAX_PAIRS,
    _floor_div_pow2,
    _shift_amount,
    build_tile_lists,
    choose_tiling,
    face_spans,
    signed_area,
)

S = 16


def _fx(pts):
    return mx.array([[int(round(x * S)), int(round(y * S))] for x, y in pts], dtype=mx.int32)


class TestSignedArea:
    def test_exact_and_signed(self):
        verts = _fx([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)])
        faces = mx.array([[0, 1, 2], [0, 2, 1]], dtype=mx.int32)
        a = signed_area(verts, faces)
        assert a.dtype == mx.int64
        assert item_int(a[0]) == -item_int(a[1])
        # |area| = 2 * triangle area in fixed-point units = 2*(4*16)^2/2
        assert abs(item_int(a[0])) == (4 * S) * (4 * S)

    def test_degenerate_is_zero(self):
        verts = _fx([(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        assert item_int(signed_area(verts, faces)[0]) == 0


class TestFaceSpans:
    def test_small_triangle_touches_one_tile(self):
        verts = _fx([(1.0, 1.0), (3.0, 1.0), (1.0, 3.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        bounds, n = face_spans(verts, faces, 64, 64, tile_size=16, cull="none")
        assert item_int(n[0]) == 1
        assert [item_int(bounds[0, i]) for i in range(4)] == [0, 0, 0, 0]

    def test_span_covers_multiple_tiles(self):
        verts = _fx([(0.0, 0.0), (40.0, 0.0), (0.0, 40.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        bounds, n = face_spans(verts, faces, 64, 64, tile_size=16, cull="none")
        assert [item_int(bounds[0, i]) for i in range(4)] == [0, 0, 2, 2]
        assert item_int(n[0]) == 9

    def test_degenerate_face_excluded(self):
        verts = _fx([(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        _, n = face_spans(verts, faces, 64, 64, tile_size=16, cull="none")
        assert item_int(n[0]) == 0

    def test_offscreen_face_excluded(self):
        verts = _fx([(100.0, 100.0), (110.0, 100.0), (100.0, 110.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        _, n = face_spans(verts, faces, 64, 64, tile_size=16, cull="none")
        assert item_int(n[0]) == 0

    def test_culling_excludes_by_orientation(self):
        verts = _fx([(1.0, 1.0), (9.0, 1.0), (1.0, 9.0)])
        cw = mx.array([[0, 1, 2]], dtype=mx.int32)
        ccw = mx.array([[0, 2, 1]], dtype=mx.int32)
        _, n_cw = face_spans(verts, cw, 64, 64, tile_size=16, cull="back")
        _, n_ccw = face_spans(verts, ccw, 64, 64, tile_size=16, cull="back")
        assert (item_int(n_cw[0]) == 0) != (item_int(n_ccw[0]) == 0)

    def test_clamps_to_screen(self):
        verts = _fx([(-50.0, -50.0), (10.0, -50.0), (-50.0, 10.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        bounds, n = face_spans(verts, faces, 64, 64, tile_size=16, cull="none")
        assert item_int(bounds[0, 0]) == 0
        assert item_int(bounds[0, 1]) == 0
        assert item_int(n[0]) >= 1


class TestFloorDivPow2:
    """Regression coverage for the mx.floor_divide-truncates-negatives bug.

    ``face_spans`` clamps every pixel bound to >= 0 before it becomes visible
    through the public API, so a regression here cannot be observed through
    ``face_spans`` itself (both a correct floor and a zero-toward truncation
    of a negative numerator clamp to the same 0). These tests pin the exact
    floor semantics on the private helper directly, so a regression to
    ``mx.floor_divide`` on a possibly-negative numerator fails immediately.
    """

    def test_matches_python_floor_division_for_negative_numerators(self):
        xs = [-33, -17, -16, -15, -1, 0, 1, 15, 16, 17, 33]
        x = mx.array(xs, dtype=mx.int32)
        result = _floor_div_pow2(x, bits=4)
        expected = [v // 16 for v in xs]
        assert [item_int(result[i]) for i in range(len(xs))] == expected

    def test_would_disagree_with_naive_floor_divide(self):
        # mx.floor_divide(-15, 16) truncates to 0; floor(-15 / 16) is -1.
        # This is exactly the case a regression back to mx.floor_divide breaks.
        x = mx.array([-15], dtype=mx.int32)
        assert item_int(_floor_div_pow2(x, bits=4)[0]) == -1
        assert item_int(mx.floor_divide(x, 16)[0]) == 0


class TestShiftAmount:
    def test_powers_of_two(self):
        assert _shift_amount(16) == 4
        assert _shift_amount(32) == 5
        assert _shift_amount(64) == 6
        assert _shift_amount(1) == 0

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError, match="power-of-two"):
            _shift_amount(48)

    def test_non_positive_raises(self):
        with pytest.raises(ValueError, match="power-of-two"):
            _shift_amount(0)


class TestChooseTiling:
    def test_picks_finest_tiling_when_it_fits(self):
        verts = _fx([(1.0, 1.0), (3.0, 1.0), (1.0, 3.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        ts, _, _, pairs = choose_tiling(verts, faces, 64, 64, "none")
        assert ts == 16
        assert pairs == 1

    def test_respects_explicit_tile_size(self):
        verts = _fx([(0.0, 0.0), (40.0, 0.0), (0.0, 40.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        ts, _, _, pairs = choose_tiling(verts, faces, 64, 64, "none", tile_size=32)
        assert ts == 32
        assert pairs == 4

    def test_zero_pairs_when_everything_excluded(self):
        verts = _fx([(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        ts, _, _, pairs = choose_tiling(verts, faces, 64, 64, "none")
        assert pairs == 0
        assert ts == 16

    def test_raises_when_budget_exceeded(self, monkeypatch):
        monkeypatch.setattr("mlx_arsenal.rasterize._binning.MAX_PAIRS", 4)
        verts = _fx([(0.0, 0.0), (4000.0, 0.0), (0.0, 4000.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        with pytest.raises(MemoryError, match="tile"):
            choose_tiling(verts, faces, 4096, 4096, "none")

    def test_budget_constant_matches_spec(self):
        assert MAX_PAIRS == 32 * 1024 * 1024


def _lists(verts, faces, width, height, tile_size):
    ts, bounds, n, total = choose_tiling(verts, faces, width, height, "none", tile_size=tile_size)
    tiles_x = (width + ts - 1) // ts
    tiles_y = (height + ts - 1) // ts
    sf, starts = build_tile_lists(bounds, n, total, tiles_x, tiles_y, faces.shape[0])
    return [
        sf[item_int(starts[t]) : item_int(starts[t + 1])].tolist() for t in range(tiles_x * tiles_y)
    ]


class TestBuildTileLists:
    def test_single_face_lands_in_its_tile_only(self):
        verts = _fx([(17.0, 17.0), (20.0, 17.0), (17.0, 20.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        per_tile = _lists(verts, faces, 64, 64, 16)
        assert per_tile[5] == [0]  # tile (1,1) of a 4x4 grid
        assert sum(len(t) for t in per_tile) == 1

    def test_faces_ascend_within_a_tile(self):
        pts, faces_list = [], []
        for k in range(3):
            base = len(pts)
            pts += [(1.0, 1.0), (5.0, 1.0), (1.0, 5.0)]
            faces_list.append([base, base + 1, base + 2])
        verts = _fx(pts)
        faces = mx.array(faces_list, dtype=mx.int32)
        per_tile = _lists(verts, faces, 64, 64, 16)
        assert per_tile[0] == [0, 1, 2]

    def test_spanning_face_appears_in_every_covered_tile(self):
        verts = _fx([(0.0, 0.0), (40.0, 0.0), (0.0, 40.0)])
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        per_tile = _lists(verts, faces, 64, 64, 16)
        covered = [i for i, t in enumerate(per_tile) if t]
        assert covered == [0, 1, 2, 4, 5, 6, 8, 9, 10]

    def test_empty_input_gives_empty_lists(self):
        verts = _fx([(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)])  # degenerate
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)
        per_tile = _lists(verts, faces, 64, 64, 16)
        assert all(t == [] for t in per_tile)

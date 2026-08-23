import mlx.core as mx
import pytest

from mlx_arsenal._typing import item_int
from mlx_arsenal.rasterize._binning import _floor_div_pow2, _shift_amount, face_spans, signed_area

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

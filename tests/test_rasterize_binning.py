import mlx.core as mx

from mlx_arsenal._typing import item_int
from mlx_arsenal.rasterize._binning import face_spans, signed_area

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

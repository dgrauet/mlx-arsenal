import mlx.core as mx
import pytest

from mlx_arsenal._typing import item_float, item_int
from mlx_arsenal.rasterize._fixedpoint import (
    MAX_DIM,
    SUBPIXEL_SCALE,
    pixel_center_fx,
    to_screen,
)


class TestToScreen:
    def test_ndc_centre_maps_to_image_centre(self):
        """A vertex at NDC (0,0,0,1) lands at the centre of the image."""
        v = mx.array([[0.0, 0.0, 0.0, 1.0]], dtype=mx.float32)
        fx, zw = to_screen(v, width=9, height=9)
        # (0 * 0.5 + 0.5) * (9 - 1) + 0.5 = 4.5 px -> 4.5 * 16 = 72
        assert item_int(fx[0, 0]) == 72
        assert item_int(fx[0, 1]) == 72
        assert item_float(zw[0, 0]) == pytest.approx(0.5)
        assert item_float(zw[0, 1]) == pytest.approx(1.0)

    def test_output_dtypes_and_shapes(self):
        v = mx.array([[0.0, 0.0, 0.0, 1.0], [0.5, -0.5, 0.25, 2.0]], dtype=mx.float32)
        fx, zw = to_screen(v, width=16, height=8)
        assert fx.shape == (2, 2)
        assert zw.shape == (2, 2)
        assert fx.dtype == mx.int32
        assert zw.dtype == mx.float32

    def test_perspective_divide_applied(self):
        """w != 1 must divide x/y/z before the screen mapping."""
        a = mx.array([[0.5, 0.0, 0.0, 1.0]], dtype=mx.float32)
        b = mx.array([[1.0, 0.0, 0.0, 2.0]], dtype=mx.float32)
        fa, _ = to_screen(a, 32, 32)
        fb, _ = to_screen(b, 32, 32)
        assert item_int(fa[0, 0]) == item_int(fb[0, 0])

    def test_snapping_is_to_sixteenths(self):
        """Coordinates are integers on a 1/16 px grid, not floats."""
        v = mx.array([[0.123, -0.456, 0.0, 1.0]], dtype=mx.float32)
        fx, _ = to_screen(v, 64, 64)
        # x_screen = (0.123 * 0.5 + 0.5) * 63 + 0.5 = 35.8745
        # x_fx = round(35.8745 * 16) = 574
        assert item_int(fx[0, 0]) == 574

    def test_rejects_oversized_image(self):
        v = mx.array([[0.0, 0.0, 0.0, 1.0]], dtype=mx.float32)
        with pytest.raises(ValueError, match="16384"):
            to_screen(v, width=MAX_DIM + 1, height=8)


class TestPixelCentre:
    def test_centre_of_pixel_zero(self):
        assert pixel_center_fx(0) == SUBPIXEL_SCALE // 2

    def test_centre_of_pixel_three(self):
        assert pixel_center_fx(3) == 3 * SUBPIXEL_SCALE + SUBPIXEL_SCALE // 2

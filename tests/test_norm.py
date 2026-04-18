"""Tests for norm module (PixelNorm, ScaleNorm)."""

import mlx.core as mx

from mlx_arsenal.norm import PixelNorm, ScaleNorm


class TestPixelNorm:
    def test_shape_preserved(self):
        layer = PixelNorm()
        x = mx.random.normal((2, 8, 16))
        out = layer(x)
        assert out.shape == x.shape

    def test_unit_rms_across_last_axis(self):
        layer = PixelNorm()
        x = mx.random.normal((4, 32))
        out = layer(x)
        rms = mx.sqrt(mx.mean(out * out, axis=-1))
        assert mx.allclose(rms, mx.ones_like(rms), atol=1e-4).item()

    def test_zero_input_stable(self):
        layer = PixelNorm(eps=1e-8)
        x = mx.zeros((2, 4))
        out = layer(x)
        assert mx.all(mx.isfinite(out)).item()


class TestScaleNorm:
    def test_shape_preserved(self):
        layer = ScaleNorm(dim=16)
        x = mx.random.normal((2, 4, 16))
        out = layer(x)
        assert out.shape == x.shape

    def test_unit_l2_with_default_scale(self):
        layer = ScaleNorm(dim=8)
        x = mx.random.normal((3, 8))
        out = layer(x)
        l2 = mx.sqrt(mx.sum(out * out, axis=-1))
        assert mx.allclose(l2, mx.ones_like(l2), atol=1e-4).item()

    def test_scale_is_applied(self):
        layer = ScaleNorm(dim=4)
        layer.scale = mx.full((4,), 2.0)
        x = mx.random.normal((2, 4))
        out = layer(x)
        l2 = mx.sqrt(mx.sum(out * out, axis=-1))
        assert mx.allclose(l2, mx.full((2,), 2.0), atol=1e-4).item()

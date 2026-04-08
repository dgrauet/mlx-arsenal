"""Tests for norm module."""

import mlx.core as mx
import pytest

from mlx_ops.norm import (
    AdaLayerNormZero,
    AdaLayerNormSingle,
    AdaLayerNormContinuous,
    PixelNorm,
    ScaleNorm,
)


class TestAdaLayerNormZero:
    def test_output_count(self):
        norm = AdaLayerNormZero(64)
        x = mx.random.normal((2, 10, 64))
        cond = mx.random.normal((2, 64))
        result = norm(x, cond)
        assert len(result) == 7  # normed, shift_a, scale_a, gate_a, shift_m, scale_m, gate_m

    def test_shapes(self):
        norm = AdaLayerNormZero(64, cond_dim=128)
        x = mx.random.normal((2, 10, 64))
        cond = mx.random.normal((2, 128))
        normed, *mods = norm(x, cond)
        mx.eval(normed, *mods)
        assert normed.shape == (2, 10, 64)
        for m in mods:
            assert m.shape[-1] == 64


class TestAdaLayerNormSingle:
    def test_output_count(self):
        norm = AdaLayerNormSingle(64)
        x = mx.random.normal((2, 10, 64))
        cond = mx.random.normal((2, 64))
        result = norm(x, cond)
        assert len(result) == 4  # normed, shift, scale, gate

    def test_shapes(self):
        norm = AdaLayerNormSingle(64)
        x = mx.random.normal((2, 10, 64))
        cond = mx.random.normal((2, 64))
        normed, shift, scale, gate = norm(x, cond)
        mx.eval(normed, shift, scale, gate)
        assert normed.shape == (2, 10, 64)
        assert shift.shape[-1] == 64


class TestAdaLayerNormContinuous:
    def test_output_shape(self):
        norm = AdaLayerNormContinuous(64)
        x = mx.random.normal((2, 10, 64))
        cond = mx.random.normal((2, 64))
        out = norm(x, cond)
        mx.eval(out)
        assert out.shape == (2, 10, 64)

    def test_with_different_cond_dim(self):
        norm = AdaLayerNormContinuous(64, cond_dim=256)
        x = mx.random.normal((2, 10, 64))
        cond = mx.random.normal((2, 256))
        out = norm(x, cond)
        mx.eval(out)
        assert out.shape == (2, 10, 64)


class TestPixelNorm:
    def test_output_shape(self):
        norm = PixelNorm()
        x = mx.random.normal((2, 10, 64))
        out = norm(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_unit_norm(self):
        norm = PixelNorm()
        x = mx.random.normal((2, 10, 64))
        out = norm(x)
        mx.eval(out)
        norms = mx.sqrt(mx.mean(out * out, axis=-1))
        mx.eval(norms)
        assert mx.allclose(norms, mx.ones_like(norms), atol=1e-3)


class TestScaleNorm:
    def test_output_shape(self):
        norm = ScaleNorm(64)
        x = mx.random.normal((2, 10, 64))
        out = norm(x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_initial_unit_norm(self):
        norm = ScaleNorm(64)
        x = mx.random.normal((2, 10, 64))
        out = norm(x)
        mx.eval(out)
        # With initial scale=1, output should have unit L2 norm per vector
        norms = mx.sqrt(mx.sum(out * out, axis=-1))
        mx.eval(norms)
        assert mx.allclose(norms, mx.ones_like(norms), atol=1e-3)

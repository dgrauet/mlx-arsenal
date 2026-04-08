"""Tests for tiling module."""

import mlx.core as mx
import pytest

from mlx_ops.tiling import tiled_process, temporal_slice_process


class TestTiledProcess:
    def test_small_input_passthrough(self):
        """Input smaller than tile_size should be processed directly."""
        x = mx.random.normal((1, 4, 4, 3))
        out = tiled_process(x, fn=lambda t: t * 2, tile_size=8, overlap=2)
        mx.eval(out)
        expected = x * 2
        mx.eval(expected)
        assert mx.allclose(out, expected, atol=1e-5)

    def test_tiled_shape(self):
        """Output should match the shape of fn(full_input)."""
        x = mx.random.normal((1, 16, 16, 3))
        out = tiled_process(x, fn=lambda t: t, tile_size=8, overlap=2)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 3)

    def test_identity_function(self):
        """Tiled identity should approximately reproduce the input."""
        x = mx.random.normal((1, 16, 16, 3))
        out = tiled_process(x, fn=lambda t: t, tile_size=8, overlap=4)
        mx.eval(out)
        # Should be very close to input since we're applying identity
        assert mx.allclose(out, x, atol=1e-4)


class TestTemporalSliceProcess:
    def test_small_input_passthrough(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        out = temporal_slice_process(x, fn=lambda t: t * 2, window_size=8)
        mx.eval(out)
        expected = x * 2
        mx.eval(expected)
        assert mx.allclose(out, expected, atol=1e-5)

    def test_temporal_shape(self):
        x = mx.random.normal((1, 32, 4, 4, 3))
        out = temporal_slice_process(x, fn=lambda t: t, window_size=8, overlap=2)
        mx.eval(out)
        assert out.shape == (1, 32, 4, 4, 3)

    def test_identity_function(self):
        x = mx.random.normal((1, 16, 4, 4, 3))
        out = temporal_slice_process(x, fn=lambda t: t, window_size=8, overlap=4)
        mx.eval(out)
        assert mx.allclose(out, x, atol=1e-4)

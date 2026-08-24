"""Tests for interpolation and pooling ops."""

import mlx.core as mx
import pytest

from mlx_arsenal._typing import item_float
from mlx_arsenal.spatial import avg_pool1d, interpolate_3d, interpolate_nearest, replicate_pad


class TestInterpolateNearest:
    def test_4d_upsample(self):
        x = mx.random.normal((1, 4, 4, 3))
        out = interpolate_nearest(x, scale_factor=2)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 3)

    def test_4d_downsample(self):
        x = mx.random.normal((1, 8, 8, 3))
        out = interpolate_nearest(x, size=(4, 4))
        mx.eval(out)
        assert out.shape == (1, 4, 4, 3)

    def test_5d_upsample(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        out = interpolate_nearest(x, scale_factor=2)
        mx.eval(out)
        assert out.shape == (1, 8, 16, 16, 3)

    def test_5d_target_size(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        out = interpolate_nearest(x, size=(2, 4, 4))
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 3)

    def test_3d(self):
        x = mx.random.normal((2, 10, 16))
        out = interpolate_nearest(x, size=(20,))
        mx.eval(out)
        assert out.shape == (2, 20, 16)

    def test_identity(self):
        x = mx.random.normal((1, 4, 4, 3))
        out = interpolate_nearest(x, size=(4, 4))
        mx.eval(out)
        assert mx.allclose(x, out).item()

    def test_values_replicate(self):
        x = mx.arange(4).reshape(1, 4, 1).astype(mx.float32)
        out = interpolate_nearest(x, scale_factor=2)
        mx.eval(out)
        assert out[0, 0, 0].item() == 0.0
        assert out[0, 1, 0].item() == 0.0
        assert out[0, 2, 0].item() == 1.0

    def test_noop_when_no_args(self):
        x = mx.random.normal((1, 4, 4, 3))
        out = interpolate_nearest(x)
        assert out.shape == x.shape
        assert mx.array_equal(out, x).item()

    def test_int_size_broadcasts(self):
        x = mx.random.normal((1, 4, 4, 3))
        out = interpolate_nearest(x, size=8)
        assert out.shape == (1, 8, 8, 3)


class TestInterpolate3d:
    def test_shape(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        out = interpolate_3d(x, size=(2, 4, 4))
        mx.eval(out)
        assert out.shape == (1, 2, 4, 4, 3)

    def test_identity(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        out = interpolate_3d(x, size=(4, 8, 8))
        mx.eval(out)
        assert mx.allclose(x, out).item()


class TestAvgPool1d:
    def test_basic(self):
        x = mx.ones((2, 8, 3))
        out = avg_pool1d(x, kernel_size=2)
        mx.eval(out)
        assert out.shape == (2, 4, 3)
        assert mx.allclose(out, mx.ones((2, 4, 3))).item()

    def test_stride(self):
        x = mx.random.normal((1, 10, 4))
        out = avg_pool1d(x, kernel_size=3, stride=2)
        mx.eval(out)
        assert out.shape == (1, 4, 4)

    def test_values(self):
        x = mx.arange(6).reshape(1, 6, 1).astype(mx.float32)
        out = avg_pool1d(x, kernel_size=2)
        mx.eval(out)
        assert out[0, 0, 0].item() == pytest.approx(0.5)
        assert out[0, 1, 0].item() == pytest.approx(2.5)
        assert out[0, 2, 0].item() == pytest.approx(4.5)


class TestReplicatePad:
    def test_1d(self):
        x = mx.array([[1.0, 2.0, 3.0]])
        out = replicate_pad(x, [(0, 0), (2, 1)])
        mx.eval(out)
        assert out.shape == (1, 6)
        assert out[0, 0].item() == 1.0
        assert out[0, 1].item() == 1.0
        assert out[0, 5].item() == 3.0

    def test_4d_spatial(self):
        x = mx.random.normal((1, 4, 4, 3))
        out = replicate_pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        mx.eval(out)
        assert out.shape == (1, 6, 6, 3)

    def test_5d_temporal(self):
        x = mx.random.normal((1, 4, 8, 8, 3))
        out = replicate_pad(x, [(0, 0), (2, 0), (0, 0), (0, 0), (0, 0)])
        mx.eval(out)
        assert out.shape == (1, 6, 8, 8, 3)
        assert mx.allclose(out[:, 0], out[:, 1]).item()
        assert mx.allclose(out[:, 0], x[:, 0]).item()


class TestInterpolateNearestSizeLengthMismatch:
    # interpolate_nearest does not validate len(size) against the number of
    # spatial dims — a defect: a too-short tuple crashes with a bare
    # IndexError deep in the loop, and a too-long tuple silently ignores the
    # extra entries instead of rejecting them.
    def test_short_size_tuple_crashes(self):
        x = mx.random.normal((1, 4, 4, 3))  # two spatial dims
        with pytest.raises(IndexError):
            interpolate_nearest(x, size=(2,))

    def test_long_size_tuple_silently_ignores_extras(self):
        x = mx.random.normal((1, 4, 4, 3))
        out = interpolate_nearest(x, size=(2, 2, 99))  # third entry ignored
        assert out.shape == (1, 2, 2, 3)


class TestAvgPool1dRemainder:
    def test_length_not_divisible_by_stride(self):
        """L=7, k=2, s=2 -> L_out = (7-2)//2 + 1 = 3; the tail element is
        dropped, matching torch.nn.functional.avg_pool1d."""
        x = mx.arange(7).reshape(1, 7, 1).astype(mx.float32)
        out = avg_pool1d(x, kernel_size=2, stride=2)
        assert out.shape == (1, 3, 1)
        for i in range(3):
            start = i * 2
            manual = sum(item_float(x[0, start + j, 0]) for j in range(2)) / 2
            assert out[0, i, 0].item() == pytest.approx(manual)

    def test_kernel_larger_than_stride_remainder(self):
        """L=10, k=3, s=4 -> L_out = (10-3)//4 + 1 = 2, overlappingly strided."""
        x = mx.arange(10).reshape(1, 10, 1).astype(mx.float32)
        out = avg_pool1d(x, kernel_size=3, stride=4)
        assert out.shape == (1, 2, 1)
        assert out[0, 0, 0].item() == pytest.approx(1.0)  # mean(0,1,2)
        assert out[0, 1, 0].item() == pytest.approx(5.0)  # mean(4,5,6)

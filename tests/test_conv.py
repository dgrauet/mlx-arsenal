"""Tests for conv module."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_ops.conv import CausalConv1d, CausalConv2d, CausalConv3d, weight_norm


class TestCausalConv1d:
    def test_output_shape(self):
        conv = CausalConv1d(4, 8, kernel_size=3)
        x = mx.random.normal((2, 10, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 10, 8), f"Expected (2, 10, 8), got {out.shape}"

    def test_causal_no_future_leak(self):
        """Output at position t should not depend on inputs at t+1."""
        conv = CausalConv1d(1, 1, kernel_size=3, padding_mode="zero")
        x = mx.zeros((1, 5, 1))
        # Set a value at position 3
        x = x.at[0, 3, 0].add(1.0)
        out = conv(x)
        mx.eval(out)
        # Positions 0, 1, 2 should be unaffected by the impulse at position 3
        for t in range(3):
            val_before = conv(mx.zeros((1, 5, 1)))
            mx.eval(val_before)
            assert mx.allclose(out[0, t, :], val_before[0, t, :])

    def test_stride(self):
        conv = CausalConv1d(4, 8, kernel_size=3, stride=2)
        x = mx.random.normal((2, 10, 4))
        out = conv(x)
        mx.eval(out)
        # With causal padding of 2 on left: input becomes 12, stride 2 -> floor(12/2) = 5 or 6
        assert out.shape[1] in (5, 6)

    def test_replicate_padding(self):
        conv = CausalConv1d(4, 8, kernel_size=3, padding_mode="replicate")
        x = mx.random.normal((2, 10, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 10, 8)


class TestCausalConv2d:
    def test_output_shape(self):
        conv = CausalConv2d(4, 8, kernel_size=3)
        x = mx.random.normal((2, 10, 10, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 10, 10, 8)

    def test_asymmetric_kernel(self):
        conv = CausalConv2d(4, 8, kernel_size=(3, 5))
        x = mx.random.normal((2, 10, 10, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 10, 10, 8)


class TestCausalConv3d:
    def test_output_shape(self):
        conv = CausalConv3d(4, 8, kernel_size=3)
        x = mx.random.normal((1, 4, 8, 8, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 8)

    def test_asymmetric_kernel(self):
        conv = CausalConv3d(4, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        x = mx.random.normal((1, 4, 8, 8, 4))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 8)

    def test_replicate_mode(self):
        conv = CausalConv3d(2, 4, kernel_size=3, padding_mode="replicate")
        x = mx.random.normal((1, 4, 6, 6, 2))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 4, 6, 6, 4)


class TestWeightNorm:
    def test_linear(self):
        linear = nn.Linear(8, 16)
        wn = weight_norm(linear)
        x = mx.random.normal((2, 8))
        out = wn(x)
        mx.eval(out)
        assert out.shape == (2, 16)

    def test_conv1d(self):
        conv = nn.Conv1d(4, 8, 3, padding=1)
        wn = weight_norm(conv)
        x = mx.random.normal((2, 10, 4))
        out = wn(x)
        mx.eval(out)
        assert out.shape == (2, 10, 8)

    def test_weight_normalized(self):
        """After applying weight norm, direction should be unit norm."""
        linear = nn.Linear(8, 4)
        wn = weight_norm(linear, dim=0)
        w = wn._compute_weight()
        mx.eval(w)
        # Each row (output channel) should have norm == g
        norms = mx.sqrt(mx.sum(w * w, axis=1))
        expected = wn.g.squeeze()
        mx.eval(norms, expected)
        assert mx.allclose(norms, expected, atol=1e-5)

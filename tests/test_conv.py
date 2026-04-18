"""Tests for conv module (weight normalization)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_arsenal.conv import weight_norm


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
        norms = mx.sqrt(mx.sum(w * w, axis=1))
        expected = wn.g.squeeze()
        mx.eval(norms, expected)
        assert mx.allclose(norms, expected, atol=1e-5)

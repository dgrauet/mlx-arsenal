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
        assert mx.allclose(norms, expected, atol=1e-5).item()


class TestWeightNormStateIntegrity:
    def test_wrapped_weight_not_shadowed_after_call(self):
        lin = nn.Linear(4, 4)
        wn = weight_norm(lin)
        wn(mx.ones((1, 4)))
        # The wrapper must not leave an instance-dict entry that shadows the
        # module's parameter dict (it would freeze `lin.weight` forever).
        assert "weight" not in lin.__dict__

    def test_update_after_call_is_visible(self):
        lin = nn.Linear(4, 4)
        wn = weight_norm(lin)
        wn(mx.ones((1, 4)))
        new_w = mx.zeros((4, 4))
        lin.update({"weight": new_w})
        assert mx.array_equal(lin.weight, new_w).item()

    def test_output_matches_reference(self):
        lin = nn.Linear(4, 3)
        wn = weight_norm(lin)
        x = mx.random.normal((2, 4))
        v = wn.v
        g = wn.g
        norm = mx.sqrt(mx.sum(v * v, axis=1, keepdims=True) + 1e-12)
        expected = x @ (g * (v / norm)).T + lin.bias
        assert mx.allclose(wn(x), expected, atol=1e-6).item()

    def test_weight_restored_after_module_raises(self):
        class Boom(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mx.ones((2, 2))

            def __call__(self, x):
                raise RuntimeError("boom")

        boom = Boom()
        original = boom.weight
        wn = weight_norm(boom)
        with pytest.raises(RuntimeError, match="boom"):
            wn(mx.ones((1, 2)))
        assert mx.array_equal(boom.weight, original).item()
        assert "weight" not in boom.__dict__


class TestWeightNormVariants:
    def test_dim_1_normalizes_over_other_axes(self):
        """dim=1 keeps per-input-channel magnitudes: norm over axis 0."""
        lin = nn.Linear(4, 3)
        wn = weight_norm(lin, dim=1)
        assert wn.g.shape == (1, 4)
        x = mx.random.normal((2, 4))
        v = wn.v
        norm = mx.sqrt(mx.sum(v * v, axis=0, keepdims=True) + 1e-12)
        expected = x @ (wn.g * (v / norm)).T + lin.bias
        assert mx.allclose(wn(x), expected, atol=1e-6).item()

    def test_custom_weight_name(self):
        class Kernelized(nn.Module):
            def __init__(self):
                super().__init__()
                self.kernel = mx.random.normal((3, 4))

            def __call__(self, x: mx.array) -> mx.array:
                return x @ self.kernel.T

        mod = Kernelized()
        original = mod.kernel
        wn = weight_norm(mod, weight_name="kernel")
        x = mx.random.normal((2, 4))
        norm = mx.sqrt(mx.sum(wn.v * wn.v, axis=1, keepdims=True) + 1e-12)
        expected = x @ (wn.g * (wn.v / norm)).T
        assert mx.allclose(wn(x), expected, atol=1e-6).item()
        # Wrapped param untouched after the call.
        assert mx.array_equal(mod.kernel, original).item()

    def test_reentrant_calls_identical_and_side_effect_free(self):
        lin = nn.Linear(4, 3)
        wn = weight_norm(lin)
        original = lin.weight
        x = mx.random.normal((2, 4))
        out1 = wn(x)
        out2 = wn(x)
        assert mx.allclose(out1, out2, atol=1e-6).item()
        assert mx.array_equal(lin.weight, original).item()
        assert "weight" not in lin.__dict__

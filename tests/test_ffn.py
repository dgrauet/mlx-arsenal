"""Tests for ffn module."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest

from mlx_arsenal.ffn import FeedForward, GatedFFN, GeGLU, SwiGLU


class TestFeedForward:
    def test_shape_default(self):
        ff = FeedForward(dim=8)
        x = mx.random.normal((2, 4, 8))
        y = ff(x)
        assert y.shape == x.shape

    def test_inner_dim_from_mult(self):
        ff = FeedForward(dim=8, mult=2.0)
        # proj_in: (8 → 16); proj_out: (16 → 8)
        assert ff.proj_in.weight.shape == (16, 8)
        assert ff.proj_out.weight.shape == (8, 16)

    def test_explicit_inner_dim_overrides_mult(self):
        ff = FeedForward(dim=8, mult=4.0, inner_dim=10)
        assert ff.proj_in.weight.shape == (10, 8)

    def test_dim_out_asymmetric(self):
        ff = FeedForward(dim=8, dim_out=16)
        x = mx.random.normal((1, 2, 8))
        y = ff(x)
        assert y.shape == (1, 2, 16)

    @pytest.mark.parametrize("activation", ["gelu", "gelu_approx", "silu", "relu"])
    def test_named_activations(self, activation):
        ff = FeedForward(dim=4, activation=activation)
        y = ff(mx.random.normal((1, 4)))
        assert y.shape == (1, 4)

    def test_callable_activation(self):
        ff = FeedForward(dim=4, activation=lambda x: x * 2)
        x = mx.ones((1, 4))
        y = ff(x)
        # Manually compute the same thing.
        inner = ff.proj_in(x)
        expected = ff.proj_out(inner * 2)
        assert mx.allclose(y, expected).item()

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="unknown activation"):
            FeedForward(dim=4, activation="banana")

    def test_no_bias(self):
        ff = FeedForward(dim=4, bias=False)
        keys = {name for name, _ in mlx.utils.tree_flatten(ff.parameters())}
        assert "proj_in.weight" in keys
        assert "proj_in.bias" not in keys
        assert "proj_out.bias" not in keys

    def test_weight_keys(self):
        ff = FeedForward(dim=4)
        keys = {name for name, _ in mlx.utils.tree_flatten(ff.parameters())}
        assert keys == {"proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias"}


class TestGatedFFN:
    def test_shape_default(self):
        ff = GatedFFN(dim=8)
        x = mx.random.normal((2, 4, 8))
        y = ff(x)
        assert y.shape == x.shape

    def test_three_projections(self):
        ff = GatedFFN(dim=8, mult=2.0)
        assert ff.gate_proj.weight.shape == (16, 8)
        assert ff.up_proj.weight.shape == (16, 8)
        assert ff.down_proj.weight.shape == (8, 16)

    def test_weight_keys_no_bias(self):
        ff = GatedFFN(dim=8)
        keys = {name for name, _ in mlx.utils.tree_flatten(ff.parameters())}
        assert keys == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}

    def test_weight_keys_with_bias(self):
        ff = GatedFFN(dim=8, bias=True)
        keys = {name for name, _ in mlx.utils.tree_flatten(ff.parameters())}
        assert "gate_proj.bias" in keys
        assert "up_proj.bias" in keys
        assert "down_proj.bias" in keys

    def test_gating_math(self):
        """Output should equal down(silu(gate(x)) * up(x))."""
        ff = GatedFFN(dim=4, mult=1.0, gate_activation="silu")
        x = mx.random.normal((1, 4))
        gated = nn.silu(ff.gate_proj(x)) * ff.up_proj(x)
        expected = ff.down_proj(gated)
        assert mx.allclose(ff(x), expected).item()

    def test_gelu_variant(self):
        ff = GatedFFN(dim=4, gate_activation="gelu")
        # Sanity: forward runs and shape preserved.
        assert ff(mx.random.normal((1, 4))).shape == (1, 4)


class TestAliases:
    def test_geglu_uses_gelu_gate(self):
        ff = GeGLU(dim=4)
        # Numerically verify by computing the GeGLU formula directly.
        x = mx.random.normal((1, 4))
        expected = ff.down_proj(nn.gelu(ff.gate_proj(x)) * ff.up_proj(x))
        assert mx.allclose(ff(x), expected).item()

    def test_swiglu_uses_silu_gate(self):
        ff = SwiGLU(dim=4)
        x = mx.random.normal((1, 4))
        expected = ff.down_proj(nn.silu(ff.gate_proj(x)) * ff.up_proj(x))
        assert mx.allclose(ff(x), expected).item()

    def test_aliases_return_gated_ffn(self):
        assert isinstance(GeGLU(4), GatedFFN)
        assert isinstance(SwiGLU(4), GatedFFN)

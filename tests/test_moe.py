from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_arsenal.moe import MoEGate, MoELayer


class TestImports:
    def test_import_moe_gate(self):
        assert MoEGate is not None

    def test_import_moe_layer(self):
        assert MoELayer is not None


class TestMoEGate:
    def test_output_shapes(self):
        """Gate returns top-k indices and weights with correct shapes."""
        gate = MoEGate(hidden_size=64, num_experts=8, top_k=2)
        x = mx.random.normal((4, 10, 64))  # batch=4, seq=10, hidden=64
        flat_x = x.reshape(-1, 64)  # (40, 64)
        indices, weights = gate(flat_x)
        assert indices.shape == (40, 2), f"Expected (40, 2), got {indices.shape}"
        assert weights.shape == (40, 2), f"Expected (40, 2), got {weights.shape}"

    def test_indices_in_range(self):
        """All expert indices are within [0, num_experts)."""
        gate = MoEGate(hidden_size=32, num_experts=4, top_k=2)
        x = mx.random.normal((20, 32))
        indices, _ = gate(x)
        mx.eval(indices)
        assert mx.all(indices >= 0).item()
        assert mx.all(indices < 4).item()

    def test_weights_are_positive(self):
        """Routing weights are positive (come from softmax)."""
        gate = MoEGate(hidden_size=32, num_experts=4, top_k=2)
        x = mx.random.normal((20, 32))
        _, weights = gate(x)
        mx.eval(weights)
        assert mx.all(weights > 0).item()

    def test_top_k_1(self):
        """Works with top_k=1."""
        gate = MoEGate(hidden_size=32, num_experts=8, top_k=1)
        x = mx.random.normal((10, 32))
        indices, weights = gate(x)
        assert indices.shape == (10, 1)
        assert weights.shape == (10, 1)


class TestMoELayer:
    def _make_expert_fn(self, hidden_size):
        """Factory that creates a simple FFN expert."""

        def expert_fn():
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        return expert_fn

    def test_output_shape(self):
        """MoELayer preserves input shape."""
        hidden = 64
        layer = MoELayer(
            hidden_size=hidden,
            num_experts=4,
            top_k=2,
            expert_fn=self._make_expert_fn(hidden),
        )
        x = mx.random.normal((4, 10, hidden))
        y = layer(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_output_shape_with_shared_expert(self):
        """MoELayer with shared expert preserves input shape."""
        hidden = 64
        shared = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        layer = MoELayer(
            hidden_size=hidden,
            num_experts=4,
            top_k=2,
            expert_fn=self._make_expert_fn(hidden),
            shared_expert=shared,
        )
        x = mx.random.normal((4, 10, hidden))
        y = layer(x)
        assert y.shape == x.shape

    def test_different_num_experts(self):
        """Works with different expert counts."""
        for n_experts in [2, 4, 8, 16]:
            hidden = 32
            layer = MoELayer(
                hidden_size=hidden,
                num_experts=n_experts,
                top_k=2,
                expert_fn=self._make_expert_fn(hidden),
            )
            x = mx.random.normal((2, 5, hidden))
            y = layer(x)
            mx.eval(y)
            assert y.shape == x.shape

    def test_top_k_1(self):
        """Works with single expert selection."""
        hidden = 32
        layer = MoELayer(
            hidden_size=hidden,
            num_experts=4,
            top_k=1,
            expert_fn=self._make_expert_fn(hidden),
        )
        x = mx.random.normal((2, 5, hidden))
        y = layer(x)
        assert y.shape == x.shape

    def test_single_token(self):
        """Works with single token input."""
        hidden = 32
        layer = MoELayer(
            hidden_size=hidden,
            num_experts=4,
            top_k=2,
            expert_fn=self._make_expert_fn(hidden),
        )
        x = mx.random.normal((1, 1, hidden))
        y = layer(x)
        assert y.shape == x.shape

    def test_dtype_preserved_fp16(self):
        """fp16 input stays fp16 through routing (no silent fp32 promotion).

        Regression: the routing one-hot/accumulator buffers were created as
        default-fp32 zeros, promoting the whole layer output to fp32 for
        fp16 models (found via smeltr profiling of Hunyuan3D-2.1-mlx).
        """
        hidden = 32
        layer = MoELayer(
            hidden_size=hidden,
            num_experts=4,
            top_k=2,
            expert_fn=self._make_expert_fn(hidden),
        )
        layer.set_dtype(mx.float16)
        x = mx.random.normal((2, 5, hidden), dtype=mx.float16)
        y = layer(x)
        assert y.dtype == mx.float16


class _IdentityExpert(nn.Module):
    """Expert that returns its input unchanged."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class TestMoEGateFullSelection:
    def test_top_k_equals_num_experts_argsort_branch(self):
        # top_k == num_experts exercises the argsort branch: indices are a
        # permutation of all experts and the gathered weights are the full
        # softmax row (sum to 1).
        gate = MoEGate(hidden_size=16, num_experts=4, top_k=4)
        x = mx.random.normal((10, 16), key=mx.random.key(0))
        indices, weights = gate(x)
        assert indices.shape == (10, 4)
        assert weights.shape == (10, 4)
        for row in cast(list[list[int]], indices.tolist()):
            assert sorted(row) == [0, 1, 2, 3]
        assert mx.allclose(mx.sum(weights, axis=-1), mx.ones((10,)), atol=1e-5).item()

    def test_weights_match_manual_softmax_gather(self):
        gate = MoEGate(hidden_size=16, num_experts=4, top_k=4)
        x = mx.random.normal((10, 16), key=mx.random.key(1))
        indices, weights = gate(x)
        scores = mx.softmax(gate.gate(x), axis=-1)
        expected = mx.take_along_axis(scores, indices, axis=-1)
        assert mx.allclose(weights, expected, atol=1e-6).item()


class TestMoELayerWeightedSum:
    def test_full_selection_identity_experts_is_identity(self):
        # With identity experts, output = sum_k(weight_k) * x. When
        # top_k == num_experts the weights are a full softmax row (sum 1),
        # so the layer is exactly the identity.
        layer = MoELayer(
            hidden_size=8,
            num_experts=3,
            top_k=3,
            expert_fn=_IdentityExpert,
        )
        x = mx.random.normal((2, 5, 8), key=mx.random.key(2))
        y = layer(x)
        assert mx.allclose(y, x, atol=1e-5).item()

    def test_shared_expert_contribution_adds(self):
        # Identity routed experts (weights sum to 1) + identity shared
        # expert → output == x + x == 2x.
        layer = MoELayer(
            hidden_size=8,
            num_experts=3,
            top_k=3,
            expert_fn=_IdentityExpert,
            shared_expert=_IdentityExpert(),
        )
        x = mx.random.normal((2, 5, 8), key=mx.random.key(3))
        y = layer(x)
        assert mx.allclose(y, 2.0 * x, atol=1e-5).item()

    def test_partial_top_k_output_scales_by_routing_weight_sum(self):
        # With top_k < num_experts and identity experts, each token's output
        # is (sum of its top-k routing weights) * x.
        layer = MoELayer(
            hidden_size=8,
            num_experts=4,
            top_k=2,
            expert_fn=_IdentityExpert,
        )
        x = mx.random.normal((3, 4, 8), key=mx.random.key(4))
        hidden = x.reshape(-1, 8)
        _, weights = layer.gate(hidden)
        expected = (mx.sum(weights, axis=-1, keepdims=True) * hidden).reshape(x.shape)
        y = layer(x)
        assert mx.allclose(y, expected, atol=1e-5).item()


class TestMoEGateValidation:
    def test_top_k_greater_than_num_experts_raises(self):
        with pytest.raises(ValueError):
            MoEGate(hidden_size=4, num_experts=3, top_k=5)

    def test_nonpositive_top_k_raises(self):
        with pytest.raises(ValueError):
            MoEGate(hidden_size=4, num_experts=3, top_k=0)

    def test_nonpositive_num_experts_raises(self):
        with pytest.raises(ValueError):
            MoEGate(hidden_size=4, num_experts=0, top_k=1)

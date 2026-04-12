import mlx.core as mx
import mlx.nn as nn
from mlx_ops.moe import MoEGate, MoELayer


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

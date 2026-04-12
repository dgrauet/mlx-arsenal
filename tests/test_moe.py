import mlx.core as mx
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

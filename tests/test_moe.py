import mlx.core as mx
from mlx_ops.moe import MoEGate, MoELayer


class TestImports:
    def test_import_moe_gate(self):
        assert MoEGate is not None

    def test_import_moe_layer(self):
        assert MoELayer is not None

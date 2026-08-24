"""Tests for modulation module."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest

from mlx_arsenal.modulation import (
    AdaLNModulation,
    ScaleShiftTable,
    gated_residual,
    modulate,
)


class TestModulateFn:
    def test_identity_when_zero(self):
        x = mx.random.normal((2, 4, 8))
        shift = mx.zeros((2, 4, 8))
        scale = mx.zeros((2, 4, 8))
        out = modulate(x, shift, scale)
        assert mx.allclose(out, x).item()

    def test_apply_scale_only(self):
        x = mx.ones((1, 1, 4))
        shift = mx.zeros((1, 1, 4))
        scale = mx.ones((1, 1, 4)) * 0.5  # (1+0.5) = 1.5
        out = modulate(x, shift, scale)
        assert mx.allclose(out, mx.ones((1, 1, 4)) * 1.5).item()

    def test_broadcast_over_sequence(self):
        x = mx.random.normal((2, 5, 8))
        shift = mx.random.normal((2, 1, 8))
        scale = mx.random.normal((2, 1, 8))
        out = modulate(x, shift, scale)
        assert out.shape == (2, 5, 8)
        # Verify broadcast applied identically along S
        manual = x * (1 + scale) + shift
        assert mx.allclose(out, manual).item()


class TestGatedResidual:
    def test_zero_gate_returns_residual(self):
        x = mx.random.normal((2, 4, 8))
        h = mx.random.normal((2, 4, 8))
        gate = mx.zeros((2, 1, 8))
        out = gated_residual(x, gate, h)
        assert mx.allclose(out, x).item()

    def test_one_gate_adds_branch(self):
        x = mx.zeros((1, 4, 8))
        h = mx.ones((1, 4, 8))
        gate = mx.ones((1, 1, 8))
        out = gated_residual(x, gate, h)
        assert mx.allclose(out, h).item()


class TestAdaLNModulation:
    def test_output_shape_default(self):
        mod = AdaLNModulation(dim=16, num_chunks=6)
        c = mx.random.normal((2, 16))
        out = mod(c)
        assert out.shape == (2, 96)  # 6 * 16

    def test_split_into_chunks(self):
        mod = AdaLNModulation(dim=8, num_chunks=6)
        c = mx.random.normal((4, 8))
        out = mod(c)
        chunks = mx.split(out, 6, axis=-1)
        assert len(chunks) == 6
        assert all(chunk.shape == (4, 8) for chunk in chunks)

    @pytest.mark.parametrize("n", [1, 2, 4, 6, 9])
    def test_arbitrary_num_chunks(self, n):
        mod = AdaLNModulation(dim=4, num_chunks=n)
        c = mx.random.normal((1, 4))
        out = mod(c)
        assert out.shape == (1, n * 4)

    def test_per_token_conditioning(self):
        """Conditioning may be (B, S, dim) for per-token modulation."""
        mod = AdaLNModulation(dim=8, num_chunks=2)
        c = mx.random.normal((2, 5, 8))
        out = mod(c)
        assert out.shape == (2, 5, 16)

    def test_use_silu_false_skips_activation(self):
        # With use_silu=False, output equals linear(c) directly.
        mod = AdaLNModulation(dim=4, num_chunks=2, use_silu=False)
        c = mx.random.normal((1, 4))
        out = mod(c)
        expected = mod.linear(c)
        assert mx.allclose(out, expected).item()

    def test_weight_keys(self):
        mod = AdaLNModulation(dim=8, num_chunks=6)
        keys = {name for name, _ in mlx.utils.tree_flatten(mod.parameters())}
        assert "linear.weight" in keys
        assert "linear.bias" in keys

    def test_no_bias(self):
        mod = AdaLNModulation(dim=8, num_chunks=2, bias=False)
        keys = {name for name, _ in mlx.utils.tree_flatten(mod.parameters())}
        assert "linear.weight" in keys
        assert "linear.bias" not in keys


class TestScaleShiftTable:
    def test_init_zeros(self):
        tbl = ScaleShiftTable(dim=4, num_params=2)
        assert mx.allclose(tbl.table, mx.zeros((2, 4))).item()

    def test_zero_table_returns_embedded(self):
        """With zero-initialized table, each chunk equals embedded[:, None, :]."""
        tbl = ScaleShiftTable(dim=4, num_params=2)
        embedded = mx.random.normal((3, 4))
        scale, shift = tbl(embedded)
        # Both should be embedded broadcast to (B, 1, dim)
        assert scale.shape == (3, 1, 4)
        assert shift.shape == (3, 1, 4)
        assert mx.allclose(scale, embedded[:, None, :]).item()
        assert mx.allclose(shift, embedded[:, None, :]).item()

    def test_nonzero_table(self):
        tbl = ScaleShiftTable(dim=4, num_params=2)
        # Directly set the table to known values for testing.
        tbl.table = mx.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        embedded = mx.zeros((1, 4))
        scale, shift = tbl(embedded)
        assert mx.allclose(scale.squeeze(1), mx.array([[1.0, 2.0, 3.0, 4.0]])).item()
        assert mx.allclose(shift.squeeze(1), mx.array([[5.0, 6.0, 7.0, 8.0]])).item()

    def test_num_params_3(self):
        tbl = ScaleShiftTable(dim=8, num_params=3)
        embedded = mx.random.normal((2, 8))
        chunks = tbl(embedded)
        assert len(chunks) == 3
        for c in chunks:
            assert c.shape == (2, 1, 8)


class TestValidation:
    def test_adaln_num_chunks_zero_raises(self):
        with pytest.raises(ValueError, match="num_chunks"):
            AdaLNModulation(dim=4, num_chunks=0)

    def test_adaln_num_chunks_negative_raises(self):
        with pytest.raises(ValueError, match="num_chunks"):
            AdaLNModulation(dim=4, num_chunks=-1)

    def test_scale_shift_table_wrong_row_count_raises(self):
        # A (4, dim) table with num_params=2 splits "evenly" into two
        # (B, 2, dim) chunks — silent misbehavior without validation.
        tbl = ScaleShiftTable(dim=4, num_params=2)
        tbl.table = mx.zeros((4, 4))
        with pytest.raises(ValueError, match="table"):
            tbl(mx.zeros((1, 4)))

    def test_scale_shift_table_non_dividing_table_raises(self):
        tbl = ScaleShiftTable(dim=4, num_params=2)
        tbl.table = mx.zeros((3, 4))
        with pytest.raises(ValueError, match="table"):
            tbl(mx.zeros((1, 4)))


class TestIntegration:
    """End-to-end test of a small DiT-style block using all primitives."""

    def test_dit_block_forward(self):
        dim = 16
        modulation = AdaLNModulation(dim, num_chunks=6)
        norm_msa = nn.RMSNorm(dim)
        norm_mlp = nn.RMSNorm(dim)
        # stand-ins for attention + mlp
        attn = nn.Linear(dim, dim)
        mlp = nn.Linear(dim, dim)

        x = mx.random.normal((2, 8, dim))
        t_emb = mx.random.normal((2, dim))

        params = modulation(t_emb)  # (2, 6*dim)
        chunks = mx.split(params, 6, axis=-1)
        # Caller adds the broadcast dim for sequence axis.
        broadcast = tuple(c[:, None, :] for c in chunks)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = broadcast

        h = modulate(norm_msa(x), shift_msa, scale_msa)
        x = gated_residual(x, gate_msa, attn(h))
        h = modulate(norm_mlp(x), shift_mlp, scale_mlp)
        x = gated_residual(x, gate_mlp, mlp(h))

        assert x.shape == (2, 8, 16)

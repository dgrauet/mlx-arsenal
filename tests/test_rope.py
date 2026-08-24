"""Tests for rope module."""

import math

import mlx.core as mx
import pytest

from mlx_arsenal._typing import item_float
from mlx_arsenal.rope import (
    apply_rotary_emb,
    meshgrid_nd,
    rope_frequencies_1d,
    rope_frequencies_nd,
    rotate_half,
)


class TestRopeFrequencies1D:
    def test_shape(self):
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.arange(5).astype(mx.float32))
        assert cos.shape == (5, 4)
        assert sin.shape == (5, 4)

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            rope_frequencies_1d(dim=7, positions=mx.arange(3).astype(mx.float32))

    def test_position_zero_yields_identity(self):
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.zeros((3,)))
        assert mx.allclose(cos, mx.ones((3, 4))).item()
        assert mx.allclose(sin, mx.zeros((3, 4)), atol=1e-6).item()

    def test_frequency_falls_off_with_dim(self):
        """Highest-index pair rotates slowest (lowest freq) at theta=10000."""
        cos, sin = rope_frequencies_1d(dim=64, positions=mx.array([1.0]))
        # Index 0 should rotate fastest, last index slowest.
        # |sin[0]| > |sin[-1]|.
        assert abs(item_float(sin[0, 0])) > abs(item_float(sin[0, -1]))

    def test_theta_rescale_increases_period(self):
        """Rescaling theta upward should slow the rotation."""
        pos = mx.array([1.0])
        _, sin_default = rope_frequencies_1d(dim=8, positions=pos, theta=10000.0)
        _, sin_rescaled = rope_frequencies_1d(
            dim=8, positions=pos, theta=10000.0, theta_rescale_factor=4.0
        )
        # Rescaled → smaller angles → smaller sin magnitudes.
        assert item_float(mx.abs(sin_rescaled).sum()) < item_float(mx.abs(sin_default).sum())

    def test_interpolation_factor_scales_positions(self):
        """interpolation_factor=2 ≡ doubling all positions."""
        pos = mx.array([1.0])
        cos_a, sin_a = rope_frequencies_1d(dim=8, positions=pos, interpolation_factor=2.0)
        cos_b, sin_b = rope_frequencies_1d(dim=8, positions=mx.array([2.0]))
        assert mx.allclose(cos_a, cos_b, atol=1e-6).item()
        assert mx.allclose(sin_a, sin_b, atol=1e-6).item()


class TestRopeFrequenciesND:
    def test_axis_dims_must_match_grids(self):
        with pytest.raises(ValueError, match="len"):
            rope_frequencies_nd(
                dims_per_axis=[4, 4],
                position_grids=[mx.zeros((3,))],
            )

    def test_concatenates_axis_outputs(self):
        """Output dim is sum of per-axis dims; output rows = sequence length."""
        pos = mx.arange(6).astype(mx.float32)
        cos, sin = rope_frequencies_nd(
            dims_per_axis=[4, 8],
            position_grids=[pos, pos],
        )
        # 4/2 + 8/2 = 2 + 4 = 6
        assert cos.shape == (6, 6)
        assert sin.shape == (6, 6)

    def test_per_axis_theta_rescale_broadcasts(self):
        pos = mx.arange(4).astype(mx.float32)
        # Scalar rescale → broadcast to both axes.
        cos_scalar, _ = rope_frequencies_nd(
            dims_per_axis=[4, 4],
            position_grids=[pos, pos],
            theta_rescale_factor=2.0,
        )
        # Same as explicit per-axis list of length 2.
        cos_list, _ = rope_frequencies_nd(
            dims_per_axis=[4, 4],
            position_grids=[pos, pos],
            theta_rescale_factor=[2.0, 2.0],
        )
        assert mx.allclose(cos_scalar, cos_list).item()

    def test_bad_rescale_length_raises(self):
        pos = mx.zeros((3,))
        with pytest.raises(ValueError, match="theta_rescale_factor"):
            rope_frequencies_nd(
                dims_per_axis=[4, 4],
                position_grids=[pos, pos],
                theta_rescale_factor=[1.0, 2.0, 3.0],
            )


class TestRotateHalf:
    def test_interleaved_known_input(self):
        # [a, b, c, d] → [-b, a, -d, c]
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_half(x, interleaved=True)
        expected = mx.array([[-2.0, 1.0, -4.0, 3.0]])
        assert mx.allclose(out, expected).item()

    def test_half_rotated_known_input(self):
        # [a, b, c, d] (halves) → [-c, -d, a, b]
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_half(x, interleaved=False)
        expected = mx.array([[-3.0, -4.0, 1.0, 2.0]])
        assert mx.allclose(out, expected).item()

    def test_double_rotation_is_negation(self):
        """rotate_half ∘ rotate_half == -x (both conventions)."""
        x = mx.random.normal((2, 8))
        for interleaved in (True, False):
            twice = rotate_half(rotate_half(x, interleaved=interleaved), interleaved=interleaved)
            assert mx.allclose(twice, -x, atol=1e-5).item()


class TestApplyRotaryEmb:
    def test_identity_at_position_zero(self):
        x = mx.random.normal((1, 4, 8))
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.zeros((4,)))
        out = apply_rotary_emb(x, cos, sin)
        assert mx.allclose(out, x, atol=1e-5).item()

    def test_rotation_preserves_norm(self):
        """RoPE is a unitary rotation: ‖x‖ should equal ‖rotated‖."""
        x = mx.random.normal((1, 4, 16))
        positions = mx.arange(4).astype(mx.float32)
        cos, sin = rope_frequencies_1d(dim=16, positions=positions)
        out = apply_rotary_emb(x, cos, sin)
        # Compare per-token norms.
        x_norms = mx.sqrt(mx.sum(x * x, axis=-1))
        out_norms = mx.sqrt(mx.sum(out * out, axis=-1))
        assert mx.allclose(x_norms, out_norms, atol=1e-4).item()

    def test_interleaved_known_angle(self):
        """At pos=1, dim=2, theta=pi: pair rotates by 1 radian.

        Note: pair angle is computed as ``1 / theta^(0/dim) = 1`` for k=0,
        so with positions=[1.0] and dim=2 the angle is exactly 1 radian
        regardless of theta.
        """
        x = mx.array([[[1.0, 0.0]]])  # (B=1, S=1, D=2)
        cos, sin = rope_frequencies_1d(dim=2, positions=mx.array([1.0]))
        out = apply_rotary_emb(x, cos, sin, interleaved=True)
        # Rotation of (1, 0) by 1 rad → (cos(1), sin(1)).
        assert mx.allclose(out, mx.array([[[math.cos(1.0), math.sin(1.0)]]]), atol=1e-5).item()

    def test_half_rotated_variant_runs(self):
        x = mx.random.normal((1, 4, 8))
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.arange(4).astype(mx.float32))
        out = apply_rotary_emb(x, cos, sin, interleaved=False)
        # Shape preserved and not identical (rotation occurred).
        assert out.shape == x.shape
        assert not mx.allclose(out, x).item()

    def test_dtype_round_trip(self):
        """Output dtype matches input dtype even if internal math used fp32."""
        x = mx.random.normal((1, 4, 8)).astype(mx.float16)
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.arange(4).astype(mx.float32))
        out = apply_rotary_emb(x, cos, sin)
        assert out.dtype == mx.float16

    def test_broadcasts_over_extra_axes(self):
        """Works with a (B, S, H, D) shape (multi-head)."""
        x = mx.random.normal((2, 4, 3, 8))
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.arange(4).astype(mx.float32))
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == (2, 4, 3, 8)

    def test_dim_mismatch_raises(self):
        x = mx.random.normal((1, 4, 8))
        cos, sin = rope_frequencies_1d(dim=4, positions=mx.arange(4).astype(mx.float32))
        with pytest.raises(ValueError, match="x.shape"):
            apply_rotary_emb(x, cos, sin)

    def test_seq_axis_mismatch_raises(self):
        x = mx.random.normal((1, 5, 8))  # S=5
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.arange(4).astype(mx.float32))  # S=4
        with pytest.raises(ValueError, match="no axis"):
            apply_rotary_emb(x, cos, sin)


class TestMeshgridND:
    def test_2d_meshgrid(self):
        grids = meshgrid_nd([2, 3])
        # Both grids flat of length 2*3=6
        assert len(grids) == 2
        assert grids[0].shape == (6,)
        # Axis-0 grid: [0,0,0,1,1,1] (broadcasted)
        assert grids[0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        # Axis-1 grid: [0,1,2,0,1,2]
        assert grids[1].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]

    def test_3d_compose_with_rope_nd(self):
        """End-to-end: meshgrid_nd composes with rope_frequencies_nd."""
        grids = meshgrid_nd([2, 2, 2])  # T=H=W=2, S=8
        cos, sin = rope_frequencies_nd(
            dims_per_axis=[4, 4, 4],
            position_grids=grids,
        )
        assert cos.shape == (8, 6)  # S=8, sum(dims)/2 = 12/2 = 6
        assert sin.shape == (8, 6)


class TestRoPECrossAttentionSemantics:
    """Sanity check: attention scores should be position-aware after RoPE."""

    def test_position_aware_dot_product(self):
        # Two identical queries at different positions; one identical key at pos 0.
        dim = 8
        q_same = mx.array([[1.0] * dim])  # base query
        cos, sin = rope_frequencies_1d(dim=dim, positions=mx.array([0.0, 5.0]))

        # Apply RoPE to q at two positions; key stays at pos 0 (identity).
        q_at_0 = apply_rotary_emb(q_same[None, :, :], cos[:1], sin[:1])
        q_at_5 = apply_rotary_emb(q_same[None, :, :], cos[1:], sin[1:])

        dot_0 = item_float((q_at_0 * q_same[None, :, :]).sum())
        dot_5 = item_float((q_at_5 * q_same[None, :, :]).sum())
        # The two dot products should differ — RoPE encoded position.
        assert abs(dot_0 - dot_5) > 1e-3


class TestApplyRotaryEmbSeqAxis:
    def test_ambiguous_seq_axis_raises(self):
        # batch == seq: the axis-length heuristic cannot know which axis is
        # the sequence, and silently picking the first (batch) corrupts the
        # rotation. It must fail loudly instead.
        x = mx.random.normal((4, 4, 2, 8))
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.arange(4, dtype=mx.float32))
        with pytest.raises(ValueError):
            apply_rotary_emb(x, cos, sin)

    def test_explicit_seq_axis_matches_per_sample(self):
        x = mx.random.normal((4, 4, 2, 8))
        cos, sin = rope_frequencies_1d(dim=8, positions=mx.arange(4, dtype=mx.float32))
        out = apply_rotary_emb(x, cos, sin, seq_axis=1)
        for b in range(4):
            per_sample = apply_rotary_emb(x[b], cos, sin)
            assert mx.allclose(out[b], per_sample, atol=1e-6).item()


class TestRopeFrequenciesNDInterpolation:
    def test_interpolation_factor_equals_prescaled_grids(self):
        """interpolation_factor scales positions before frequency computation:
        per-axis factors must equal passing pre-scaled position grids."""
        g0 = mx.arange(6).astype(mx.float32)
        g1 = (mx.arange(6) % 3).astype(mx.float32)
        cos_a, sin_a = rope_frequencies_nd(
            dims_per_axis=[4, 6],
            position_grids=[g0, g1],
            interpolation_factor=[2.0, 3.0],
        )
        cos_b, sin_b = rope_frequencies_nd(
            dims_per_axis=[4, 6],
            position_grids=[g0 * 2.0, g1 * 3.0],
        )
        assert mx.allclose(cos_a, cos_b, atol=1e-6).item()
        assert mx.allclose(sin_a, sin_b, atol=1e-6).item()

    def test_scalar_interpolation_broadcasts(self):
        g = mx.arange(4).astype(mx.float32)
        cos_a, sin_a = rope_frequencies_nd(
            dims_per_axis=[4, 4],
            position_grids=[g, g],
            interpolation_factor=0.5,
        )
        cos_b, sin_b = rope_frequencies_nd(
            dims_per_axis=[4, 4],
            position_grids=[g * 0.5, g * 0.5],
        )
        assert mx.allclose(cos_a, cos_b, atol=1e-6).item()
        assert mx.allclose(sin_a, sin_b, atol=1e-6).item()


class TestMeshgridNDSingleAxis:
    def test_single_axis_is_arange(self):
        grids = meshgrid_nd([5])
        assert len(grids) == 1
        assert grids[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestApplyRotaryEmbHalfRotatedReference:
    def test_hf_rotate_half_hand_computed(self):
        """interleaved=False against a hand-built HF-style reference (S=2, D=4).

        Non-interleaved: cos/sin of shape (S, D/2) are tiled to (S, D) by
        concatenation, and rotate_half maps [x0, x1, x2, x3] to
        [-x2, -x3, x0, x1]. Expected = x*cos + rotate_half(x)*sin.
        """
        theta = 10000.0
        # freqs for D=4: [1/theta^0, 1/theta^(2/4)] = [1.0, 0.01]
        f0, f1 = 1.0, 1.0 / theta**0.5
        positions = [0.0, 1.0]
        x = mx.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])  # (1, 2, 4)

        expected_rows = []
        for s, pos in enumerate(positions):
            a0, a1 = pos * f0, pos * f1
            cos_full = [math.cos(a0), math.cos(a1), math.cos(a0), math.cos(a1)]
            sin_full = [math.sin(a0), math.sin(a1), math.sin(a0), math.sin(a1)]
            row = [item_float(x[0, s, d]) for d in range(4)]
            rot = [-row[2], -row[3], row[0], row[1]]
            expected_rows.append([row[d] * cos_full[d] + rot[d] * sin_full[d] for d in range(4)])
        expected = mx.array([expected_rows])

        cos, sin = rope_frequencies_1d(dim=4, positions=mx.array(positions), theta=theta)
        out = apply_rotary_emb(x, cos, sin, interleaved=False)
        assert mx.allclose(out, expected, atol=1e-5).item()

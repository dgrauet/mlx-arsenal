"""Tests for normalized attention guidance (NAG).

The reference transcribes ``NAGProcessor`` from the authors' repository
(MIT, https://github.com/ChenDarYen/Normalized-Attention-Guidance,
nag/attention_nag.py, lines 103-110).
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal.diffusion import normalized_attention_guidance


def reference(pos, neg, scale, tau, alpha, axis=-1):
    guidance = pos * scale - neg * (scale - 1)
    norm_pos = np.abs(pos).sum(axis=axis, keepdims=True)
    norm_guidance = np.abs(guidance).sum(axis=axis, keepdims=True)
    ratio = norm_guidance / norm_pos
    guidance = guidance * np.minimum(ratio, tau) / ratio
    return guidance * alpha + pos * (1 - alpha)


def _pair(shape=(2, 7, 16), seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=shape)
    neg = pos + spread * rng.normal(size=shape)
    return pos, neg


class TestNormalizedAttentionGuidance:
    @pytest.mark.parametrize("scale", [1.5, 5.0, 11.0])
    @pytest.mark.parametrize("tau", [1.2, 2.5, 100.0])
    @pytest.mark.parametrize("alpha", [0.0, 0.25, 1.0])
    def test_matches_reference(self, scale, tau, alpha):
        pos, neg = _pair()
        out = normalized_attention_guidance(
            mx.array(pos, dtype=mx.float32), mx.array(neg, dtype=mx.float32),
            scale=scale, tau=tau, alpha=alpha,
        )  # fmt: skip
        np.testing.assert_allclose(
            np.array(out), reference(pos, neg, scale, tau, alpha), rtol=1e-4, atol=1e-5
        )

    def test_clipping_is_active(self):
        pos, neg = _pair(spread=3.0)
        guided = reference(pos, neg, 11.0, 1e9, 1.0)
        clipped = reference(pos, neg, 11.0, 2.5, 1.0)
        assert not np.allclose(guided, clipped)

    def test_multi_axis_norm_spans_heads(self):
        pos, neg = _pair(shape=(2, 4, 7, 8), seed=1, spread=2.0)  # (B, H, N, D)
        out = normalized_attention_guidance(
            mx.array(pos, dtype=mx.float32), mx.array(neg, dtype=mx.float32),
            scale=8.0, tau=2.0, alpha=0.5, axis=(1, 3),
        )  # fmt: skip
        flat = lambda a: a.transpose(0, 2, 1, 3).reshape(2, 7, 32)  # noqa: E731
        ref = reference(flat(pos), flat(neg), 8.0, 2.0, 0.5)
        np.testing.assert_allclose(flat(np.array(out)), ref, rtol=1e-4, atol=1e-5)

    def test_scale_one_returns_positive(self):
        pos, neg = _pair()
        out = normalized_attention_guidance(
            mx.array(pos, dtype=mx.float32), mx.array(neg, dtype=mx.float32),
            scale=1.0, tau=2.5, alpha=0.5,
        )  # fmt: skip
        np.testing.assert_allclose(np.array(out), pos, rtol=1e-5, atol=1e-6)

    def test_zero_positive_rows_stay_finite(self):
        pos = np.zeros((1, 3, 4))
        neg = np.ones((1, 3, 4))
        neg[0, 0] = 0.0  # both zero on this token
        out = normalized_attention_guidance(
            mx.array(pos, dtype=mx.float32), mx.array(neg, dtype=mx.float32),
            scale=4.0, tau=2.5, alpha=0.5,
        )  # fmt: skip
        assert bool(mx.all(mx.isfinite(out)))
        assert np.array(out).tolist() == np.zeros((1, 3, 4)).tolist()

    def test_keeps_dtype_and_computes_in_float32(self):
        pos, neg = _pair(shape=(1, 4, 4096), spread=0.5)
        out = normalized_attention_guidance(
            mx.array(pos, dtype=mx.bfloat16), mx.array(neg, dtype=mx.bfloat16),
            scale=11.0, tau=2.5, alpha=0.25,
        )  # fmt: skip
        assert out.dtype == mx.bfloat16
        p32 = np.array(mx.array(pos, dtype=mx.bfloat16).astype(mx.float32))
        n32 = np.array(mx.array(neg, dtype=mx.bfloat16).astype(mx.float32))
        ref = reference(p32, n32, 11.0, 2.5, 0.25)
        np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref, rtol=2e-2, atol=2e-2)

    def test_long_bf16_rows_reduce_in_float32(self):
        # bf16 axis sums saturate past ~2^19 elements per row on MLX 0.32.
        pos = mx.ones((1, 1, 2**20), dtype=mx.bfloat16)
        out = normalized_attention_guidance(
            pos, mx.zeros_like(pos), scale=11.0, tau=2.5, alpha=0.25
        )  # ratio 11 -> clipped to 2.5: 0.25 * 2.5 + 0.75
        assert mx.allclose(out.astype(mx.float32), mx.full(pos.shape, 1.375)).item()

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            normalized_attention_guidance(mx.zeros((1, 2, 4)), mx.zeros((1, 3, 4)), scale=2.0)

    @pytest.mark.parametrize(
        "kw", [{"scale": 0.5}, {"scale": 2.0, "tau": 0.0}, {"scale": 2.0, "alpha": 1.5}]
    )
    def test_rejects_bad_parameters(self, kw):
        with pytest.raises(ValueError):
            normalized_attention_guidance(mx.zeros((1, 2, 4)), mx.zeros((1, 2, 4)), **kw)

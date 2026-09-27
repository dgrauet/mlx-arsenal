"""Tests for mlx_arsenal.diffusion.masked_decode."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any, item_float
from mlx_arsenal.diffusion import token_stats


def _np_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _random_logits(shape: tuple[int, ...], seed: int, scale: float = 3.0) -> np.ndarray:
    return (scale * np.random.default_rng(seed).normal(size=shape)).astype(np.float32)


class TestTokenStats:
    def test_greedy_matches_float64_reference(self):
        logits = _random_logits((2, 5, 11), 0)
        stats = token_stats(array_from_any(logits))
        p = _np_softmax(logits.astype(np.float64))
        x0 = p.argmax(axis=-1)
        assert stats.x0.dtype == mx.int32
        assert np.array(stats.x0).tolist() == x0.tolist()
        np.testing.assert_allclose(
            np.array(stats.prob), np.take_along_axis(p, x0[..., None], -1)[..., 0], atol=1e-6
        )
        entropy = -(p * np.log(p)).sum(axis=-1)
        np.testing.assert_allclose(np.array(stats.entropy), entropy, atol=1e-5)
        assert stats.prob.dtype == stats.entropy.dtype == mx.float32

    def test_suppressed_ids_never_proposed(self):
        logits = _random_logits((1, 4, 9), 1)
        logits[..., 3] = 50.0  # id 3 would win everywhere
        stats = token_stats(array_from_any(logits), suppress_ids=[3])
        kept = np.delete(logits.astype(np.float64), 3, axis=-1)
        p = _np_softmax(kept)
        ref_x0 = p.argmax(axis=-1)
        ref_x0 = ref_x0 + (ref_x0 >= 3)  # map back to full-vocabulary ids
        assert np.array(stats.x0).tolist() == ref_x0.tolist()
        np.testing.assert_allclose(np.array(stats.prob), p.max(axis=-1), atol=1e-6)

    def test_neg_inf_logits_are_nan_free(self):
        logits = _random_logits((2, 3, 6), 2)
        logits[..., :4] = -np.inf  # only 2 live tokens per position
        for dtype in (mx.float32, mx.bfloat16):
            stats = token_stats(array_from_any(logits).astype(dtype))
            assert mx.all(mx.isfinite(stats.prob)).item()
            assert mx.all(mx.isfinite(stats.entropy)).item()
            assert item_float(mx.max(stats.entropy)) <= math.log(2) + 1e-5
            assert item_float(mx.min(mx.array(stats.x0))) >= 4

    def test_bf16_logits(self):
        logits = array_from_any(_random_logits((2, 8, 32), 3)).astype(mx.bfloat16)
        stats = token_stats(logits)
        ref = token_stats(logits.astype(mx.float32))
        assert mx.array_equal(stats.x0, ref.x0).item()
        assert mx.allclose(stats.entropy, ref.entropy, atol=1e-5).item()

    def test_sampling_is_deterministic_for_a_key(self):
        logits = array_from_any(_random_logits((2, 16, 50), 4, scale=1.0))
        a = token_stats(logits, temperature=1.0, key=mx.random.key(7))
        b = token_stats(logits, temperature=1.0, key=mx.random.key(7))
        c = token_stats(logits, temperature=1.0, key=mx.random.key(8))
        assert mx.array_equal(a.x0, b.x0).item()
        assert not mx.array_equal(a.x0, c.x0).item()

    @pytest.mark.parametrize(("temperature", "expected"), [(1.0, 0.8), (0.5, 0.64 / 0.68)])
    def test_sampling_follows_tempered_softmax(self, temperature, expected):
        n = 20000
        logits = mx.broadcast_to(mx.log(mx.array([0.8, 0.2])), (1, n, 2))
        stats = token_stats(logits, temperature=temperature, key=mx.random.key(0))
        freq = item_float(mx.mean((stats.x0 == 0).astype(mx.float32)))
        assert freq == pytest.approx(expected, abs=0.015)

    def test_confidence_uses_unnoised_logits(self):
        logits = _random_logits((1, 64, 5), 5, scale=1.0)
        stats = token_stats(array_from_any(logits), temperature=2.0, key=mx.random.key(1))
        p = _np_softmax(logits.astype(np.float64))
        x0 = np.array(stats.x0)
        np.testing.assert_allclose(
            np.array(stats.prob), np.take_along_axis(p, x0[..., None], -1)[..., 0], atol=1e-6
        )

    def test_greedy_ignores_key(self):
        logits = array_from_any(_random_logits((1, 6, 7), 6))
        a = token_stats(logits)
        b = token_stats(logits, key=mx.random.key(3))
        assert mx.array_equal(a.x0, b.x0).item()

    def test_validation(self):
        logits = array_from_any(_random_logits((1, 3, 5), 7))
        with pytest.raises(ValueError, match="rank 3"):
            token_stats(logits[0])
        with pytest.raises(ValueError, match="temperature"):
            token_stats(logits, temperature=-1.0)
        with pytest.raises(ValueError, match="key"):
            token_stats(logits, temperature=1.0)
        with pytest.raises(ValueError, match="suppress_ids"):
            token_stats(logits, suppress_ids=[5])
        with pytest.raises(ValueError, match="suppress_ids"):
            token_stats(logits, suppress_ids=[-1])
        with pytest.raises(ValueError, match="suppress_ids"):
            token_stats(logits, suppress_ids=range(5))

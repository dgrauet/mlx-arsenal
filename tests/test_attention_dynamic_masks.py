"""Tests for mlx_arsenal.attention.dynamic_masks."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any
from mlx_arsenal.attention import (
    antidiagonal_block_scores,
    centroid_compensated_attention,
    top_p_block_mask,
)


def _qk(B: int, H: int, Nq: int, Nk: int, D: int, seed: int) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    q = array_from_any(rng.normal(size=(B, H, Nq, D)).astype(np.float32))
    k = array_from_any(rng.normal(size=(B, H, Nk, D)).astype(np.float32))
    return q, k


def _top_p_reference(scores: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """XAttention non-causal rule: keep a block while the mass ranked before it < tau * total."""
    B, H, Cq, Ck = scores.shape
    keep = np.zeros(scores.shape, dtype=bool)
    for b in range(B):
        for h in range(H):
            for i in range(Cq):
                row = scores[b, h, i].astype(np.float64)
                order = np.argsort(-row, kind="stable")
                before = np.concatenate([[0.0], np.cumsum(row[order])[:-1]])
                kept = order[before < tau[h] * row.sum()]
                keep[b, h, i, kept if kept.size else order[:1]] = True
    return keep


def _brute_force_scores(q, k, block_size: int, stride: int, scale: float) -> np.ndarray:
    """Anti-diagonal sums of every stride x stride sub-block, softmax, tile sums."""
    qn, kn = np.array(q, dtype=np.float64), np.array(k, dtype=np.float64)
    B, H, Nq, _ = qn.shape
    Nk = kn.shape[2]
    rq, rk = Nq // stride, Nk // stride
    tile = block_size // stride
    out = np.zeros((B, H, Nq // block_size, Nk // block_size))
    for b in range(B):
        for h in range(H):
            logits = np.zeros((rq, rk))
            for i in range(rq):
                for j in range(rk):
                    s = sum(
                        qn[b, h, i * stride + stride - 1 - r] @ kn[b, h, j * stride + r]
                        for r in range(stride)
                    )
                    logits[i, j] = s * scale / stride
            p = np.exp(logits - logits.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            sums = p.reshape(rq // tile, tile, rk // tile, tile).sum(axis=(1, 3))
            out[b, h] = sums / tile
    return out


class TestAntidiagonalBlockScores:
    @pytest.mark.parametrize(("Nq", "Nk", "block_size", "stride"), [(32, 32, 8, 4), (16, 48, 8, 2)])
    def test_matches_brute_force(self, Nq, Nk, block_size, stride):
        q, k = _qk(1, 2, Nq, Nk, 8, 0)
        out = antidiagonal_block_scores(q, k, block_size=block_size, stride=stride)
        ref = _brute_force_scores(q, k, block_size, stride, 1.0 / math.sqrt(8))
        assert out.dtype == mx.float32
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_rows_sum_to_one(self):
        q, k = _qk(2, 3, 64, 64, 16, 1)
        out = antidiagonal_block_scores(q, k, block_size=16, stride=4)
        assert out.shape == (2, 3, 4, 4)
        assert mx.allclose(mx.sum(out, axis=-1), mx.ones((2, 3, 4)), atol=1e-5).item()

    @pytest.mark.parametrize("stride", [1, 8])
    def test_stride_extremes(self, stride):
        q, k = _qk(1, 1, 16, 16, 4, 2)
        out = antidiagonal_block_scores(q, k, block_size=8, stride=stride, scale=0.7)
        ref = _brute_force_scores(q, k, 8, stride, 0.7)
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_bf16_inputs(self):
        q, k = _qk(1, 2, 32, 32, 8, 3)
        out = antidiagonal_block_scores(
            q.astype(mx.bfloat16), k.astype(mx.bfloat16), block_size=8, stride=4
        )
        ref = antidiagonal_block_scores(q, k, block_size=8, stride=4)
        assert out.dtype == mx.float32
        assert mx.all(mx.isfinite(out)).item()
        assert mx.allclose(out, ref, atol=2e-2).item()

    def test_validation(self):
        q, k = _qk(1, 2, 32, 32, 8, 4)
        with pytest.raises(ValueError, match="rank 4"):
            antidiagonal_block_scores(q[0], k, block_size=8, stride=4)
        with pytest.raises(ValueError, match="heads"):
            antidiagonal_block_scores(q, k[:, :1], block_size=8, stride=4)
        with pytest.raises(ValueError, match="head dim"):
            antidiagonal_block_scores(q, k[..., :4], block_size=8, stride=4)
        with pytest.raises(ValueError, match="multiple of block_size"):
            antidiagonal_block_scores(q[:, :, :30], k, block_size=8, stride=4)
        with pytest.raises(ValueError, match="multiple of block_size"):
            antidiagonal_block_scores(q, k[:, :, :30], block_size=8, stride=4)
        with pytest.raises(ValueError, match="multiple of stride"):
            antidiagonal_block_scores(q, k, block_size=8, stride=3)
        with pytest.raises(ValueError, match="block_size"):
            antidiagonal_block_scores(q, k, block_size=0, stride=1)
        with pytest.raises(ValueError, match="stride"):
            antidiagonal_block_scores(q, k, block_size=8, stride=0)


class TestTopPBlockMask:
    def _scores(self, seed: int, shape=(2, 3, 4, 6)) -> mx.array:
        rng = np.random.default_rng(seed)
        raw = rng.random(shape) ** 3  # skewed, like attention mass
        return array_from_any((raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32))

    @pytest.mark.parametrize("tau", [0.3, 0.5, 0.9, 0.99])
    def test_matches_reference(self, tau):
        scores = self._scores(10)
        out = top_p_block_mask(scores, tau)
        ref = _top_p_reference(np.array(scores), np.full(3, tau))
        assert out.dtype == mx.float32
        assert np.array_equal(np.array(out) == 0.0, ref)
        assert np.isneginf(np.array(out)[~ref]).all()

    def test_per_head_thresholds(self):
        scores = self._scores(11)
        tau = mx.array([0.2, 0.6, 0.95])
        out = top_p_block_mask(scores, tau)
        ref = _top_p_reference(np.array(scores), np.array(tau))
        assert np.array_equal(np.array(out) == 0.0, ref)
        kept = np.array(mx.sum(out == 0.0, axis=(0, 2, 3)))
        assert kept[0] < kept[1] < kept[2]

    def test_kept_mass_reaches_threshold(self):
        scores = self._scores(12)
        out = top_p_block_mask(scores, 0.8)
        kept_mass = mx.sum(mx.where(out == 0.0, scores, 0.0), axis=-1)
        assert mx.all(kept_mass >= 0.8 - 1e-6).item()

    def test_zero_row_keeps_one(self):
        scores = mx.zeros((1, 1, 2, 4))
        out = top_p_block_mask(scores, 0.9)
        assert not mx.any(mx.isnan(out)).item()
        assert np.array(mx.sum(out == 0.0, axis=-1)).tolist() == [[[1, 1]]]

    def test_block_zero_is_not_forced(self):
        # XAttention's scatter always re-marks key block 0; we keep only the top-p set.
        scores = mx.array([[[[0.01, 0.9, 0.09]]]])
        assert (np.array(top_p_block_mask(scores, 0.5)) == 0.0).tolist() == [
            [[[False, True, False]]]
        ]

    def test_full_threshold_with_compensation_equals_dense(self):
        q, k = _qk(1, 2, 32, 32, 8, 13)
        v = _qk(1, 2, 32, 32, 8, 14)[0]
        bm = top_p_block_mask(antidiagonal_block_scores(q, k, block_size=8, stride=4), 1.0)
        labels = mx.arange(32) // 8
        out = centroid_compensated_attention(
            q, k, v, q_labels=labels, k_labels=labels, block_mask=bm
        )
        dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(8))
        assert mx.allclose(out, dense, atol=1e-5).item()

    def test_validation(self):
        scores = self._scores(15)
        with pytest.raises(ValueError, match="rank 4"):
            top_p_block_mask(scores[0], 0.9)
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="threshold"):
                top_p_block_mask(scores, bad)
        with pytest.raises(ValueError, match="threshold"):
            top_p_block_mask(scores, mx.array([0.9, 0.9]))
        with pytest.raises(ValueError, match="threshold"):
            top_p_block_mask(scores, mx.array([0.9, 0.0, 0.9]))

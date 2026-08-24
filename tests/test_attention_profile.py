"""Tests for mlx_arsenal.attention.profile."""

import mlx.core as mx
import pytest

from mlx_arsenal._typing import array_from_any, item_float
from mlx_arsenal.attention import (
    Kind,
    classify,
    classify_heads_from_probs,
    classify_heads_from_qk,
)


class TestClassify:
    def test_labels(self):
        scores = mx.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.2], [0.6, 0.7]])
        labels = classify(scores)
        # Last row: both over threshold → SPATIAL (tie-break).
        assert labels == [Kind.SPATIAL, Kind.TEMPORAL, Kind.OTHER, Kind.SPATIAL]

    def test_tie_break_spatial_wins(self):
        scores = mx.array([[0.8, 0.8]])
        assert classify(scores) == [Kind.SPATIAL]

    def test_thresholds_respected(self):
        scores = mx.array([[0.5, 0.0], [0.0, 0.5]])
        assert classify(scores) == [Kind.SPATIAL, Kind.TEMPORAL]
        assert classify(scores, spatial_threshold=0.51) == [Kind.OTHER, Kind.TEMPORAL]

    def test_validation_shape(self):
        with pytest.raises(ValueError):
            classify(mx.array([0.5, 0.5]))  # 1D
        with pytest.raises(ValueError):
            classify(mx.array([[0.5, 0.5, 0.5]]))  # second dim != 2

    def test_validation_thresholds(self):
        scores = mx.array([[0.5, 0.5]])
        with pytest.raises(ValueError):
            classify(scores, spatial_threshold=-0.1)
        with pytest.raises(ValueError):
            classify(scores, temporal_threshold=1.5)


class TestClassifyHeadsFromProbs:
    def _make_same_frame_probs(self, T: int, H: int, W: int, nH: int) -> mx.array:
        # For each query in frame t, place uniform mass on the H*W same-frame keys.
        # Built via numpy to avoid relying on MLX item-assignment semantics.
        import numpy as np

        S = T * H * W
        mask = np.zeros((S, S), dtype=np.float32)
        for t in range(T):
            block_start = t * H * W
            block_end = (t + 1) * H * W
            mask[block_start:block_end, block_start:block_end] = 1.0 / (H * W)
        probs = array_from_any(mask).reshape(1, 1, S, S)
        return mx.broadcast_to(probs, (1, nH, S, S))

    def _make_same_pos_probs(self, T: int, H: int, W: int, nH: int) -> mx.array:
        import numpy as np

        S = T * H * W
        mask = np.zeros((S, S), dtype=np.float32)
        for q in range(S):
            qh = (q % (H * W)) // W
            qw = q % W
            for tk in range(T):
                k = tk * H * W + qh * W + qw
                mask[q, k] = 1.0 / T
        probs = array_from_any(mask).reshape(1, 1, S, S)
        return mx.broadcast_to(probs, (1, nH, S, S))

    def test_shape(self):
        T, H, W = 2, 2, 2
        S = T * H * W
        probs = mx.ones((2, 4, S, S)) / S
        scores = classify_heads_from_probs(probs, T, H, W)
        assert scores.shape == (4, 2)

    def test_pure_spatial(self):
        T, H, W = 2, 2, 2
        probs = self._make_same_frame_probs(T, H, W, nH=3)
        scores = classify_heads_from_probs(probs, T, H, W)
        for h in range(3):
            assert scores[h, 0].item() == pytest.approx(1.0, abs=1e-5)
            assert scores[h, 1].item() == pytest.approx(1.0 / (H * W), abs=1e-5)

    def test_pure_temporal(self):
        T, H, W = 2, 2, 2
        probs = self._make_same_pos_probs(T, H, W, nH=3)
        scores = classify_heads_from_probs(probs, T, H, W)
        for h in range(3):
            assert scores[h, 1].item() == pytest.approx(1.0, abs=1e-5)
            assert scores[h, 0].item() == pytest.approx(1.0 / T, abs=1e-5)

    def test_uniform_baseline(self):
        T, H, W = 4, 3, 3
        S = T * H * W
        probs = mx.ones((1, 2, S, S)) / S
        scores = classify_heads_from_probs(probs, T, H, W)
        for h in range(2):
            assert scores[h, 0].item() == pytest.approx(1.0 / T, abs=1e-5)
            assert scores[h, 1].item() == pytest.approx(1.0 / (H * W), abs=1e-5)

    def test_validation(self):
        T, H, W = 2, 2, 2
        S = T * H * W
        good = mx.ones((1, 1, S, S)) / S
        with pytest.raises(ValueError):
            classify_heads_from_probs(good, T=0, H=H, W=W)
        with pytest.raises(ValueError):
            classify_heads_from_probs(mx.ones((S, S)), T, H, W)
        with pytest.raises(ValueError):
            classify_heads_from_probs(mx.ones((1, 1, S, S + 1)), T, H, W)


class TestClassifierAgreement:
    def test_qk_matches_probs_on_full_sample(self):
        # With n_samples=S every query is sampled, so the sampled estimator
        # averages the exact same per-query masses as the full-probs path —
        # the two classifiers must agree numerically and in labels.
        T, H, W, D = 2, 2, 2, 4
        S = T * H * W
        q = mx.random.normal((1, 3, S, D), key=mx.random.key(0))
        k = mx.random.normal((1, 3, S, D), key=mx.random.key(1))
        scale = 1.0 / mx.sqrt(mx.array(D, dtype=q.dtype))
        probs = mx.softmax(mx.matmul(q, mx.swapaxes(k, 2, 3)) * scale, axis=-1)
        from_probs = classify_heads_from_probs(probs, T, H, W)
        from_qk = classify_heads_from_qk(q, k, T, H, W, n_samples=S)
        assert mx.allclose(from_probs, from_qk, atol=1e-5).item()
        assert classify(from_probs) == classify(from_qk)

    def test_agreement_on_unambiguous_spatial_heads(self):
        # Frame-aligned one-hot Q/K concentrate mass on same-frame keys —
        # far from the 0.5 threshold, so both classifiers must say SPATIAL.
        T, H, W = 2, 2, 2
        S = T * H * W
        D = T
        ids = mx.repeat(mx.arange(T), H * W)
        onehot = mx.take(mx.eye(D), ids, axis=0)
        q = mx.broadcast_to((onehot * 20.0).reshape(1, 1, S, D), (1, 2, S, D))
        k = mx.broadcast_to(onehot.reshape(1, 1, S, D), (1, 2, S, D))
        scale = 1.0 / mx.sqrt(mx.array(D, dtype=q.dtype))
        probs = mx.softmax(mx.matmul(q, mx.swapaxes(k, 2, 3)) * scale, axis=-1)
        labels_probs = classify(classify_heads_from_probs(probs, T, H, W))
        labels_qk = classify(classify_heads_from_qk(q, k, T, H, W, n_samples=S))
        assert labels_probs == labels_qk == [Kind.SPATIAL, Kind.SPATIAL]


class TestClassifyHeadsFromQK:
    def test_shape(self):
        T, H, W = 2, 2, 2
        S = T * H * W
        D = 4
        q = mx.random.normal((2, 3, S, D), key=mx.random.key(0))
        k = mx.random.normal((2, 3, S, D), key=mx.random.key(1))
        scores = classify_heads_from_qk(q, k, T, H, W, n_samples=4)
        assert scores.shape == (3, 2)

    def test_deterministic_with_key(self):
        T, H, W = 2, 2, 2
        S = T * H * W
        D = 4
        q = mx.random.normal((1, 2, S, D), key=mx.random.key(42))
        k = mx.random.normal((1, 2, S, D), key=mx.random.key(43))
        key = mx.random.key(7)
        a = classify_heads_from_qk(q, k, T, H, W, n_samples=4, key=key)
        b = classify_heads_from_qk(q, k, T, H, W, n_samples=4, key=key)
        assert mx.all(mx.equal(a, b)).item()

    def test_recovers_spatial_when_qk_aligned(self):
        # Construct Q, K so softmax concentrates mass on same-frame keys.
        T, H, W = 2, 2, 2
        S = T * H * W
        D = T  # one-hot dim
        ids = mx.repeat(mx.arange(T), H * W)  # (S,) frame id per token
        onehot = mx.take(mx.eye(D), ids, axis=0)  # (S, D)
        q = (onehot * 20.0).reshape(1, 1, S, D)
        k = onehot.reshape(1, 1, S, D)
        q = mx.broadcast_to(q, (1, 2, S, D))
        k = mx.broadcast_to(k, (1, 2, S, D))
        scores = classify_heads_from_qk(q, k, T, H, W, n_samples=S)
        for h in range(2):
            assert item_float(scores[h, 0]) > 0.8

    def test_validation(self):
        T, H, W = 2, 2, 2
        S = T * H * W
        D = 4
        q = mx.random.normal((1, 2, S, D), key=mx.random.key(0))
        k = mx.random.normal((1, 2, S, D), key=mx.random.key(1))
        with pytest.raises(ValueError):
            classify_heads_from_qk(q, k, T, H, W, n_samples=0)
        with pytest.raises(ValueError):
            classify_heads_from_qk(q, k, T, H, W, n_samples=S + 1)
        k_bad = mx.random.normal((1, 2, S, D + 1), key=mx.random.key(2))
        with pytest.raises(ValueError):
            classify_heads_from_qk(q, k_bad, T, H, W, n_samples=4)
        q_bad = q.reshape(2, S, D)
        with pytest.raises(ValueError):
            classify_heads_from_qk(q_bad, q_bad, T, H, W, n_samples=4)

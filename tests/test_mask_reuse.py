"""Tests for mlx_arsenal.diffusion.mask_reuse."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any
from mlx_arsenal.diffusion import pooled_qk, qk_drift


def _qk(seed: int, B: int = 2, H: int = 3, N: int = 16, D: int = 8) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    q = array_from_any(rng.normal(size=(B, H, N, D)).astype(np.float32))
    k = array_from_any(rng.normal(size=(B, H, N, D)).astype(np.float32))
    return q, k


class TestPooledDrift:
    def test_pooled_is_token_mean(self):
        q, k = _qk(0)
        qbar, kbar = pooled_qk(q, k)
        assert qbar.shape == kbar.shape == (2, 3, 8)
        assert qbar.dtype == mx.float32
        np.testing.assert_allclose(np.array(qbar), np.array(q).mean(axis=2), atol=1e-6)
        np.testing.assert_allclose(np.array(kbar), np.array(k).mean(axis=2), atol=1e-6)

    def test_drift_is_l1_of_pooled_difference(self):
        qa, ka = pooled_qk(*_qk(1))
        qb, kb = pooled_qk(*_qk(2))
        ref = np.abs(np.array(qa) - np.array(qb)).sum(-1) + np.abs(np.array(ka) - np.array(kb)).sum(
            -1
        )
        out = qk_drift(qa, ka, qb, kb)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(np.array(out), ref, atol=1e-5)

    def test_relative_drift_is_scale_free(self):
        qa, ka = pooled_qk(*_qk(3))
        qb, kb = pooled_qk(*_qk(4))
        base = qk_drift(qa, ka, qb, kb, relative=True)
        scaled = qk_drift(10 * qa, 10 * ka, 10 * qb, 10 * kb, relative=True)
        assert mx.allclose(base, scaled, atol=1e-5).item()
        assert not mx.allclose(
            qk_drift(qa, ka, qb, kb), qk_drift(10 * qa, 10 * ka, 10 * qb, 10 * kb)
        ).item()

    def test_bf16_inputs(self):
        q, k = _qk(5)
        qbar, _ = pooled_qk(q.astype(mx.bfloat16), k.astype(mx.bfloat16))
        assert qbar.dtype == mx.float32
        assert mx.all(mx.isfinite(qbar)).item()

    def test_validation(self):
        q, k = _qk(6)
        with pytest.raises(ValueError, match="rank 4"):
            pooled_qk(q[0], k)
        with pytest.raises(ValueError, match="shape"):
            pooled_qk(q, k[:, :2])
        qa, ka = pooled_qk(q, k)
        with pytest.raises(ValueError, match="shape"):
            qk_drift(qa, ka, qa[:, :2], ka)

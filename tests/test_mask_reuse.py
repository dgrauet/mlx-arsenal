"""Tests for mlx_arsenal.diffusion.mask_reuse."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any, item_float
from mlx_arsenal.diffusion import HeadMaskCache, pooled_qk, qk_drift


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


def _masks(value: float, B: int = 2, H: int = 3) -> mx.array:
    return mx.full((B, H, 4, 4), value)


class TestHeadMaskCache:
    D = 8

    def test_first_call_refreshes_every_head(self):
        cache = HeadMaskCache(delta=1.0)
        q, k = _qk(0)
        refresh = cache.should_refresh(q, k)
        assert refresh.dtype == mx.bool_
        assert mx.all(refresh).item()
        out = cache.update(_masks(1.0), refresh)
        assert mx.array_equal(out, _masks(1.0)).item()
        assert mx.array_equal(cache.mask, out).item()

    def test_unchanged_input_reuses_mask(self):
        cache = HeadMaskCache(delta=1e-3)
        q, k = _qk(1)
        cache.update(_masks(1.0), cache.should_refresh(q, k))
        refresh = cache.should_refresh(q, k)
        assert not mx.any(refresh).item()
        out = cache.update(_masks(2.0), refresh)  # new masks ignored for reused heads
        assert mx.array_equal(out, _masks(1.0)).item()

    def test_only_drifting_heads_refresh(self):
        cache = HeadMaskCache(delta=1.0)
        q, k = _qk(2)
        cache.update(_masks(1.0), cache.should_refresh(q, k))
        q2 = mx.concatenate([q[:, :1], q[:, 1:2] + 5.0, q[:, 2:]], axis=1)
        refresh = cache.should_refresh(q2, k)
        assert refresh.tolist() == [[False, True, False], [False, True, False]]
        out = cache.update(_masks(2.0), refresh)
        assert np.array(out[:, :, 0, 0]).tolist() == [[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]]

    def test_drift_accumulates_against_the_anchor(self):
        # each step shifts q by 0.1 on every feature: drift vs anchor = 0.8 * steps
        cache = HeadMaskCache(delta=2.0)
        q, k = _qk(3)
        cache.update(_masks(0.0), cache.should_refresh(q, k))
        decisions = []
        for step in range(1, 6):
            refresh = cache.should_refresh(q + 0.1 * step, k)
            decisions.append(bool(mx.all(refresh).item()))
            cache.update(_masks(float(step)), refresh)
        # drift 0.8, 1.6 -> reuse; 2.4 -> refresh (anchor moves to step 3); 0.8, 1.6 -> reuse
        assert decisions == [False, False, True, False, False]
        assert item_float(cache.mask[0, 0, 0, 0]) == 3.0

    def test_relative_threshold(self):
        cache = HeadMaskCache(delta=0.5, relative=True)
        q, k = _qk(4)
        cache.update(_masks(0.0), cache.should_refresh(100 * q, 100 * k))
        assert not mx.any(cache.should_refresh(100 * q + 1.0, 100 * k)).item()

    def test_layer_gate_per_batch_row(self):
        cache = HeadMaskCache(delta=1.0, layer_gate=(0.4, 0.8))
        q, k = _qk(5, B=3, H=5)
        cache.update(_masks(0.0, B=3, H=5), cache.should_refresh(q, k))
        bump = mx.zeros((3, 5, 1, 1))
        bump[0, :1] = 5.0  # 1/5 = 0.2 < 0.4  -> whole row reuses
        bump[1, :3] = 5.0  # 3/5 = 0.6        -> per-head decisions stand
        bump[2, :5] = 5.0  # 5/5 = 1.0 > 0.8  -> whole row refreshes
        refresh = cache.should_refresh(q + bump, k)
        assert refresh.tolist() == [
            [False] * 5,
            [True, True, True, False, False],
            [True] * 5,
        ]

    def test_layer_gate_high_forces_refresh(self):
        cache = HeadMaskCache(delta=1.0, layer_gate=(0.0, 0.5))
        q, k = _qk(6, B=1, H=4)
        cache.update(_masks(0.0, B=1, H=4), cache.should_refresh(q, k))
        bump = mx.zeros((1, 4, 1, 1))
        bump[0, :3] = 5.0  # 3/4 = 0.75 > 0.5 -> all four refresh
        assert mx.all(cache.should_refresh(q + bump, k)).item()

    def test_call_order_is_enforced(self):
        cache = HeadMaskCache(delta=1.0)
        q, k = _qk(7)
        with pytest.raises(RuntimeError, match="should_refresh"):
            cache.update(_masks(1.0), mx.ones((2, 3), dtype=mx.bool_))
        refresh = cache.should_refresh(q, k)
        with pytest.raises(RuntimeError, match="update"):
            cache.should_refresh(q, k)
        with pytest.raises(RuntimeError, match="every head"):
            cache.update(_masks(1.0), mx.zeros_like(refresh))

    def test_mask_before_first_update(self):
        with pytest.raises(RuntimeError, match="mask"):
            _ = HeadMaskCache(delta=1.0).mask

    def test_shape_changes_are_rejected(self):
        cache = HeadMaskCache(delta=1.0)
        q, k = _qk(8)
        cache.update(_masks(1.0), cache.should_refresh(q, k))
        with pytest.raises(ValueError, match="shape"):
            cache.should_refresh(*_qk(8, H=2))
        refresh = cache.should_refresh(q, k)
        with pytest.raises(ValueError, match="shape"):
            cache.update(mx.zeros((2, 3, 5, 5)), refresh)

    def test_update_validates_refresh(self):
        cache = HeadMaskCache(delta=1.0)
        q, k = _qk(9)
        cache.should_refresh(q, k)
        with pytest.raises(ValueError, match="refresh"):
            cache.update(_masks(1.0), mx.ones((2, 2), dtype=mx.bool_))

    def test_reset(self):
        cache = HeadMaskCache(delta=1.0)
        q, k = _qk(10)
        cache.update(_masks(1.0), cache.should_refresh(q, k))
        cache.reset()
        assert mx.all(cache.should_refresh(q, k)).item()

    def test_validation(self):
        with pytest.raises(ValueError, match="delta"):
            HeadMaskCache(delta=-1.0)
        for gate in ((0.5, 0.4), (-0.1, 0.5), (0.2, 1.5)):
            with pytest.raises(ValueError, match="layer_gate"):
                HeadMaskCache(delta=1.0, layer_gate=gate)

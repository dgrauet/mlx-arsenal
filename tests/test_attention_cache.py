"""Tests for mlx_arsenal.diffusion.attention_cache."""

import mlx.core as mx
import pytest

from mlx_arsenal.diffusion import PerHeadAttentionCache, PerLayerAttentionCache


class TestPerLayerAttentionCache:
    def test_boundary_steps_force_compute(self):
        cache = PerLayerAttentionCache(num_steps=4, rel_l1_thresh=0.1)
        x = mx.ones((2, 4, 8, 16))
        assert cache.should_compute(0, x) is True
        cache.cache_output(x)
        assert cache.should_compute(1, x) is False
        assert cache.should_compute(2, x) is False
        assert cache.should_compute(3, x) is True

    def test_skip_when_input_static(self):
        cache = PerLayerAttentionCache(num_steps=5, rel_l1_thresh=0.1)
        x = mx.ones((1, 2, 4, 4))
        assert cache.should_compute(0, x) is True
        cache.cache_output(x)
        assert cache.should_compute(1, x) is False
        assert cache.should_compute(2, x) is False
        assert cache.should_compute(3, x) is False

    def test_recompute_when_input_changes(self):
        cache = PerLayerAttentionCache(num_steps=5, rel_l1_thresh=0.05)
        x0 = mx.ones((1, 2, 4, 4))
        x1 = mx.ones((1, 2, 4, 4)) * 2.0
        cache.should_compute(0, x0)
        cache.cache_output(x0)
        assert cache.should_compute(1, x1) is True

    def test_previous_output_raises_before_cache(self):
        cache = PerLayerAttentionCache(num_steps=3, rel_l1_thresh=0.1)
        with pytest.raises(RuntimeError):
            _ = cache.previous_output

    def test_reset_clears_state(self):
        cache = PerLayerAttentionCache(num_steps=4, rel_l1_thresh=0.1)
        x = mx.ones((1, 2, 4, 4))
        cache.should_compute(0, x)
        cache.cache_output(x)
        cache.reset()
        with pytest.raises(RuntimeError):
            _ = cache.previous_output

    def test_validation(self):
        with pytest.raises(ValueError):
            PerLayerAttentionCache(num_steps=1, rel_l1_thresh=0.1)
        with pytest.raises(ValueError):
            PerLayerAttentionCache(num_steps=4, rel_l1_thresh=0.0)
        with pytest.raises(ValueError):
            PerLayerAttentionCache(num_steps=4, rel_l1_thresh=-0.5)

    def test_should_compute_from_summary(self):
        cache = PerLayerAttentionCache(num_steps=5, rel_l1_thresh=0.1)
        assert cache.should_compute_from_summary(0, 99.0) is True
        cache.should_compute(0, mx.ones((1, 2, 4, 4)))
        cache.cache_output(mx.ones((1, 2, 4, 4)))
        assert cache.should_compute_from_summary(1, 0.01) is False
        assert cache.should_compute_from_summary(2, 0.2) is True


class TestPerHeadAttentionCache:
    def test_boundary_steps_all_heads_compute(self):
        cache = PerHeadAttentionCache(num_heads=3, num_steps=4, rel_l1_thresh=0.1)
        x = mx.ones((2, 3, 8, 16))
        m = cache.should_compute(0, x)
        assert m.shape == (3,)
        assert mx.all(m).item()
        cache.cache_output(x)
        m_last = cache.should_compute(3, x)
        assert mx.all(m_last).item()

    def test_per_head_decisions_differ(self):
        # Head 0 doubles between steps; head 1 stays the same.
        T, S, D = 1, 4, 4
        x0 = mx.ones((T, 2, S, D))
        x1_h0 = mx.ones((T, 1, S, D)) * 2.0
        x1_h1 = mx.ones((T, 1, S, D))
        x1 = mx.concatenate([x1_h0, x1_h1], axis=1)
        cache = PerHeadAttentionCache(num_heads=2, num_steps=5, rel_l1_thresh=0.1)
        cache.should_compute(0, x0)
        cache.cache_output(x0)
        m = cache.should_compute(1, x1)
        assert m.shape == (2,)
        assert bool(m[0].item()) is True
        assert bool(m[1].item()) is False

    def test_should_compute_from_summary(self):
        cache = PerHeadAttentionCache(num_heads=3, num_steps=5, rel_l1_thresh=0.1)
        # Boundary forces all True regardless of summary content.
        m = cache.should_compute_from_summary(0, mx.array([0.0, 0.0, 0.0]))
        assert mx.all(m).item()
        cache.should_compute(0, mx.ones((1, 3, 4, 4)))
        cache.cache_output(mx.ones((1, 3, 4, 4)))
        m = cache.should_compute_from_summary(1, mx.array([0.5, 0.05, 0.5]))
        assert bool(m[0].item()) is True
        assert bool(m[1].item()) is False
        assert bool(m[2].item()) is True

    def test_summary_input_shape_validation(self):
        cache = PerHeadAttentionCache(num_heads=3, num_steps=5, rel_l1_thresh=0.1)
        cache.should_compute(0, mx.ones((1, 3, 4, 4)))
        cache.cache_output(mx.ones((1, 3, 4, 4)))
        with pytest.raises(ValueError):
            cache.should_compute_from_summary(1, mx.array([0.1, 0.2]))
        with pytest.raises(ValueError):
            cache.should_compute_from_summary(1, mx.array([[0.1], [0.2], [0.3]]))

    def test_reset_clears_state(self):
        cache = PerHeadAttentionCache(num_heads=2, num_steps=4, rel_l1_thresh=0.1)
        cache.should_compute(0, mx.ones((1, 2, 4, 4)))
        cache.cache_output(mx.ones((1, 2, 4, 4)))
        cache.reset()
        with pytest.raises(RuntimeError):
            _ = cache.previous_output

    def test_previous_output_raises_before_cache(self):
        cache = PerHeadAttentionCache(num_heads=2, num_steps=4, rel_l1_thresh=0.1)
        with pytest.raises(RuntimeError):
            _ = cache.previous_output

    def test_validation(self):
        with pytest.raises(ValueError):
            PerHeadAttentionCache(num_heads=0, num_steps=4, rel_l1_thresh=0.1)
        with pytest.raises(ValueError):
            PerHeadAttentionCache(num_heads=2, num_steps=1, rel_l1_thresh=0.1)
        with pytest.raises(ValueError):
            PerHeadAttentionCache(num_heads=2, num_steps=4, rel_l1_thresh=0.0)

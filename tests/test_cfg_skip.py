"""Tests for mlx_arsenal.diffusion.cfg_skip."""

from typing import Any

import mlx.core as mx
import pytest

from mlx_arsenal.diffusion import cfg_head_similarity, cfg_skip_mask


class TestCfgHeadSimilarity:
    def test_identical_cond_uncond_cosine_one(self):
        cond = mx.random.normal((2, 4, 8, 16), key=mx.random.key(0))
        uncond = cond
        sim = cfg_head_similarity(cond, uncond, metric="cosine")
        assert sim.shape == (4,)
        for h in range(4):
            assert sim[h].item() == pytest.approx(1.0, abs=1e-5)

    def test_identical_cond_uncond_relative_l1_zero(self):
        cond = mx.random.normal((2, 4, 8, 16), key=mx.random.key(1))
        sim = cfg_head_similarity(cond, cond, metric="relative_l1")
        assert sim.shape == (4,)
        for h in range(4):
            assert sim[h].item() == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_cosine_zero(self):
        # Two heads, each (B=1, S=4, D=2). Make cond and uncond orthogonal.
        cond = mx.array([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])
        uncond = mx.array([[[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]])
        sim = cfg_head_similarity(cond, uncond, metric="cosine")
        assert sim.shape == (1,)
        assert sim[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_validation(self):
        a = mx.zeros((1, 2, 4, 4))
        b = mx.zeros((1, 2, 4, 5))  # shape mismatch
        with pytest.raises(ValueError):
            cfg_head_similarity(a, b, metric="cosine")
        with pytest.raises(ValueError):
            bad_metric: Any = "bogus"
            cfg_head_similarity(a, a, metric=bad_metric)
        c = mx.zeros((2, 4))  # ndim < 4
        with pytest.raises(ValueError):
            cfg_head_similarity(c, c, metric="cosine")


class TestCfgSkipMask:
    def test_cosine_skip_above_threshold(self):
        scores = mx.array([0.99, 0.5, 0.95, 0.10])
        mask = cfg_skip_mask(scores, threshold=0.9, metric="cosine")
        assert mask.dtype == mx.bool_
        assert mask.tolist() == [True, False, True, False]

    def test_relative_l1_skip_below_threshold(self):
        scores = mx.array([0.01, 0.5, 0.05, 1.2])
        mask = cfg_skip_mask(scores, threshold=0.1, metric="relative_l1")
        assert mask.dtype == mx.bool_
        assert mask.tolist() == [True, False, True, False]

    def test_metric_validation(self):
        bad_metric: Any = "bogus"
        with pytest.raises(ValueError):
            cfg_skip_mask(mx.array([0.5]), threshold=0.5, metric=bad_metric)

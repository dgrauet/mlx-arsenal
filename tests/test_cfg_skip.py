"""Tests for mlx_arsenal.diffusion.cfg_skip."""

from typing import Any

import mlx.core as mx
import pytest

from mlx_arsenal.diffusion import (
    CFGSimilarityProfiler,
    cfg_head_similarity,
    cfg_skip_mask,
)


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


class TestCFGSimilarityProfiler:
    def test_record_and_scores_average(self):
        prof = CFGSimilarityProfiler(num_blocks=2, num_heads=3, metric="cosine")
        # Two records on block 0: identical inputs → cosine = 1.
        x = mx.random.normal((1, 3, 4, 4), key=mx.random.key(0))
        prof.record(0, x, x)
        prof.record(0, x, x)
        scores = prof.scores
        assert scores.shape == (2, 3)
        for h in range(3):
            assert scores[0, h].item() == pytest.approx(1.0, abs=1e-5)
        # Block 1 untouched → mean defined as 0.0.
        for h in range(3):
            assert scores[1, h].item() == pytest.approx(0.0, abs=1e-5)

    def test_call_counts_per_block(self):
        prof = CFGSimilarityProfiler(num_blocks=3, num_heads=2, metric="cosine")
        x = mx.random.normal((1, 2, 4, 4), key=mx.random.key(1))
        prof.record(0, x, x)
        prof.record(0, x, x)
        prof.record(2, x, x)
        counts = prof.call_counts
        assert counts.shape == (3,)
        assert counts.tolist() == [2, 0, 1]

    def test_build_skip_schedule(self):
        prof = CFGSimilarityProfiler(num_blocks=2, num_heads=2, metric="cosine")
        # Block 0 head 0: cond == uncond → cosine = 1 → skip.
        # Block 0 head 1: orthogonal → cosine = 0 → no skip.
        cond = mx.array(
            [
                [
                    [[1.0, 0.0], [1.0, 0.0]],  # head 0
                    [[1.0, 0.0], [1.0, 0.0]],  # head 1
                ]
            ]
        )
        uncond = mx.array(
            [
                [
                    [[1.0, 0.0], [1.0, 0.0]],  # head 0 same
                    [[0.0, 1.0], [0.0, 1.0]],  # head 1 orthogonal
                ]
            ]
        )
        prof.record(0, cond, uncond)
        schedule = prof.build_skip_schedule(threshold=0.5)
        assert schedule.shape == (2, 2)
        assert schedule.dtype == mx.bool_
        assert bool(schedule[0, 0].item()) is True
        assert bool(schedule[0, 1].item()) is False
        # Block 1 never recorded → scores 0.0 → not skipped at thresh 0.5.
        assert bool(schedule[1, 0].item()) is False
        assert bool(schedule[1, 1].item()) is False

    def test_reset_clears(self):
        prof = CFGSimilarityProfiler(num_blocks=2, num_heads=2, metric="cosine")
        x = mx.random.normal((1, 2, 4, 4), key=mx.random.key(2))
        prof.record(0, x, x)
        prof.reset()
        assert prof.call_counts.tolist() == [0, 0]
        for b in range(2):
            for h in range(2):
                assert prof.scores[b, h].item() == 0.0

    def test_validation(self):
        with pytest.raises(ValueError):
            CFGSimilarityProfiler(num_blocks=0, num_heads=2, metric="cosine")
        with pytest.raises(ValueError):
            CFGSimilarityProfiler(num_blocks=2, num_heads=0, metric="cosine")
        bad_metric: Any = "bogus"
        with pytest.raises(ValueError):
            CFGSimilarityProfiler(num_blocks=2, num_heads=2, metric=bad_metric)
        prof = CFGSimilarityProfiler(num_blocks=2, num_heads=3, metric="cosine")
        x_ok = mx.zeros((1, 3, 4, 4))
        x_wrong_heads = mx.zeros((1, 2, 4, 4))
        with pytest.raises(ValueError):
            prof.record(-1, x_ok, x_ok)
        with pytest.raises(ValueError):
            prof.record(2, x_ok, x_ok)
        with pytest.raises(ValueError):
            prof.record(0, x_wrong_heads, x_wrong_heads)

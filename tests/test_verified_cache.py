"""Tests for VerifiedFeatureCache (SpeCa-style forecast-then-verify)."""

import mlx.core as mx
import pytest

from mlx_arsenal.diffusion import VerifiedFeatureCache, geometric_threshold


def make_cache(num_steps=10, tau_0=0.1, beta=0.5, order=2, epsilon=1e-8):
    return VerifiedFeatureCache(
        num_steps=num_steps,
        tau_0=tau_0,
        beta=beta,
        order=order,
        epsilon=epsilon,
    )


class TestConstruction:
    def test_invalid_num_steps_raises(self):
        with pytest.raises(ValueError):
            VerifiedFeatureCache(num_steps=1, tau_0=0.1, beta=0.5)

    def test_invalid_tau_0_raises(self):
        with pytest.raises(ValueError):
            VerifiedFeatureCache(num_steps=10, tau_0=0.0, beta=0.5)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            VerifiedFeatureCache(num_steps=10, tau_0=0.1, beta=-0.1)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            VerifiedFeatureCache(num_steps=10, tau_0=0.1, beta=0.5, order=0)

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            VerifiedFeatureCache(num_steps=10, tau_0=0.1, beta=0.5, epsilon=0.0)


class TestGeometricThreshold:
    def test_beta_one_is_constant(self):
        for step in range(5):
            assert geometric_threshold(step, 10, 0.1, 1.0) == pytest.approx(0.1)

    def test_beta_lt_one_grows_with_step(self):
        # τ_t = τ₀ · β^((T-1-t)/(T-1)). β=0.5, t=0 → τ₀·0.5 ; t=T-1 → τ₀.
        early = geometric_threshold(0, 10, 0.1, 0.5)
        late = geometric_threshold(9, 10, 0.1, 0.5)
        assert early == pytest.approx(0.1 * 0.5)
        assert late == pytest.approx(0.1)
        assert early < late

    def test_num_steps_one_returns_tau_0(self):
        assert geometric_threshold(0, 1, 0.1, 0.5) == pytest.approx(0.1)


class TestCanPredict:
    def test_boundary_steps_cannot_predict(self):
        c = make_cache(num_steps=10, order=1)
        # Even with anchors recorded elsewhere, step 0 and last are off-limits.
        c.record(2, mx.ones((4,)))
        c.record(3, mx.ones((4,)))
        assert c.can_predict(0) is False
        assert c.can_predict(9) is False

    def test_insufficient_anchors_cannot_predict(self):
        c = make_cache(order=2)  # needs 3 anchors
        assert c.can_predict(5) is False
        c.record(1, mx.ones((4,)))
        c.record(2, mx.ones((4,)))
        assert c.can_predict(5) is False  # only 2 anchors
        c.record(3, mx.ones((4,)))
        assert c.can_predict(5) is True

    def test_non_monotonic_target_cannot_predict(self):
        c = make_cache(order=1)
        c.record(3, mx.ones((4,)))
        c.record(4, mx.ones((4,)))
        assert c.can_predict(5) is True
        assert c.can_predict(4) is False  # not strictly greater than last anchor


class TestExtrapolate:
    def test_linear_extrapolation_order_1(self):
        # f(s) = 2s + 1. Anchors at s=2 → 5, s=3 → 7. Predict s=4 → 9.
        c = make_cache(order=1)
        c.record(2, mx.array([5.0]))
        c.record(3, mx.array([7.0]))
        pred = c.extrapolate(4)
        assert mx.allclose(pred, mx.array([9.0]), atol=1e-5).item()

    def test_quadratic_extrapolation_order_2(self):
        # f(s) = s². Anchors at s=1,2,3 → 1,4,9. Predict s=4 → 16.
        c = make_cache(order=2)
        c.record(1, mx.array([1.0]))
        c.record(2, mx.array([4.0]))
        c.record(3, mx.array([9.0]))
        pred = c.extrapolate(4)
        assert mx.allclose(pred, mx.array([16.0]), atol=1e-4).item()

    def test_extrapolate_when_not_ready_raises(self):
        c = make_cache(order=2)
        c.record(1, mx.ones((4,)))
        with pytest.raises(RuntimeError):
            c.extrapolate(5)

    def test_extrapolate_preserves_tensor_shape(self):
        c = make_cache(order=1)
        c.record(2, mx.ones((3, 4, 5)))
        c.record(3, mx.ones((3, 4, 5)) * 2)
        pred = c.extrapolate(4)
        assert pred.shape == (3, 4, 5)


class TestAccept:
    def test_exact_match_accepts(self):
        c = make_cache(tau_0=0.01, beta=1.0)
        feat = mx.ones((4,))
        assert c.accept(1, feat, feat) is True

    def test_large_error_rejects(self):
        c = make_cache(tau_0=0.01, beta=1.0)
        predicted = mx.ones((4,))
        actual = mx.ones((4,)) * 10.0
        assert c.accept(1, predicted, actual) is False

    def test_below_threshold_accepts(self):
        # Relative-L2² = ‖0.01‖²/‖1‖² = 0.0001 per element, normalized = 0.0001.
        c = make_cache(tau_0=0.001, beta=1.0)
        predicted = mx.ones((4,)) * 1.01
        actual = mx.ones((4,))
        # diff_sq = 4 * 0.0001 = 0.0004 ; actual_sq = 4 ; error = 0.0001.
        assert c.accept(1, predicted, actual) is True

    def test_threshold_drives_decision_via_step(self):
        # β=0.5: early steps strict, late steps tolerant.
        c = make_cache(num_steps=10, tau_0=0.01, beta=0.5)
        predicted = mx.ones((4,)) * 1.05
        actual = mx.ones((4,))
        # error = 4·0.0025/4 = 0.0025.
        # threshold(1) = 0.01 · 0.5^(8/9) ≈ 0.01·0.537 ≈ 0.00537 → reject.
        # threshold(8) = 0.01 · 0.5^(1/9) ≈ 0.01·0.926 ≈ 0.00926 → reject too.
        # Bump predicted closer to actual to land between thresholds.
        predicted = mx.ones((4,)) * 1.077  # error ≈ 0.0059
        assert c.accept(1, predicted, actual) is False
        assert c.accept(8, predicted, actual) is True


class TestRecord:
    def test_non_monotonic_record_raises(self):
        c = make_cache(order=1)
        c.record(3, mx.ones((4,)))
        with pytest.raises(ValueError):
            c.record(3, mx.ones((4,)))
        with pytest.raises(ValueError):
            c.record(2, mx.ones((4,)))

    def test_fifo_eviction(self):
        c = make_cache(order=1)  # capacity = 2
        c.record(1, mx.array([1.0]))
        c.record(2, mx.array([2.0]))
        c.record(3, mx.array([3.0]))  # evicts step 1
        # Linear extrapolation from anchors (2,2), (3,3) → predict 4 → 4.
        pred = c.extrapolate(4)
        assert mx.allclose(pred, mx.array([4.0]), atol=1e-5).item()


class TestReset:
    def test_reset_drops_anchors(self):
        c = make_cache(order=1)
        c.record(2, mx.ones((4,)))
        c.record(3, mx.ones((4,)))
        assert c.can_predict(4) is True
        c.reset()
        assert c.can_predict(4) is False

    def test_reset_allows_re_record_from_step_zero(self):
        c = make_cache(order=1)
        c.record(5, mx.ones((4,)))
        c.reset()
        # After reset, anchor step monotonicity restarts.
        c.record(1, mx.ones((4,)))


class TestThresholdMethod:
    def test_threshold_matches_free_function(self):
        c = make_cache(num_steps=20, tau_0=0.2, beta=0.7)
        for step in [0, 5, 10, 19]:
            assert c.threshold(step) == pytest.approx(geometric_threshold(step, 20, 0.2, 0.7))


class TestEndToEnd:
    def test_full_loop_with_smooth_feature(self):
        """Feature evolves smoothly → predictions accepted past the warmup."""
        cache = make_cache(num_steps=10, tau_0=0.1, beta=1.0, order=1)

        def truth(step: int) -> mx.array:
            # Linear in step → order-1 extrapolation is exact.
            return mx.array([float(step) * 0.5 + 1.0])

        skipped = 0
        for step in range(10):
            if cache.can_predict(step):
                pred = cache.extrapolate(step)
                actual = truth(step)
                if cache.accept(step, pred, actual):
                    cache.record(step, pred)
                    skipped += 1
                    continue
            cache.record(step, truth(step))

        # Steps 0 and 9 are boundaries (full compute), steps 1 forces a full
        # compute (only 1 anchor available), step 2 also (still need 2 anchors).
        # From step 2 onward, predictions should land exactly on truth and skip.
        assert skipped >= 6

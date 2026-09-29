"""Tests for TeaCache controller (timestep-aware residual caching)."""

import mlx.core as mx
import pytest

from mlx_arsenal.diffusion import TeaCacheController

# Linear rescaling f(x) = 2x for predictable arithmetic in tests.
LINEAR_COEFFS = [2.0, 0.0]


def make_controller(num_steps=4, rel_l1_thresh=0.1, coefficients=LINEAR_COEFFS):
    return TeaCacheController(
        num_steps=num_steps, rel_l1_thresh=rel_l1_thresh, coefficients=coefficients
    )


class TestBoundarySteps:
    def test_first_step_always_computes(self):
        c = make_controller()
        assert c.should_compute(0, mx.ones((4,))) is True

    def test_last_step_always_computes(self):
        c = make_controller(num_steps=5)
        # Tiny deltas mid-run would normally skip, but the last step must compute.
        c.should_compute(0, mx.ones((4,)))
        c.should_compute(1, mx.ones((4,)) * 1.0001)
        assert c.should_compute(4, mx.ones((4,)) * 1.0001) is True


class TestThresholding:
    def test_below_threshold_skips(self):
        # delta = |1.001 - 1| / 1 = 0.001 ; rescaled = 2 * 0.001 = 0.002 < 0.1
        c = make_controller(num_steps=10, rel_l1_thresh=0.1)
        c.should_compute(0, mx.ones((4,)))
        assert c.should_compute(1, mx.ones((4,)) * 1.001) is False

    def test_above_threshold_computes(self):
        # delta = |2 - 1| / 1 = 1 ; rescaled = 2 * 1 = 2 > 0.1
        c = make_controller(num_steps=10, rel_l1_thresh=0.1)
        c.should_compute(0, mx.ones((4,)))
        assert c.should_compute(1, mx.ones((4,)) * 2.0) is True

    def test_accumulation_crosses_threshold(self):
        # Each interior step contributes ≈0.04 to the accumulator (rescaled
        # delta of 2 * 0.02). With thresh 0.1, the cross happens at step 3.
        c = make_controller(num_steps=10, rel_l1_thresh=0.1)
        c.should_compute(0, mx.full((4,), 1.0))
        assert c.should_compute(1, mx.full((4,), 1.02)) is False
        assert c.should_compute(2, mx.full((4,), 1.0404)) is False
        assert c.should_compute(3, mx.full((4,), 1.061208)) is True

    def test_compute_resets_accumulator(self):
        c = make_controller(num_steps=10, rel_l1_thresh=0.1)
        c.should_compute(0, mx.full((4,), 1.0))
        # Large delta forces a compute and resets the accumulator.
        c.should_compute(1, mx.full((4,), 5.0))
        # A subsequent small delta should now skip (acc starts at 0).
        assert c.should_compute(2, mx.full((4,), 5.005)) is False


class TestResidualCache:
    def test_previous_residual_before_caching_raises(self):
        c = make_controller()
        with pytest.raises(RuntimeError):
            _ = c.previous_residual

    def test_cache_residual_stores_value(self):
        c = make_controller()
        residual = mx.array([1.0, 2.0, 3.0])
        c.cache_residual(residual)
        assert mx.allclose(c.previous_residual, residual).item()

    def test_cache_residual_overwrites(self):
        c = make_controller()
        c.cache_residual(mx.array([1.0]))
        c.cache_residual(mx.array([7.0]))
        assert c.previous_residual.item() == pytest.approx(7.0)


class TestReset:
    def test_reset_clears_residual(self):
        c = make_controller()
        c.cache_residual(mx.array([1.0]))
        c.reset()
        with pytest.raises(RuntimeError):
            _ = c.previous_residual

    def test_reset_clears_accumulator(self):
        c = make_controller(num_steps=10, rel_l1_thresh=0.1)
        c.should_compute(0, mx.full((4,), 1.0))
        c.should_compute(1, mx.full((4,), 1.04))  # acc=0.08
        c.reset()
        c.should_compute(0, mx.full((4,), 1.0))
        # delta=0.025 → rescaled=0.05; without reset acc would be 0.13 > 0.1 (compute).
        # After reset, acc=0.05 < 0.1 → must skip.
        assert c.should_compute(1, mx.full((4,), 1.025)) is False


class TestPolyfitRescaling:
    def test_quadratic_coefficients(self):
        # poly1d([3, 0, 0]) = 3x² ; delta=0.5 → rescaled=0.75 < thresh 1.0 → skip.
        c = make_controller(num_steps=10, rel_l1_thresh=1.0, coefficients=[3.0, 0.0, 0.0])
        c.should_compute(0, mx.full((4,), 1.0))
        assert c.should_compute(1, mx.full((4,), 1.5)) is False


class TestEndToEndCacheReuse:
    def test_skip_then_apply_residual(self):
        """Compute first step, cache residual; next step skips and reuses the residual."""
        c = make_controller(num_steps=10, rel_l1_thresh=0.1)
        x_in = mx.full((4,), 1.0)
        assert c.should_compute(0, x_in) is True
        residual = mx.full((4,), 0.5)  # output - input from the just-computed forward
        c.cache_residual(residual)

        x_in_2 = mx.full((4,), 1.001)
        assert c.should_compute(1, x_in_2) is False
        recovered = x_in_2 + c.previous_residual
        assert mx.allclose(recovered, mx.full((4,), 1.501), atol=1e-6).item()


class TestArbitraryPayloadCache:
    def test_cache_residual_accepts_dict_of_tuples(self):
        """LTX-2 caches a per-pass dict; controller must accept arbitrary payloads."""
        c = make_controller()
        payload = {
            "cond": (mx.array([1.0]), mx.array([2.0])),
            "uncond": (mx.array([3.0]), mx.array([4.0])),
        }
        c.cache_residual(payload)
        retrieved = c.previous_residual
        assert retrieved is payload  # exact identity, not a copy
        assert mx.allclose(retrieved["cond"][0], mx.array([1.0])).item()
        assert mx.allclose(retrieved["uncond"][1], mx.array([4.0])).item()


class TestZeroNormDegenerate:
    def test_zero_seed_forces_compute_and_resets_accumulator(self):
        # An all-zeros modulated input at step 0 makes the relative-L1 delta
        # undefined (0/0). The next non-boundary step must force a compute
        # and reset the accumulator instead of propagating inf/nan.
        c = make_controller(num_steps=10, rel_l1_thresh=0.1)
        c.should_compute(0, mx.zeros((4,)))
        assert c.should_compute(1, mx.full((4,), 1.0)) is True
        # The forced compute re-seeded with the step-1 input and zeroed the
        # accumulator: a subsequent tiny delta must skip.
        assert c.should_compute(2, mx.full((4,), 1.001)) is False


class TestConstructorAndStepValidation:
    def test_nonpositive_num_steps_raises(self):
        with pytest.raises(ValueError):
            make_controller(num_steps=0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            make_controller(rel_l1_thresh=-0.1)

    def test_out_of_range_step_index_raises(self):
        c = make_controller(num_steps=4)
        x = mx.ones((1, 4))
        with pytest.raises(ValueError):
            c.should_compute(4, x)
        with pytest.raises(ValueError):
            c.should_compute(-1, x)


class TestMaxConsecutiveSkips:
    """Cap on back-to-back skips (SeaCache / cache-dit pattern)."""

    @staticmethod
    def tiny(step):
        # Rescaled per-step delta ≈ 2e-4, far below the threshold: skips forever uncapped.
        return mx.full((4,), 1.0001**step)

    def make(self, cap, num_steps=20):
        return TeaCacheController(
            num_steps=num_steps,
            rel_l1_thresh=0.1,
            coefficients=LINEAR_COEFFS,
            max_consecutive_skips=cap,
        )

    def test_uncapped_by_default(self):
        c = make_controller(num_steps=20)
        c.should_compute(0, self.tiny(0))
        assert [c.should_compute(s, self.tiny(s)) for s in range(1, 8)] == [False] * 7

    def test_cap_forces_compute_after_n_skips(self):
        c = self.make(2)
        c.should_compute(0, self.tiny(0))
        got = [c.should_compute(s, self.tiny(s)) for s in range(1, 10)]
        assert got == [False, False, True, False, False, True, False, False, True]

    def test_counter_restarts_after_threshold_compute(self):
        # A compute triggered by the threshold (not by the cap) must also restart
        # the count, so the next forced compute comes a full `cap` skips later.
        c = self.make(2)
        c.should_compute(0, mx.full((4,), 1.0))
        assert c.should_compute(1, mx.full((4,), 1.0001)) is False
        assert c.should_compute(2, mx.full((4,), 5.0)) is True  # threshold hit
        assert c.should_compute(3, mx.full((4,), 5.0005)) is False
        assert c.should_compute(4, mx.full((4,), 5.001)) is False
        assert c.should_compute(5, mx.full((4,), 5.0015)) is True  # cap

    def test_forced_compute_resets_accumulator(self):
        # Rescaled delta 0.04 per step with thresh 0.1: uncapped, step 3 computes.
        # Capped at 1, step 2 is forced; the accumulator restarts, so step 3 skips.
        c = self.make(1)
        c.should_compute(0, mx.full((4,), 1.0))
        assert c.should_compute(1, mx.full((4,), 1.02)) is False
        assert c.should_compute(2, mx.full((4,), 1.0404)) is True
        assert c.should_compute(3, mx.full((4,), 1.061208)) is False

    def test_reset_clears_counter(self):
        c = self.make(2)
        c.should_compute(0, self.tiny(0))
        c.should_compute(1, self.tiny(1))
        c.should_compute(2, self.tiny(2))
        c.reset()
        c.should_compute(0, self.tiny(0))
        assert c.should_compute(1, self.tiny(1)) is False
        assert c.should_compute(2, self.tiny(2)) is False

    @pytest.mark.parametrize("cap", [0, -1])
    def test_nonpositive_cap_raises(self, cap):
        with pytest.raises(ValueError, match="max_consecutive_skips"):
            self.make(cap)


class TestLowPrecisionInputs:
    def test_bf16_distance_is_computed_in_float32(self):
        # bf16 reductions round their result to 8 mantissa bits (~0.4 %): the
        # relative-L1 distance must match a float64 reference much closer.
        import numpy as np

        from mlx_arsenal.diffusion._cache_common import RelL1State

        rng = np.random.default_rng(0)
        a = rng.standard_normal((1, 4096, 64)).astype(np.float32)
        b = a + 0.013 * rng.standard_normal(a.shape).astype(np.float32)
        xa, xb = mx.array(a).astype(mx.bfloat16), mx.array(b).astype(mx.bfloat16)
        ref_a = np.array(xa.astype(mx.float32), dtype=np.float64)
        ref_b = np.array(xb.astype(mx.float32), dtype=np.float64)
        expected = np.abs(ref_b - ref_a).mean() / np.abs(ref_a).mean()
        state = RelL1State("unused")
        state.seed(xa)
        got = state.delta(xb)
        assert got == pytest.approx(expected, rel=1e-5)

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

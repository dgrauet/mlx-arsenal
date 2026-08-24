"""Tests for mlx_arsenal.diffusion.window_residual."""

import mlx.core as mx
import pytest

from mlx_arsenal.diffusion import WindowResidualController


class TestFixed:
    def test_refresh_every_k(self):
        ctrl = WindowResidualController.fixed(num_steps=10, refresh_every=3)
        # Steps 0, 3, 6 (modulo), and 9 (boundary) → True.
        # Steps 1, 2, 4, 5, 7, 8 → False.
        expected = [True, False, False, True, False, False, True, False, False, True]
        for step in range(10):
            assert ctrl.should_refresh(step) is expected[step], f"step {step}"

    def test_boundary_steps_force_refresh(self):
        # refresh_every=5, num_steps=4 → step 3 (last) is also boundary.
        ctrl = WindowResidualController.fixed(num_steps=4, refresh_every=5)
        assert ctrl.should_refresh(0) is True
        assert ctrl.should_refresh(1) is False
        assert ctrl.should_refresh(2) is False
        assert ctrl.should_refresh(3) is True  # boundary

    def test_should_refresh_ignores_attn_input(self):
        ctrl = WindowResidualController.fixed(num_steps=5, refresh_every=2)
        # Passing attn_input is allowed but ignored in fixed mode.
        x = mx.ones((1, 2, 4, 4))
        assert ctrl.should_refresh(1, x) is False
        assert ctrl.should_refresh(2, x) is True
        assert ctrl.should_refresh(2, None) is True

    def test_validation(self):
        with pytest.raises(ValueError):
            WindowResidualController.fixed(num_steps=1, refresh_every=2)
        with pytest.raises(ValueError):
            WindowResidualController.fixed(num_steps=5, refresh_every=0)
        with pytest.raises(ValueError):
            WindowResidualController.fixed(num_steps=5, refresh_every=-1)
        ctrl = WindowResidualController.fixed(num_steps=5, refresh_every=2)
        with pytest.raises(ValueError):
            ctrl.should_refresh(-1)
        with pytest.raises(ValueError):
            ctrl.should_refresh(5)


_CACHE_FACTORIES = [
    pytest.param(
        lambda: WindowResidualController.fixed(num_steps=4, refresh_every=2),
        id="fixed",
    ),
    pytest.param(
        lambda: WindowResidualController.scheduled(num_steps=4, refresh_steps=[2]),
        id="scheduled",
    ),
    pytest.param(
        lambda: WindowResidualController.adaptive(num_steps=4, rel_l1_thresh=0.1),
        id="adaptive",
    ),
]


class TestCommonCache:
    @pytest.mark.parametrize("factory", _CACHE_FACTORIES)
    def test_cache_and_previous_residual(self, factory):
        ctrl = factory()
        r = mx.ones((1, 2, 4, 4)) * 3.14
        ctrl.cache_residual(r)
        out = ctrl.previous_residual
        assert mx.array_equal(out, r).item()

    @pytest.mark.parametrize("factory", _CACHE_FACTORIES)
    def test_previous_residual_raises_before_cache(self, factory):
        ctrl = factory()
        with pytest.raises(RuntimeError):
            _ = ctrl.previous_residual

    @pytest.mark.parametrize("factory", _CACHE_FACTORIES)
    def test_step_index_out_of_range_raises(self, factory):
        # All factories build num_steps=4 controllers; the range check runs
        # before mode dispatch, so no attn_input is needed.
        ctrl = factory()
        with pytest.raises(ValueError):
            ctrl.should_refresh(4)
        with pytest.raises(ValueError):
            ctrl.should_refresh(-1)

    @pytest.mark.parametrize("factory", _CACHE_FACTORIES)
    def test_reset_clears_state(self, factory):
        ctrl = factory()
        ctrl.cache_residual(mx.ones((1, 2, 4, 4)))
        ctrl.reset()
        with pytest.raises(RuntimeError):
            _ = ctrl.previous_residual


class TestScheduled:
    def test_explicit_steps_refresh(self):
        ctrl = WindowResidualController.scheduled(num_steps=10, refresh_steps=[2, 5, 7])
        # 2, 5, 7 from list; 0 and 9 from boundary; others False.
        expected = [True, False, True, False, False, True, False, True, False, True]
        for step in range(10):
            assert ctrl.should_refresh(step) is expected[step], f"step {step}"

    def test_boundary_always_refresh_even_if_not_in_list(self):
        ctrl = WindowResidualController.scheduled(num_steps=10, refresh_steps=[5])
        assert ctrl.should_refresh(0) is True
        assert ctrl.should_refresh(9) is True

    def test_steps_outside_schedule_no_refresh(self):
        ctrl = WindowResidualController.scheduled(num_steps=6, refresh_steps=[3])
        assert ctrl.should_refresh(1) is False
        assert ctrl.should_refresh(2) is False
        assert ctrl.should_refresh(4) is False

    def test_validation(self):
        with pytest.raises(ValueError):
            WindowResidualController.scheduled(num_steps=10, refresh_steps=[])
        with pytest.raises(ValueError):
            WindowResidualController.scheduled(num_steps=10, refresh_steps=[10])
        with pytest.raises(ValueError):
            WindowResidualController.scheduled(num_steps=10, refresh_steps=[-1])
        with pytest.raises(ValueError):
            WindowResidualController.scheduled(num_steps=1, refresh_steps=[0])


class TestAdaptive:
    def test_refresh_when_input_changes(self):
        ctrl = WindowResidualController.adaptive(num_steps=5, rel_l1_thresh=0.05)
        x0 = mx.ones((1, 2, 4, 4))
        x1 = mx.ones((1, 2, 4, 4)) * 2.0  # 100% change → above thresh
        # Step 0 is boundary: True regardless, but it also seeds.
        assert ctrl.should_refresh(0, x0) is True
        assert ctrl.should_refresh(1, x1) is True  # delta = 1.0 >= 0.05

    def test_skip_when_input_static(self):
        ctrl = WindowResidualController.adaptive(num_steps=5, rel_l1_thresh=0.05)
        x = mx.ones((1, 2, 4, 4))
        assert ctrl.should_refresh(0, x) is True  # boundary
        assert ctrl.should_refresh(1, x) is False  # delta = 0
        assert ctrl.should_refresh(2, x) is False
        assert ctrl.should_refresh(3, x) is False
        assert ctrl.should_refresh(4, x) is True  # boundary

    def test_boundary_always_refresh(self):
        ctrl = WindowResidualController.adaptive(num_steps=4, rel_l1_thresh=0.1)
        x = mx.ones((1, 2, 4, 4))
        assert ctrl.should_refresh(0, x) is True
        assert ctrl.should_refresh(3, x) is True

    def test_raises_when_attn_input_missing(self):
        ctrl = WindowResidualController.adaptive(num_steps=5, rel_l1_thresh=0.1)
        # Step 0 is boundary so attn_input is still required for seeding,
        # but the boundary path bypasses the input check. Hit a mid step.
        ctrl.should_refresh(0, mx.ones((1, 2, 4, 4)))  # seed
        with pytest.raises(RuntimeError):
            ctrl.should_refresh(1)  # no attn_input

    def test_zero_norm_seed_forces_refresh(self):
        # All-zeros seed at the boundary → delta undefined at step 1 → the
        # degenerate branch forces a refresh instead of propagating inf/nan.
        ctrl = WindowResidualController.adaptive(num_steps=5, rel_l1_thresh=0.05)
        assert ctrl.should_refresh(0, mx.zeros((1, 2, 4, 4))) is True
        assert ctrl.should_refresh(1, mx.ones((1, 2, 4, 4))) is True
        # Re-seeded with the step-1 input: a static input now skips.
        assert ctrl.should_refresh(2, mx.ones((1, 2, 4, 4))) is False

    def test_validation(self):
        with pytest.raises(ValueError):
            WindowResidualController.adaptive(num_steps=5, rel_l1_thresh=0.0)
        with pytest.raises(ValueError):
            WindowResidualController.adaptive(num_steps=5, rel_l1_thresh=-0.1)
        with pytest.raises(ValueError):
            WindowResidualController.adaptive(num_steps=1, rel_l1_thresh=0.1)

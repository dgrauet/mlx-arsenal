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


class TestCommonCache:
    def test_cache_and_previous_residual(self):
        ctrl = WindowResidualController.fixed(num_steps=4, refresh_every=2)
        r = mx.ones((1, 2, 4, 4)) * 3.14
        ctrl.cache_residual(r)
        out = ctrl.previous_residual
        assert mx.array_equal(out, r).item()

    def test_previous_residual_raises_before_cache(self):
        ctrl = WindowResidualController.fixed(num_steps=4, refresh_every=2)
        with pytest.raises(RuntimeError):
            _ = ctrl.previous_residual

    def test_reset_clears_state(self):
        ctrl = WindowResidualController.fixed(num_steps=4, refresh_every=2)
        ctrl.cache_residual(mx.ones((1, 2, 4, 4)))
        ctrl.reset()
        with pytest.raises(RuntimeError):
            _ = ctrl.previous_residual

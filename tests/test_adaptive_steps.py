"""Tests for the CAT-Flow curvature-adaptive step controller."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any
from mlx_arsenal.diffusion import CurvatureAdaptiveStepper


def field(x, t):
    """Analytic, time-varying velocity field with non-zero acceleration."""
    return -x * (1.0 + 3.0 * t * t) + np.sin(3.0 * t + x)


def reference_dts(x0, mode, lam, beta=0.3, dt_min=0.01, warmup=0):
    """Transcription of CAT-Flow Algorithms 1 (OT) and 2 (OV), batch 1."""
    x, t, steps, dts = x0.astype(np.float64), 0.0, 0, []
    u_prev, dt_prev = np.zeros_like(x), 0.0
    m2, m1 = np.zeros_like(x), np.zeros_like(x)
    while t < 1:
        u = field(x, t)
        if mode == "ot":
            if steps == 0:
                dt = dt_min
            else:
                a = (u - u_prev) / dt_prev
                n = np.linalg.norm(a)
                dt = lam / n if n > 0 else math.inf
        else:
            m2 = beta * m2 + (1 - beta) * ((1 - t) * u) ** 2
            m1 = beta * m1 + (1 - beta) * (1 - t) * u
            n = math.sqrt(np.linalg.norm(m2 - m1**2))
            dt = lam / n if n > 0 else math.inf
            if steps == 0:
                dt *= math.sqrt(1 - beta)
        if steps < warmup:
            dt = dt_min
        dt = min(max(dt, dt_min), 1 - t)
        x = x + dt * u
        t = 1.0 if dt == 1 - t else t + dt
        dt_prev, u_prev = dt, u
        steps += 1
        dts.append(dt)
    return dts


def run(stepper, x0):
    x, dts = x0.astype(np.float64), []
    while not stepper.done:
        u = field(x, stepper.t)
        dt = stepper.step(array_from_any(u.astype(np.float32)))
        x = x + dt * u
        dts.append(dt)
    return dts


X0 = np.random.default_rng(0).standard_normal((1, 6, 5, 3))


class TestAlgorithmParity:
    # λ per mode so each trajectory takes a real number of steps (12-51) on this
    # small tensor; OV norms are much smaller than OT's here.
    @pytest.mark.parametrize(
        ("mode", "lam"), [("ot", 0.5), ("ot", 2.0), ("ov", 0.01), ("ov", 0.03)]
    )
    def test_dt_sequence_matches_paper(self, mode, lam):
        ref = reference_dts(X0, mode, lam)
        got = run(CurvatureAdaptiveStepper(lam, mode=mode), X0)
        assert len(ref) >= 12
        assert len(got) == len(ref)
        np.testing.assert_allclose(got, ref, rtol=1e-4)

    @pytest.mark.parametrize(("mode", "lam"), [("ot", 2.0), ("ov", 0.03)])
    def test_warmup_steps(self, mode, lam):
        ref = reference_dts(X0, mode, lam, warmup=3)
        got = run(CurvatureAdaptiveStepper(lam, mode=mode, warmup_steps=3), X0)
        assert got[:3] == [0.01] * 3
        np.testing.assert_allclose(got, ref, rtol=1e-4)

    def test_larger_scale_takes_fewer_steps(self):
        small = run(CurvatureAdaptiveStepper(0.01), X0)
        large = run(CurvatureAdaptiveStepper(0.1), X0)
        assert len(large) < len(small)


class TestTimeline:
    @pytest.mark.parametrize("mode", ["ot", "ov"])
    def test_lands_exactly_on_one(self, mode):
        s = CurvatureAdaptiveStepper(1.0, mode=mode)
        total = sum(run(s, X0))
        assert s.t == 1.0
        assert s.done
        assert math.isclose(total, 1.0, rel_tol=1e-9)

    def test_t_start(self):
        s = CurvatureAdaptiveStepper(1.0, t_start=0.4)
        assert s.t == 0.4
        assert math.isclose(sum(run(s, X0)), 0.6, rel_tol=1e-9)

    def test_step_after_done_raises(self):
        s = CurvatureAdaptiveStepper(1.0)
        run(s, X0)
        with pytest.raises(RuntimeError):
            s.step(mx.ones((1, 3)))

    def test_reset(self):
        s = CurvatureAdaptiveStepper(1.0, mode="ot")
        first = run(s, X0)
        s.reset()
        assert s.t == 0.0 and s.steps == 0 and not s.done
        assert run(s, X0) == first


class TestBehaviour:
    @pytest.mark.parametrize("mode", ["ot", "ov"])
    def test_sign_invariant(self, mode):
        # A diffusers-convention output (v = -u) gives the same steps.
        a, b = CurvatureAdaptiveStepper(1.0, mode=mode), CurvatureAdaptiveStepper(1.0, mode=mode)
        x = X0.copy()
        while not a.done:
            u = field(x, a.t).astype(np.float32)
            da, db = a.step(array_from_any(u)), b.step(array_from_any(-u))
            assert math.isclose(da, db, rel_tol=1e-6)
            x = x + da * u

    @pytest.mark.parametrize("mode", ["ot", "ov"])
    def test_batch_takes_the_smallest_step(self, mode):
        # Row 1 is a copy of row 0 scaled by 4: larger norms, smaller steps.
        x0 = np.concatenate([X0, 4 * X0], axis=0)
        s = CurvatureAdaptiveStepper(1.0, mode=mode)
        solo = CurvatureAdaptiveStepper(1.0, mode=mode)
        u0 = field(x0, 0.0).astype(np.float32)
        s.step(array_from_any(u0))
        solo.step(array_from_any(u0[1:]))
        u1 = field(x0 + 0.01 * u0, s.t).astype(np.float32)
        assert s.step(array_from_any(u1)) == solo.step(array_from_any(u1[1:]))

    @pytest.mark.parametrize("mode", ["ot", "ov"])
    def test_constant_velocity_jumps_to_the_end(self, mode):
        # OT: zero acceleration (its first step is always dt_min). OV: a zero
        # velocity has zero variance, so its very first step already jumps.
        u = mx.ones((1, 4)) if mode == "ot" else mx.zeros((1, 4))
        s = CurvatureAdaptiveStepper(1.0, mode=mode)
        if mode == "ot":
            s.step(u)
        remaining = 1.0 - s.t
        assert s.step(u) == remaining
        assert s.done

    def test_dt_max_caps_the_jump(self):
        s = CurvatureAdaptiveStepper(1.0, mode="ot", dt_max=0.25)
        s.step(mx.ones((1, 4)))
        assert s.step(mx.ones((1, 4))) == 0.25

    def test_bf16_velocity(self):
        s = CurvatureAdaptiveStepper(1.0)
        dt = s.step(mx.ones((1, 8), dtype=mx.bfloat16))
        assert isinstance(dt, float) and dt > 0

    def test_shape_change_raises(self):
        s = CurvatureAdaptiveStepper(0.01)
        s.step(mx.ones((1, 4)))
        with pytest.raises(ValueError):
            s.step(mx.ones((1, 5)))


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scale": 0.0},
            {"scale": 1.0, "mode": "adam"},
            {"scale": 1.0, "beta": 1.0},
            {"scale": 1.0, "beta": -0.1},
            {"scale": 1.0, "dt_min": 0.0},
            {"scale": 1.0, "dt_min": 1.5},
            {"scale": 1.0, "dt_max": 0.001},
            {"scale": 1.0, "warmup_steps": -1},
            {"scale": 1.0, "t_start": 1.0},
            {"scale": 1.0, "t_start": -0.1},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            CurvatureAdaptiveStepper(**kwargs)

    def test_scalar_velocity_raises(self):
        with pytest.raises(ValueError):
            CurvatureAdaptiveStepper(1.0).step(mx.array(1.0))

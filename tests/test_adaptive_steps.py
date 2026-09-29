"""Tests for the CAT-Flow curvature-adaptive step controller."""

import math
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any, item_float
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


class TestStateDetails:
    def test_ot_differences_over_the_step_actually_taken(self):
        # Step 1's raw dt (5.0) is clipped to dt_max = 0.05; the next acceleration
        # must divide by the 0.05 that was taken, not by the raw 5.0.
        s = CurvatureAdaptiveStepper(1.0, mode="ot", dt_max=0.05)
        u0, u1 = mx.zeros((1, 4)), mx.full((1, 4), 0.001)
        assert s.step(u0) == 0.01
        assert s.step(u1) == 0.05
        u2 = u1 + 1.0
        expected = 1.0 / (np.linalg.norm(np.ones(4)) / 0.05)  # 0.025
        assert s.step(u2) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("t_start", [0.0, 0.5])
    def test_ov_first_step_bias_correction(self, t_start):
        # Keyed on the first step, not on t == 0, so it also applies from t_start.
        lam, beta = 0.05, 0.3
        scaled = (1 - t_start) * np.ones(4)
        m2, m1 = (1 - beta) * scaled**2, (1 - beta) * scaled
        raw = lam / math.sqrt(np.linalg.norm(m2 - m1**2))
        s = CurvatureAdaptiveStepper(lam, beta=beta, t_start=t_start)
        dt = s.step(mx.ones((1, 4)))
        assert 0.01 < dt < 1 - t_start  # not masked by the clip
        assert dt == pytest.approx(raw * math.sqrt(1 - beta), rel=1e-6)


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

    @pytest.mark.parametrize("shape", [(), (8,)])
    def test_velocity_without_batch_axis_raises(self, shape):
        # A flat (N,) vector would otherwise be read as N one-element samples.
        with pytest.raises(ValueError, match="batch"):
            CurvatureAdaptiveStepper(1.0).step(mx.ones(shape))

    @pytest.mark.parametrize("mode", ["ot", "ov"])
    def test_nan_velocity_raises(self, mode):
        # A NaN step would leave t at NaN and `while not done` would never end.
        s = CurvatureAdaptiveStepper(0.001, mode=mode)
        s.step(mx.ones((1, 4)))
        with pytest.raises(ValueError, match="NaN"):
            s.step(mx.array([[1.0, float("nan"), 0.0, 1.0]]))


def _load_recipe():
    note = Path(__file__).parent.parent / "docs" / "research" / "adaptive-flow-steps.md"
    match = re.search(
        r"<!-- adaptive-steps-recipe -->\s*```python\n(.*?)```", note.read_text(), re.S
    )
    assert match, "adaptive-steps recipe block not found in the research note"
    namespace: dict[str, Any] = {}
    exec(match.group(1), namespace)
    return namespace["adaptive_euler"]


class TestRecipeOnGaussianFlow:
    """Data N(mu, s^2 I), x_sigma = (1 - sigma) x0 + sigma eps, diffusers velocity eps - x0.

    The marginal velocity is closed-form and the ODE maps the start noise z to
    mu + s z exactly, so the sampler's error is measurable.
    """

    MU, S = 2.0, 0.5

    def velocity(self, x, sigma):
        var = (1 - sigma) ** 2 * self.S**2 + sigma**2
        centred = x - (1 - sigma) * self.MU
        e_eps = sigma / var * centred
        e_x0 = self.MU + (1 - sigma) * self.S**2 / var * centred
        return e_eps - e_x0

    def test_converges_as_scale_decreases(self):
        adaptive_euler = _load_recipe()
        z = mx.array(np.random.default_rng(3).standard_normal((1, 16, 16, 4)).astype(np.float32))
        exact = self.MU + self.S * z
        errors, nfes = [], []
        for lam in (0.2, 0.02):  # OV: about 10 and 28 steps
            out, nfe = adaptive_euler(self.velocity, z, CurvatureAdaptiveStepper(lam))
            errors.append(item_float(mx.abs(out - exact).max()))
            nfes.append(nfe)
        assert nfes[0] < nfes[1] < 100  # smaller lambda, more steps
        assert errors[1] < errors[0] < 0.5  # and a smaller error
        assert errors[1] < 0.15

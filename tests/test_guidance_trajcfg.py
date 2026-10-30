"""Tests for TrajCFG token-wise guidance scales."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal.diffusion import classifier_free_guidance, trajectory_guidance_scale

KW = {"min_scale": 4.0, "max_scale": 9.0, "rho_lo": 0.95}


def _cos64(a, b, eps):
    na = np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), eps)
    nb = np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), eps)
    return np.clip(np.sum(a * b, axis=-1, keepdims=True) / (na * nb), -1.0, 1.0)


def _g64(q, s_min, s_max, rho_lo):
    return s_min + (s_max - s_min) * np.clip((1.0 - q) / (1.0 - rho_lo), 0.0, 1.0)


def trajcfg_reference(z0, sigmas, velocity, base, s_min, s_max, rho_lo, eps=1e-6):
    """Float64 transliteration of the paper's Algorithm 1 with an Euler solver.

    ``velocity(z, sigma)`` returns ``(cond, uncond)``; every step k >= 2 adapts.
    """
    z, cache, scales = z0.astype(np.float64), [], []
    for k in range(len(sigmas) - 1):
        s = np.full(z.shape[:-1] + (1,), base)
        if k >= 2:
            s = _g64(_cos64(cache[k - 2], cache[k - 1], eps), s_min, s_max, rho_lo)
            if k >= 3:
                q_long = _cos64(cache[k - 3], cache[k - 1], eps)
                d_short = abs(sigmas[k - 1] - sigmas[k - 2])
                d_long = abs(sigmas[k - 1] - sigmas[k - 3])
                q_t = np.cos(d_short / d_long * np.arccos(q_long))
                s = 0.8 * s + 0.2 * _g64(q_t, s_min, s_max, rho_lo)
        vc, vu = velocity(z, sigmas[k])
        v = vu + s * (vc - vu)
        z = z + (sigmas[k + 1] - sigmas[k]) * v
        cache.append(v)
        scales.append(s)
    return z, scales


def trajcfg_loop(z0, sigmas, velocity, base, s_min, s_max, rho_lo):
    """The loop a caller writes with the primitive, in MLX float32."""
    z, hist, hsig, scales = mx.array(z0), [], [], []
    for k in range(len(sigmas) - 1):
        if len(hist) >= 2:
            s = trajectory_guidance_scale(
                hist, hsig, min_scale=s_min, max_scale=s_max, rho_lo=rho_lo
            )
        else:
            s = mx.full(z.shape[:-1] + (1,), base)
        vc, vu = velocity(z, sigmas[k])
        v = classifier_free_guidance(mx.array(vc), mx.array(vu), s)
        z = z + (sigmas[k + 1] - sigmas[k]) * v
        hist, hsig = [*hist, v][-3:], [*hsig, sigmas[k]][-3:]
        scales.append(s)
    return z, scales


def _toy_velocity(seed):
    """Nonlinear (cond, uncond) velocity field with token-dependent turning."""
    rng = np.random.default_rng(seed)
    C = 6
    Wc, Wu = rng.standard_normal((2, C, C)) / math.sqrt(C)
    bc, bu = rng.standard_normal((2, C))

    def velocity(z, sigma):
        zn = np.asarray(z, dtype=np.float64)
        vc = np.tanh(zn @ Wc) + sigma * bc + 0.3 * np.sin(3.0 * sigma * zn)
        vu = np.tanh(zn @ Wu) + sigma * bu
        return vc, vu

    return velocity


class TestReferenceParity:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize(
        "sigmas",
        [
            list(np.linspace(1.0, 0.0, 11)),
            # shifted (non-uniform) schedule: the stride-2 normalisation matters
            [4 * s / (1 + 3 * s) for s in np.linspace(1.0, 0.0, 13)],
        ],
    )
    def test_loop_matches_algorithm_1(self, seed, sigmas):
        rng = np.random.default_rng(100 + seed)
        z0 = rng.standard_normal((2, 5, 7, 6)).astype(np.float32)
        velocity = _toy_velocity(seed)
        z_ref, s_ref = trajcfg_reference(z0, sigmas, velocity, 5.0, 4.0, 9.0, 0.95)
        z, s = trajcfg_loop(z0, sigmas, velocity, 5.0, 4.0, 9.0, 0.95)
        for k, (a, b) in enumerate(zip(s, s_ref, strict=True)):
            np.testing.assert_allclose(np.array(a), b, atol=2e-3, err_msg=f"step {k}")
        np.testing.assert_allclose(np.array(z), z_ref, rtol=1e-4, atol=1e-4)
        # The toy field must exercise the whole ramp, not one end of it.
        spread = np.concatenate([x.ravel() for x in s_ref[2:]])
        assert spread.min() < 5.0 and spread.max() > 8.0


class TestProperties:
    def test_constant_direction_gives_min_scale(self):
        v = mx.random.normal((2, 4, 3, 8), key=mx.random.key(0))
        s = trajectory_guidance_scale([v, 2.0 * v, 0.5 * v], [1.0, 0.9, 0.8], **KW)
        assert s.shape == (2, 4, 3, 1) and s.dtype == mx.float32
        np.testing.assert_allclose(np.array(s), 4.0, atol=1e-4)

    def test_reversed_direction_gives_max_scale(self):
        v = mx.random.normal((3, 8), key=mx.random.key(1))
        s = trajectory_guidance_scale([v, -v], [1.0, 0.9], **KW)
        np.testing.assert_allclose(np.array(s), 9.0)

    def test_linear_ramp_and_saturation(self):
        # Unit vectors at angle theta: cos = 1, 0.975, 0.95, 0.5 -> 4, 6.5, 9, 9.
        cosines = [1.0, 0.975, 0.95, 0.5]
        a = mx.array([[1.0, 0.0]] * 4)
        b = mx.array([[c, math.sqrt(1 - c * c)] for c in cosines])
        s = trajectory_guidance_scale([a, b], [1.0, 0.5], **KW)
        np.testing.assert_allclose(np.array(s).ravel(), [4.0, 6.5, 9.0, 9.0], atol=1e-3)

    def test_scale_is_per_token(self):
        a = mx.array([[1.0, 0.0], [1.0, 0.0]])
        b = mx.array([[1.0, 0.0], [0.0, 1.0]])
        s = np.array(trajectory_guidance_scale([a, b], [1.0, 0.5], **KW)).ravel()
        np.testing.assert_allclose(s, [4.0, 9.0], atol=1e-5)

    def test_long_weight_zero_equals_adjacent_only(self):
        k = mx.random.split(mx.random.key(2), 3)
        h = [mx.random.normal((4, 5, 8), key=k[i]) for i in range(3)]
        three = trajectory_guidance_scale(h, [1.0, 0.8, 0.5], long_weight=0.0, **KW)
        two = trajectory_guidance_scale(h[1:], [0.8, 0.5], **KW)
        np.testing.assert_allclose(np.array(three), np.array(two), atol=1e-6)

    def test_long_weight_one_uses_normalised_stride2_only(self):
        # Oldest and latest at 90 degrees; equal intervals halve the angle to
        # 45 degrees, cos = 0.7071 -> scale saturates at max.
        a, mid, c = mx.array([[1.0, 0.0]]), mx.array([[1.0, 0.0]]), mx.array([[0.0, 1.0]])
        s = trajectory_guidance_scale(
            [a, mid, c], [1.0, 0.5, 0.0], long_weight=1.0, min_scale=0.0, max_scale=1.0, rho_lo=0.0
        )
        # g(q) with rho_lo = 0: (1 - q) / 1 -> 1 - cos(pi / 4)
        np.testing.assert_allclose(np.array(s), 1.0 - math.cos(math.pi / 4), atol=1e-5)

    def test_interval_ratio_uses_sigma_gaps(self):
        # Same velocities, the stride-2 angle is scaled by short / long.
        a, mid, c = mx.array([[1.0, 0.0]]), mx.array([[1.0, 0.0]]), mx.array([[0.0, 1.0]])
        kw = {"long_weight": 1.0, "min_scale": 0.0, "max_scale": 1.0, "rho_lo": -1.0}
        for sig, ratio in (([1.0, 0.5, 0.0], 0.5), ([1.0, 0.75, 0.0], 0.75)):
            s = trajectory_guidance_scale([a, mid, c], sig, **kw)
            expect = (1.0 - math.cos(ratio * math.pi / 2)) / 2.0
            np.testing.assert_allclose(np.array(s), expect, atol=1e-5)

    def test_zero_velocity_is_finite(self):
        z = mx.zeros((2, 4))
        v = mx.ones((2, 4))
        for hist in ([z, v], [v, z], [z, z], [z, v, z]):
            sig = [1.0, 0.5, 0.2][: len(hist)]
            s = np.array(trajectory_guidance_scale(hist, sig, **KW))
            assert np.isfinite(s).all()
            # A floored norm reads as cosine 0: no information, full guidance.
            np.testing.assert_allclose(s, 9.0)

    def test_rounding_above_one_does_not_nan(self):
        # Nearly parallel large vectors: the raw ratio can exceed 1 by rounding.
        v = mx.array([[1e3, 1e-3, 3.3e2]], dtype=mx.float32)
        s = np.array(trajectory_guidance_scale([v, v * 1.0000001, v], [1.0, 0.5, 0.1], **KW))
        assert np.isfinite(s).all()
        np.testing.assert_allclose(s, 4.0, atol=1e-3)

    def test_low_precision_inputs(self):
        v = mx.random.normal((2, 3, 16), key=mx.random.key(3))
        w = v + 0.1 * mx.random.normal(v.shape, key=mx.random.key(4))
        ref = trajectory_guidance_scale([v, w], [1.0, 0.9], **KW)
        s = trajectory_guidance_scale(
            [v.astype(mx.bfloat16), w.astype(mx.bfloat16)], [1.0, 0.9], **KW
        )
        assert s.dtype == mx.float32
        np.testing.assert_allclose(np.array(s), np.array(ref), atol=0.2)

    def test_cfg_accepts_token_scales(self):
        cond, uncond = mx.ones((2, 3, 4)), mx.zeros((2, 3, 4))
        s = mx.array([[[2.0], [3.0], [4.0]]] * 2)
        out = classifier_free_guidance(cond, uncond, s)
        np.testing.assert_allclose(np.array(out)[0, :, 0], [2.0, 3.0, 4.0])


class TestValidation:
    v = mx.ones((2, 4))

    @pytest.mark.parametrize("n", [0, 1, 4])
    def test_history_length(self, n):
        with pytest.raises(ValueError, match="2 or 3"):
            trajectory_guidance_scale([self.v] * n, [1.0] * n, **KW)

    def test_sigmas_length(self):
        with pytest.raises(ValueError, match="sigmas must match"):
            trajectory_guidance_scale([self.v, self.v], [1.0], **KW)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="one shape"):
            trajectory_guidance_scale([self.v, mx.ones((2, 5))], [1.0, 0.5], **KW)

    def test_scalar_velocity(self):
        with pytest.raises(ValueError, match="channel axis"):
            trajectory_guidance_scale([mx.array(1.0), mx.array(1.0)], [1.0, 0.5], **KW)

    @pytest.mark.parametrize(
        ("lo", "hi"), [(-1.0, 2.0), (5.0, 5.0), (6.0, 5.0), (1.0, math.inf), (math.nan, 2.0)]
    )
    def test_scale_range(self, lo, hi):
        with pytest.raises(ValueError, match="min_scale"):
            trajectory_guidance_scale([self.v, self.v], [1.0, 0.5], min_scale=lo, max_scale=hi)

    @pytest.mark.parametrize("rho", [1.0, -1.5, math.nan])
    def test_rho_lo(self, rho):
        with pytest.raises(ValueError, match="rho_lo"):
            trajectory_guidance_scale([self.v, self.v], [1.0, 0.5], **{**KW, "rho_lo": rho})

    @pytest.mark.parametrize("w", [-0.1, 1.1, math.nan])
    def test_long_weight(self, w):
        with pytest.raises(ValueError, match="long_weight"):
            trajectory_guidance_scale([self.v, self.v], [1.0, 0.5], long_weight=w, **KW)

    @pytest.mark.parametrize("eps", [0.0, -1.0, math.nan])
    def test_eps(self, eps):
        with pytest.raises(ValueError, match="eps"):
            trajectory_guidance_scale([self.v, self.v], [1.0, 0.5], eps=eps, **KW)

    @pytest.mark.parametrize(
        "sig", [[1.0, 1.0], [1.0, 1.0, 0.5], [1.0, 0.5, 0.5], [1.0, math.nan], [math.inf, 0.5]]
    )
    def test_sigmas_values(self, sig):
        with pytest.raises(ValueError, match="sigmas"):
            trajectory_guidance_scale([self.v] * len(sig), sig, **KW)

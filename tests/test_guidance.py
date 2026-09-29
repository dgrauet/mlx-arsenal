"""Tests for posterior-mean-capped guidance (PMC-CFG)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any
from mlx_arsenal.diffusion import classifier_free_guidance, posterior_mean_capped_guidance


def reference(cond, uncond, scale, x, sigma, cap):
    """Transcription of the paper's Appendix D.4, per sample, float64.

    diffusers convention: posterior mean m = x - sigma * v (paper: x + (1 - t) u).
    """
    out = []
    for i in range(cond.shape[0]):
        vc, vu, xi = (a[i].astype(np.float64).ravel() for a in (cond, uncond, x))
        s = sigma[i] if np.ndim(sigma) else sigma
        mc, mu = xi - s * vc, xi - s * vu
        d = mc - mu
        a = d @ d
        if a == 0:
            beta = scale - 1
        else:
            b, q = mc @ d, mc @ mc
            disc = b * b + (cap * cap - 1) * a * q
            beta = 0.0 if disc < 0 else min(scale - 1, max(0.0, (-b + np.sqrt(disc)) / a))
        out.append((vc + beta * (vc - vu)).reshape(cond.shape[1:]))
    return np.stack(out)


def data(seed=0, shape=(3, 8, 6, 4)):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    vc = rng.standard_normal(shape).astype(np.float32)
    vu = vc + 0.5 * rng.standard_normal(shape).astype(np.float32)
    return x, vc, vu


def call(x, vc, vu, scale, sigma, cap):
    return np.array(
        posterior_mean_capped_guidance(
            array_from_any(vc), array_from_any(vu), scale, x=array_from_any(x), sigma=sigma, cap=cap
        )
    )


class TestReferenceParity:
    @pytest.mark.parametrize("cap", [1.0, 1.05, 1.1, 1.5])
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 0.9])
    def test_matches_appendix_d4(self, cap, sigma):
        x, vc, vu = data()
        got = call(x, vc, vu, 7.0, sigma, cap)
        np.testing.assert_allclose(got, reference(vc, vu, 7.0, x, sigma, cap), atol=1e-4, rtol=1e-4)

    def test_per_sample_sigma(self):
        x, vc, vu = data(1)
        sig = np.array([0.2, 0.6, 0.95], dtype=np.float32)
        got = np.array(
            posterior_mean_capped_guidance(
                array_from_any(vc),
                array_from_any(vu),
                5.0,
                x=array_from_any(x),
                sigma=array_from_any(sig),
                cap=1.05,
            )
        )
        np.testing.assert_allclose(got, reference(vc, vu, 5.0, x, sig, 1.05), atol=1e-4, rtol=1e-4)

    def test_zero_d_and_numpy_sigma(self):
        x, vc, vu = data(10)
        ref = reference(vc, vu, 5.0, x, 0.4, 1.05)
        for sig in (mx.array(0.4), np.float32(0.4), np.array(0.4)):
            got = np.array(
                posterior_mean_capped_guidance(
                    array_from_any(vc),
                    array_from_any(vu),
                    5.0,
                    x=array_from_any(x),
                    sigma=sig,
                    cap=1.05,
                )
            )
            np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


class TestProperties:
    @pytest.mark.parametrize("cap", [1.02, 1.1])
    def test_guided_mean_respects_the_cap(self, cap):
        x, vc, vu = data(2)
        sigma = 0.7
        v = call(x, vc, vu, 9.0, sigma, cap)
        for i in range(x.shape[0]):
            m = x[i] - sigma * v[i]
            mc = x[i] - sigma * vc[i]
            assert np.linalg.norm(m) <= cap * np.linalg.norm(mc) * (1 + 1e-4)

    def test_binding_cap_is_tight(self):
        # When the cap binds, beta lies strictly inside (0, scale - 1) and the
        # guided posterior mean sits exactly on the cap.
        x, vc, vu = data(2)
        sigma, cap, scale = 0.5, 1.05, 7.0
        v = call(x, vc, vu, scale, sigma, cap)
        for i in range(x.shape[0]):
            d = vc[i] - vu[i]
            beta = float(((v[i] - vc[i]) * d).sum() / (d * d).sum())
            assert 0.1 < beta < scale - 1 - 0.1
            m, mc = x[i] - sigma * v[i], x[i] - sigma * vc[i]
            assert np.linalg.norm(m) == pytest.approx(cap * np.linalg.norm(mc), rel=1e-4)

    def test_loose_cap_is_plain_cfg(self):
        x, vc, vu = data(3)
        got = call(x, vc, vu, 4.0, 0.5, 1e6)
        ref = np.array(classifier_free_guidance(array_from_any(vc), array_from_any(vu), 4.0))
        np.testing.assert_allclose(got, ref, atol=1e-4)

    def test_scale_one_is_conditional(self):
        x, vc, vu = data(4)
        np.testing.assert_allclose(call(x, vc, vu, 1.0, 0.5, 1.05), vc, atol=1e-6)

    def test_sigma_zero_is_plain_cfg(self):
        # At sigma = 0 both posterior means equal x: no gap, nominal guidance.
        x, vc, vu = data(5)
        got = call(x, vc, vu, 6.0, 0.0, 1.01)
        ref = np.array(classifier_free_guidance(array_from_any(vc), array_from_any(vu), 6.0))
        np.testing.assert_allclose(got, ref, atol=1e-4)

    def test_samples_are_capped_independently(self):
        x, vc, vu = data(6)
        full = call(x, vc, vu, 7.0, 0.6, 1.05)
        for i in range(x.shape[0]):
            one = call(x[i : i + 1], vc[i : i + 1], vu[i : i + 1], 7.0, 0.6, 1.05)
            np.testing.assert_allclose(full[i : i + 1], one, atol=1e-5)

    def test_paper_convention_by_negation(self):
        # Paper convention: t = 1 is data, u = x0 - noise, m = x + (1 - t) u.
        # Pass -u and sigma = 1 - t, negate the result: the paper's capped u.
        x, uc, uu = data(7)
        t, scale, cap = 0.6, 5.0, 1.05
        u = -call(x, -uc, -uu, scale, 1.0 - t, cap)
        expected = []
        for i in range(x.shape[0]):
            xi, c, un = (a[i].astype(np.float64).ravel() for a in (x, uc, uu))
            mc, mu = xi + (1 - t) * c, xi + (1 - t) * un
            d = mc - mu
            a, b, q = d @ d, mc @ d, mc @ mc
            disc = b * b + (cap * cap - 1) * a * q
            beta = min(scale - 1, max(0.0, (-b + np.sqrt(disc)) / a))
            expected.append((c + beta * (c - un)).reshape(x.shape[1:]))
        np.testing.assert_allclose(u, np.stack(expected), atol=1e-4, rtol=1e-4)

    def test_bf16_dtype_preserved(self):
        x, vc, vu = data(8)
        out = posterior_mean_capped_guidance(
            array_from_any(vc).astype(mx.bfloat16),
            array_from_any(vu).astype(mx.bfloat16),
            5.0,
            x=array_from_any(x).astype(mx.bfloat16),
            sigma=0.5,
            cap=1.05,
        )
        assert out.dtype == mx.bfloat16


class TestValidation:
    def _args(self, **kw):
        x, vc, vu = data(9)
        base = {
            "cond": array_from_any(vc),
            "uncond": array_from_any(vu),
            "scale": 5.0,
            "x": array_from_any(x),
            "sigma": 0.5,
            "cap": 1.05,
        }
        base.update(kw)
        return base

    @pytest.mark.parametrize(
        "kw",
        [
            {"scale": 0.5},
            {"cap": 0.0},
            {"cap": 0.99},
            {"sigma": mx.array([0.5, 1.5, 0.5])},
            {"sigma": mx.array(-0.1)},
            {"sigma": -0.1},
            {"sigma": 1.5},
            {"uncond": mx.zeros((3, 8, 6, 5))},
            {"x": mx.zeros((3, 8, 6, 5))},
            {"sigma": mx.array([0.5, 0.5])},
        ],
    )
    def test_invalid_arguments_raise(self, kw):
        with pytest.raises(ValueError):
            posterior_mean_capped_guidance(**self._args(**kw))

    def test_batchless_input_raises(self):
        with pytest.raises(ValueError, match="batch"):
            posterior_mean_capped_guidance(
                mx.ones((4,)), mx.ones((4,)), 5.0, x=mx.ones((4,)), sigma=0.5, cap=1.05
            )

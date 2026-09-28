"""Tests for the SeaCache SEA spectral filter."""

import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_arsenal._typing import array_from_any
from mlx_arsenal.diffusion import TeaCacheController, sea_filter


def reference_sea(x, a, b, axes, power_exp, eps=1e-16):
    """Numpy transcription of the SeaCache reference filter (full fftn, mean norm)."""
    x = x.astype(np.float64)
    gain = None
    for ax in axes:
        f = np.abs(np.fft.fftfreq(x.shape[ax]))
        s = 1.0 / (f**power_exp + eps)
        g = a * s / (a * a * s + b * b + eps)
        shape = [1] * x.ndim
        shape[ax] = g.shape[0]
        g = g.reshape(shape)
        gain = g if gain is None else gain * g
    assert gain is not None
    mean = gain.mean()
    if np.isfinite(mean) and mean > 0:
        gain = gain / mean
    return np.fft.ifftn(np.fft.fftn(x, axes=axes) * gain, axes=axes).real


def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


class TestReferenceParity:
    @pytest.mark.parametrize("shape", [(2, 8, 12, 4), (1, 7, 9, 3), (1, 6, 5, 2)])
    def test_image_power_2(self, shape):
        x = rand(shape)
        out = sea_filter(array_from_any(x), 0.7, 0.3, power_exp=2.0)
        ref = reference_sea(x, 0.7, 0.3, (1, 2), 2.0)
        np.testing.assert_allclose(np.array(out), ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("shape", [(1, 4, 6, 8, 3), (2, 3, 5, 7, 2)])
    def test_video_power_3(self, shape):
        x = rand(shape, seed=1)
        out = sea_filter(array_from_any(x), 0.4, 0.6, power_exp=3.0)
        ref = reference_sea(x, 0.4, 0.6, (1, 2, 3), 3.0)
        np.testing.assert_allclose(np.array(out), ref, atol=1e-4, rtol=1e-4)

    def test_explicit_negative_axes(self):
        x = rand((3, 5, 6, 2), seed=2)  # channels-first-ish: filter the last two axes
        out = sea_filter(array_from_any(x), 0.5, 0.5, axes=(-2, -1))
        ref = reference_sea(x, 0.5, 0.5, (2, 3), 2.0)
        np.testing.assert_allclose(np.array(out), ref, atol=1e-4, rtol=1e-4)


class TestBehaviour:
    def test_shape_and_dtype_preserved(self):
        x = mx.random.normal((1, 4, 6, 8)).astype(mx.bfloat16)
        out = sea_filter(x, 0.6, 0.4)
        assert out.shape == x.shape
        assert out.dtype == mx.bfloat16

    def test_noise_free_is_identity(self):
        # b = 0: the gain is flat (1/a everywhere), so unit-mean normalisation leaves x unchanged.
        x = rand((1, 6, 7, 3), seed=3)
        out = sea_filter(array_from_any(x), 0.8, 0.0)
        np.testing.assert_allclose(np.array(out), x, atol=1e-5)

    def test_zero_signal_scale_gives_zeros(self):
        x = rand((1, 4, 4, 2), seed=4)
        out = np.array(sea_filter(array_from_any(x), 0.0, 1.0))
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_low_pass_at_high_noise(self):
        # With heavy noise the filter keeps low frequencies and damps high ones.
        n = 16
        grid = np.arange(n)
        low = np.cos(2 * np.pi * grid / n)
        high = np.cos(2 * np.pi * 6 * grid / n)
        x = (low + high)[None, :, None, None] * np.ones((1, n, 4, 1))
        out = np.array(sea_filter(array_from_any(x.astype(np.float32)), 0.2, 0.8))[0, :, 0, 0]
        spec = np.abs(np.fft.fft(out))
        assert spec[1] > 2 * spec[6]  # analytic gain ratio ≈ 3.06


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"signal_scale": -0.1, "noise_scale": 0.5},
            {"signal_scale": 0.5, "noise_scale": -0.1},
            {"signal_scale": 0.5, "noise_scale": 0.5, "power_exp": 0.0},
            {"signal_scale": 0.5, "noise_scale": 0.5, "eps": 0.0},
            {"signal_scale": 0.5, "noise_scale": 0.5, "axes": ()},
            {"signal_scale": 0.5, "noise_scale": 0.5, "axes": (1, 1)},
            {"signal_scale": 0.5, "noise_scale": 0.5, "axes": (1, -3)},
            {"signal_scale": 0.5, "noise_scale": 0.5, "axes": (4,)},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            sea_filter(mx.zeros((1, 4, 4, 2)), **kwargs)

    def test_default_axes_need_a_grid(self):
        with pytest.raises(ValueError):
            sea_filter(mx.zeros((4, 2)), 0.5, 0.5)


class TestTeaCacheWithoutCoefficients:
    def test_none_is_identity_rescale(self):
        seq = [mx.full((4,), v) for v in (1.0, 1.03, 1.05, 1.2, 1.21, 1.3)]
        a = TeaCacheController(num_steps=6, rel_l1_thresh=0.05)
        b = TeaCacheController(num_steps=6, rel_l1_thresh=0.05, coefficients=[1.0, 0.0])
        got_a = [a.should_compute(i, x) for i, x in enumerate(seq)]
        got_b = [b.should_compute(i, x) for i, x in enumerate(seq)]
        assert got_a == got_b
        assert got_a == [True, False, False, True, False, True]


def _load_recipe():
    note = Path(__file__).parent.parent / "docs" / "research" / "seacache.md"
    match = re.search(r"<!-- seacache-recipe -->\s*```python\n(.*?)```", note.read_text(), re.S)
    assert match, "SeaCache recipe block not found in the research note"
    namespace: dict[str, Any] = {}
    exec(match.group(1), namespace)
    return namespace["seacache_should_compute"]


def reference_gating(tokens, grid, sigmas, thresh, power_exp):
    """Reference SeaCache loop (filter every step, raw relative L1, TeaCache boundaries)."""
    decisions, prev, acc = [], None, 0.0
    n = len(tokens)
    for i, (t, sigma) in enumerate(zip(tokens, sigmas)):
        sigma = min(max(sigma, 1e-6), 1 - 1e-6)
        x = t.reshape(t.shape[0], *grid, t.shape[-1])
        f = reference_sea(x, 1 - sigma, sigma, tuple(range(1, x.ndim - 1)), power_exp)
        if i == 0 or i == n - 1:
            compute, acc = True, 0.0
        else:
            assert prev is not None
            acc += np.abs(f - prev).mean() / (np.abs(prev).mean() + 1e-16)
            compute = acc >= thresh
            if compute:
                acc = 0.0
        decisions.append(compute)
        prev = f
    return decisions


class TestSeaCacheRecipe:
    @pytest.mark.parametrize(
        ("grid", "power_exp", "thresh"), [((8, 12), 2.0, 0.3), ((3, 6, 8), 3.0, 0.2)]
    )
    def test_decisions_match_reference_loop(self, grid, power_exp, thresh):
        should_compute = _load_recipe()
        rng = np.random.default_rng(7)
        C, steps = 4, 20
        n = int(np.prod(grid))
        x0 = rng.standard_normal((1, n, C))
        noise = rng.standard_normal((1, n, C))
        sigmas = list(np.linspace(1.0, 0.0, steps))
        # Flow-matching trajectory plus a slowly drifting "modulation" term.
        tokens = [
            ((1 - s) * x0 + s * noise + 0.05 * i * x0).astype(np.float32)
            for i, s in enumerate(sigmas)
        ]
        ref = reference_gating(tokens, grid, sigmas, thresh, power_exp)
        controller = TeaCacheController(num_steps=steps, rel_l1_thresh=thresh)
        got = [
            should_compute(controller, i, array_from_any(t), grid, float(s), power_exp=power_exp)
            for i, (t, s) in enumerate(zip(tokens, sigmas))
        ]
        assert got == ref
        assert 0 < got.count(False) < steps - 2  # the gate both skips and computes

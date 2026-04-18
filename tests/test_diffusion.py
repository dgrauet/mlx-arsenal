"""Tests for diffusion primitives."""

import math

import mlx.core as mx
import pytest

from mlx_arsenal.diffusion import (
    FlowMatchEulerDiscreteScheduler,
    TimestepEmbedding,
    classifier_free_guidance,
    dynamic_shift_schedule,
    euler_step,
    get_sampling_sigmas,
    get_timestep_embedding,
)


class TestTimestepEmbedding:
    def test_shape_even(self):
        t = mx.array([0.0, 250.0, 500.0, 999.0])
        out = get_timestep_embedding(t, embedding_dim=64)
        assert out.shape == (4, 64)

    def test_shape_odd_pads(self):
        t = mx.array([1.0])
        out = get_timestep_embedding(t, embedding_dim=7)
        assert out.shape == (1, 7)
        # Last column is the pad (always 0)
        assert out[0, -1].item() == pytest.approx(0.0)

    def test_flip_sin_to_cos(self):
        t = mx.array([10.0])
        a = get_timestep_embedding(t, embedding_dim=16, flip_sin_to_cos=True)
        b = get_timestep_embedding(t, embedding_dim=16, flip_sin_to_cos=False)
        # Swapping the halves reproduces the other ordering.
        swapped = mx.concatenate([b[:, 8:], b[:, :8]], axis=-1)
        assert mx.allclose(a, swapped, atol=1e-5).item()

    def test_zero_timestep_matches_cos_zero_sin_zero(self):
        # At t=0, embedding should be [1,1,...,0,0,...] (cos, sin) with flip.
        t = mx.array([0.0])
        out = get_timestep_embedding(t, embedding_dim=8, flip_sin_to_cos=True)
        # First half = cos(0) = 1, second half = sin(0) = 0
        assert mx.allclose(out[0, :4], mx.ones(4)).item()
        assert mx.allclose(out[0, 4:], mx.zeros(4), atol=1e-6).item()

    def test_mlp_shape(self):
        embed = TimestepEmbedding(in_channels=64, time_embed_dim=256)
        sample = mx.random.normal((2, 64))
        out = embed(sample)
        assert out.shape == (2, 256)


class TestGetSamplingSigmas:
    def test_endpoints_and_length(self):
        sigmas = get_sampling_sigmas(num_steps=8)
        assert len(sigmas) == 9
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas[-1] == pytest.approx(0.0)

    def test_monotonic_descending(self):
        sigmas = get_sampling_sigmas(num_steps=16)
        for a, b in zip(sigmas[:-1], sigmas[1:]):
            assert a >= b

    def test_shift_compresses_high_noise(self):
        unshifted = get_sampling_sigmas(num_steps=8, shift=1.0)
        shifted = get_sampling_sigmas(num_steps=8, shift=3.0)
        # shift>1 keeps interior sigmas closer to 1 (slower denoising at high noise)
        assert shifted[4] > unshifted[4]
        # Endpoints preserved
        assert shifted[0] == pytest.approx(1.0)
        assert shifted[-1] == pytest.approx(0.0)


class TestDynamicShiftSchedule:
    def test_length_and_terminal(self):
        sigmas = dynamic_shift_schedule(num_steps=10, num_tokens=4096)
        assert len(sigmas) == 11
        assert sigmas[-1] == pytest.approx(0.0)

    def test_monotonic(self):
        sigmas = dynamic_shift_schedule(num_steps=12, num_tokens=2048)
        for a, b in zip(sigmas[:-1], sigmas[1:]):
            assert a >= b

    def test_stretch_hits_terminal(self):
        sigmas = dynamic_shift_schedule(num_steps=6, num_tokens=4096, stretch=True, terminal=0.1)
        # After stretching, the last non-zero sigma equals 1 - terminal = 0.9
        non_zero = [s for s in sigmas if s != 0.0]
        assert non_zero[-1] == pytest.approx(0.1, abs=1e-6)

    def test_no_stretch_preserves_endpoints(self):
        sigmas = dynamic_shift_schedule(num_steps=6, num_tokens=4096, stretch=False)
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas[-1] == pytest.approx(0.0)


class TestFlowMatchEulerDiscreteScheduler:
    def test_set_timesteps_shapes(self):
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        sched.set_timesteps(num_inference_steps=25)
        assert sched.sigmas.shape == (26,)
        assert sched.timesteps.shape == (25,)

    def test_step_without_setup_raises(self):
        sched = FlowMatchEulerDiscreteScheduler()
        with pytest.raises(RuntimeError):
            sched.step(mx.zeros((1, 4)), mx.array([0.0]), mx.zeros((1, 4)))

    def test_add_noise_interpolates(self):
        sched = FlowMatchEulerDiscreteScheduler()
        original = mx.ones((2, 4))
        noise = mx.zeros((2, 4))
        # sigma=0 → pure original
        out = sched.add_noise(original, noise, mx.array(0.0))
        assert mx.allclose(out, original).item()
        # sigma=1 → pure noise
        out = sched.add_noise(original, noise, mx.array(1.0))
        assert mx.allclose(out, noise).item()

    def test_step_advances_index_and_moves_sample(self):
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        sched.set_timesteps(num_inference_steps=4)
        sample = mx.zeros((1, 4))
        velocity = mx.ones((1, 4))
        out = sched.step(velocity, sched.timesteps[0], sample)
        assert sched._step_index == 1
        # sigma_next > sigma for ascending schedule, so update is positive
        assert (out > sample).all().item()

    def test_custom_sigmas_accepted(self):
        sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        sched.set_timesteps(num_inference_steps=3, sigmas=[0.1, 0.5, 0.9])
        # 3 sigmas + terminal 1.0 = 4
        assert sched.sigmas.shape == (4,)
        assert sched.sigmas[-1].item() == pytest.approx(1.0)


class TestEulerStep:
    def test_sigma_zero_returns_x0(self):
        x = mx.random.normal((1, 8))
        x0 = mx.random.normal((1, 8))
        out = euler_step(x, x0, sigma=0.0, sigma_next=0.0)
        assert mx.allclose(out, x0).item()

    def test_identity_when_sigma_equals_sigma_next(self):
        x = mx.random.normal((1, 8))
        x0 = mx.random.normal((1, 8))
        out = euler_step(x, x0, sigma=0.5, sigma_next=0.5)
        assert mx.allclose(out, x).item()

    def test_full_denoise_step_reaches_x0(self):
        # One step from sigma to 0 should land exactly on x0 (up to fp).
        x = mx.array([[2.0, 4.0]])
        x0 = mx.array([[1.0, 2.0]])
        out = euler_step(x, x0, sigma=1.0, sigma_next=0.0)
        assert mx.allclose(out, x0, atol=1e-5).item()

    def test_partial_step_is_linear_interpolation(self):
        x = mx.array([[2.0]])
        x0 = mx.array([[0.0]])
        out = euler_step(x, x0, sigma=1.0, sigma_next=0.5)
        # d = (x - x0) / sigma = 2 ; out = x + (0.5 - 1) * 2 = 1
        assert out[0, 0].item() == pytest.approx(1.0, abs=1e-5)


class TestClassifierFreeGuidance:
    def test_scale_one_returns_cond(self):
        cond = mx.random.normal((2, 8))
        uncond = mx.random.normal((2, 8))
        out = classifier_free_guidance(cond, uncond, scale=1.0)
        assert mx.allclose(out, cond, atol=1e-5).item()

    def test_scale_zero_returns_uncond(self):
        cond = mx.random.normal((2, 8))
        uncond = mx.random.normal((2, 8))
        out = classifier_free_guidance(cond, uncond, scale=0.0)
        assert mx.allclose(out, uncond, atol=1e-5).item()

    def test_amplifies_conditioning(self):
        cond = mx.array([[2.0]])
        uncond = mx.array([[0.0]])
        out = classifier_free_guidance(cond, uncond, scale=3.0)
        # 0 + 3 * (2 - 0) = 6
        assert out[0, 0].item() == pytest.approx(6.0)


class TestDiffusionIntegration:
    def test_full_denoise_roundtrip_identity(self):
        """With a perfect x0-predictor (returns x0 directly), the Euler loop
        should converge to x0 regardless of schedule."""
        x0_true = mx.array([[1.0, -1.0, 0.5]])
        sigmas = get_sampling_sigmas(num_steps=4)
        # Start from pure noise interpolation at sigma[0]=1
        x = mx.random.normal(x0_true.shape) * sigmas[0] + x0_true * (1.0 - sigmas[0])
        for s, s_next in zip(sigmas[:-1], sigmas[1:]):
            x = euler_step(x, x0_true, s, s_next)
        assert mx.allclose(x, x0_true, atol=1e-4).item()

    def test_sigmas_pair_count_matches_steps(self):
        num_steps = 7
        sigmas = get_sampling_sigmas(num_steps=num_steps)
        pairs = list(zip(sigmas[:-1], sigmas[1:]))
        assert len(pairs) == num_steps

    def test_timestep_embedding_then_mlp(self):
        t = mx.array([0.0, 500.0, 999.0])
        sin_emb = get_timestep_embedding(t, embedding_dim=32)
        embed = TimestepEmbedding(in_channels=32, time_embed_dim=128)
        out = embed(sin_emb)
        assert out.shape == (3, 128)
        # Non-trivial: different timesteps produce different embeddings
        assert not mx.allclose(out[0], out[1]).item()


def test_no_extraneous_nan_from_shift_endpoints():
    """Endpoints 0.0 must stay finite regardless of shift values."""
    for shift in (1.0, 3.0, 7.0):
        sigmas = get_sampling_sigmas(num_steps=8, shift=shift)
        for s in sigmas:
            assert not math.isnan(s) and math.isfinite(s)

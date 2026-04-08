"""Tests for diffusion module."""

import mlx.core as mx
import pytest

from mlx_ops.diffusion import (
    sinusoidal_embedding,
    TimestepEmbedding,
    ResnetBlock2d,
    ResnetBlock3d,
    FlowMatchEulerScheduler,
)


class TestSinusoidalEmbedding:
    def test_shape(self):
        t = mx.array([0, 1, 2, 3])
        emb = sinusoidal_embedding(t, dim=64)
        mx.eval(emb)
        assert emb.shape == (4, 64)

    def test_different_timesteps(self):
        t = mx.array([0, 100])
        emb = sinusoidal_embedding(t, dim=32)
        mx.eval(emb)
        assert not mx.allclose(emb[0], emb[1])


class TestTimestepEmbedding:
    def test_output_shape(self):
        emb = TimestepEmbedding(dim=128)
        t = mx.array([0.0, 0.5, 1.0])
        out = emb(t)
        mx.eval(out)
        assert out.shape == (3, 128)

    def test_custom_hidden_dim(self):
        emb = TimestepEmbedding(dim=64, hidden_dim=256)
        t = mx.array([0.0, 1.0])
        out = emb(t)
        mx.eval(out)
        assert out.shape == (2, 64)


class TestResnetBlock2d:
    def test_same_channels(self):
        block = ResnetBlock2d(32)
        x = mx.random.normal((1, 8, 8, 32))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 32)

    def test_different_channels(self):
        block = ResnetBlock2d(32, 64)
        x = mx.random.normal((1, 8, 8, 32))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 64)

    def test_with_time_embedding(self):
        block = ResnetBlock2d(32, time_emb_dim=128)
        x = mx.random.normal((1, 8, 8, 32))
        t = mx.random.normal((1, 128))
        out = block(x, time_emb=t)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 32)


class TestResnetBlock3d:
    def test_same_channels(self):
        block = ResnetBlock3d(16)
        x = mx.random.normal((1, 4, 4, 4, 16))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 4, 4, 4, 16)

    def test_different_channels(self):
        block = ResnetBlock3d(16, 32)
        x = mx.random.normal((1, 4, 4, 4, 16))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 4, 4, 4, 32)

    def test_with_time_embedding(self):
        block = ResnetBlock3d(16, time_emb_dim=64)
        x = mx.random.normal((1, 4, 4, 4, 16))
        t = mx.random.normal((1, 64))
        out = block(x, time_emb=t)
        mx.eval(out)
        assert out.shape == (1, 4, 4, 4, 16)


class TestFlowMatchEulerScheduler:
    def test_timesteps(self):
        sched = FlowMatchEulerScheduler(num_inference_steps=10)
        assert sched.timesteps.shape == (10,)

    def test_sigmas_range(self):
        sched = FlowMatchEulerScheduler(num_inference_steps=10)
        sigmas = sched.sigmas
        mx.eval(sigmas)
        assert float(sigmas[0]) == pytest.approx(1.0, abs=0.01)
        assert float(sigmas[-1]) == pytest.approx(0.0, abs=0.01)

    def test_step_denoises(self):
        sched = FlowMatchEulerScheduler(num_inference_steps=50)
        x = mx.random.normal((1, 4, 4, 16))
        velocity = mx.random.normal((1, 4, 4, 16))
        out = sched.step(velocity, 0, x)
        mx.eval(out)
        assert out.shape == x.shape

    def test_shift(self):
        sched_no_shift = FlowMatchEulerScheduler(num_inference_steps=10, shift=1.0)
        sched_shift = FlowMatchEulerScheduler(num_inference_steps=10, shift=3.0)
        mx.eval(sched_no_shift.sigmas, sched_shift.sigmas)
        assert not mx.allclose(sched_no_shift.sigmas, sched_shift.sigmas)

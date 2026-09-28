"""Tests for mlx_arsenal.diffusion.uniform_decode."""

import gc

import mlx.core as mx
import pytest

from mlx_arsenal._typing import item_int
from mlx_arsenal.diffusion import (
    StableConfidentStopping,
    linear_temperature,
    renoise,
    uniform_canvas,
)


class TestLinearTemperature:
    def test_matches_reference_formula(self):
        # HF LinearTemperatureScheduleLogitsProcessor: t_min + (t_max - t_min) * n / N
        for n in range(0, 49):
            assert linear_temperature(n, 48, t_min=0.4, t_max=0.8) == pytest.approx(
                0.4 + 0.4 * n / 48
            )

    def test_endpoints(self):
        assert linear_temperature(0, 10) == pytest.approx(0.4)
        assert linear_temperature(10, 10) == pytest.approx(0.8)

    def test_validation(self):
        with pytest.raises(ValueError, match="num_steps"):
            linear_temperature(0, 0)
        with pytest.raises(ValueError, match="remaining"):
            linear_temperature(11, 10)
        with pytest.raises(ValueError, match="remaining"):
            linear_temperature(-1, 10)
        with pytest.raises(ValueError, match="t_min"):
            linear_temperature(1, 10, t_min=0.0)
        with pytest.raises(ValueError, match="t_min"):
            linear_temperature(1, 10, t_min=0.9, t_max=0.8)


class TestUniformCanvas:
    def test_range_dtype_shape(self):
        c = uniform_canvas((3, 256), 17, key=mx.random.key(0))
        assert c.dtype == mx.int32
        assert c.shape == (3, 256)
        assert item_int(mx.min(c)) >= 0 and item_int(mx.max(c)) < 17

    def test_deterministic_with_key(self):
        a = uniform_canvas((2, 64), 1000, key=mx.random.key(1))
        b = uniform_canvas((2, 64), 1000, key=mx.random.key(1))
        assert mx.array_equal(a, b).item()

    def test_global_rng_matches_randint(self):
        # key=None draws from the global PRNG exactly like the reference's randint.
        mx.random.seed(7)
        ours = uniform_canvas((2, 32), 500)
        mx.random.seed(7)
        ref = mx.random.randint(0, 500, (2, 32))
        assert mx.array_equal(ours, ref).item()

    def test_validation(self):
        with pytest.raises(ValueError, match="vocab_size"):
            uniform_canvas((2, 4), 0)
        with pytest.raises(ValueError, match="shape"):
            uniform_canvas((2, 0), 10)
        with pytest.raises(ValueError, match="shape"):
            uniform_canvas((), 10)


class TestRenoise:
    def test_keeps_accepted_and_replaces_the_rest(self):
        canvas = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        noise = mx.array([[9, 9, 9, 9]], dtype=mx.int32)
        accepted = mx.array([[True, False, True, False]])
        out = renoise(canvas, accepted, noise)
        assert out.tolist() == [[1, 9, 3, 9]]
        assert out.dtype == mx.int32

    def test_validation(self):
        canvas = mx.zeros((1, 4), dtype=mx.int32)
        accepted = mx.ones((1, 4), dtype=mx.bool_)
        with pytest.raises(ValueError, match="shape"):
            renoise(canvas, accepted[:, :2], canvas)
        with pytest.raises(ValueError, match="shape"):
            renoise(canvas, accepted, canvas[:, :2])
        with pytest.raises(ValueError, match="bool"):
            renoise(canvas, accepted.astype(mx.int32), canvas)
        with pytest.raises(ValueError, match="integer"):
            renoise(canvas.astype(mx.float32), accepted, canvas)


class TestStableConfidentStopping:
    LOW = mx.full((2, 4), 0.001)  # confident
    HIGH = mx.full((2, 4), 1.0)  # not confident

    def _canvas(self, a: int, b: int) -> mx.array:
        return mx.array([[a] * 4, [b] * 4], dtype=mx.int32)

    def test_needs_full_history(self):
        stop = StableConfidentStopping(stability_threshold=2, confidence_threshold=0.005)
        c = self._canvas(1, 1)
        assert stop(c, self.LOW).tolist() == [False, False]  # no history
        assert stop(c, self.LOW).tolist() == [False, False]  # only 1 previous
        assert stop(c, self.LOW).tolist() == [True, True]  # 2 identical previous

    def test_per_row(self):
        stop = StableConfidentStopping(stability_threshold=1, confidence_threshold=0.005)
        stop(self._canvas(1, 1), self.LOW)
        assert stop(self._canvas(1, 2), self.LOW).tolist() == [True, False]

    def test_confidence_gate(self):
        stop = StableConfidentStopping(stability_threshold=1, confidence_threshold=0.005)
        stop(self._canvas(1, 1), self.HIGH)
        entropy = mx.array([[0.001] * 4, [1.0] * 4])
        assert stop(self._canvas(1, 1), entropy).tolist() == [True, False]

    def test_zero_threshold_is_always_stable(self):
        stop = StableConfidentStopping(stability_threshold=0, confidence_threshold=0.005)
        assert stop(self._canvas(1, 2), self.LOW).tolist() == [True, True]

    def test_change_resets_stability(self):
        stop = StableConfidentStopping(stability_threshold=1, confidence_threshold=0.005)
        stop(self._canvas(1, 1), self.LOW)
        stop(self._canvas(2, 2), self.LOW)
        assert stop(self._canvas(2, 2), self.LOW).tolist() == [True, True]

    def test_reset(self):
        stop = StableConfidentStopping(stability_threshold=1, confidence_threshold=0.005)
        c = self._canvas(1, 1)
        stop(c, self.LOW)
        stop.reset()
        assert stop(c, self.LOW).tolist() == [False, False]

    def test_history_does_not_pin_logits(self):
        # A lazy argmax kept in the history would keep the (B, L, V) logits alive.
        stop = StableConfidentStopping(stability_threshold=2)
        logits = mx.random.normal((2, 256, 16384), key=mx.random.key(0))
        mx.eval(logits)
        stop(mx.argmax(logits, axis=-1), mx.zeros((2, 256)))
        del logits
        gc.collect()
        mx.clear_cache()
        assert mx.get_active_memory() < 2 * 256 * 16384 * 4 // 4

    def test_validation(self):
        with pytest.raises(ValueError, match="stability_threshold"):
            StableConfidentStopping(stability_threshold=-1)
        with pytest.raises(ValueError, match="confidence_threshold"):
            StableConfidentStopping(confidence_threshold=0.0)
        stop = StableConfidentStopping()
        with pytest.raises(ValueError, match="shape"):
            stop(self._canvas(1, 1), self.LOW[:, :2])
        stop(self._canvas(1, 1), self.LOW)
        with pytest.raises(ValueError, match="shape"):
            stop(mx.zeros((2, 5), dtype=mx.int32), mx.zeros((2, 5)))

"""Tests for mlx_arsenal.diffusion.uniform_decode."""

import gc
import re
from pathlib import Path
from typing import Any

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


def _load_renoise_loop() -> Any:
    note = Path(__file__).parents[1] / "docs" / "research" / "dllm-block-decoding.md"
    match = re.search(r"<!-- renoise-loop -->\s*```python\n(.*?)```", note.read_text(), re.S)
    assert match, "renoise loop block not found in the research note"
    namespace: dict[str, Any] = {}
    exec(match.group(1), namespace)
    return namespace["decode_uniform"]


class TestRenoiseLoop:
    V = 50

    def _model(self, sharpness: float):
        target = mx.arange(8) % self.V  # the "answer" the synthetic model knows

        def model(canvas: mx.array) -> mx.array:
            onehot = mx.equal(mx.expand_dims(target, -1), mx.arange(self.V)).astype(mx.float32)
            noise = mx.sin(canvas.astype(mx.float32))[..., None]  # depends on the canvas
            return mx.broadcast_to(sharpness * onehot, (canvas.shape[0], 8, self.V)) + noise

        return model

    def test_confident_model_stops_early(self):
        decode = _load_renoise_loop()
        mx.random.seed(0)
        out, steps = decode(self._model(40.0), batch=2, canvas_len=8, vocab_size=self.V)
        assert out.tolist() == [list(range(8))] * 2
        assert steps == 2  # stable after one repeat, entropy ~0

    def test_uncertain_model_runs_all_steps(self):
        decode = _load_renoise_loop()
        mx.random.seed(0)
        out, steps = decode(self._model(0.0), batch=1, canvas_len=8, vocab_size=self.V, num_steps=6)
        assert steps == 6
        assert out.shape == (1, 8)

    def test_deterministic_under_a_seed(self):
        decode = _load_renoise_loop()
        runs = []
        for _ in range(2):
            mx.random.seed(3)
            runs.append(
                decode(self._model(2.0), batch=1, canvas_len=8, vocab_size=self.V, num_steps=5)
            )
        assert mx.array_equal(runs[0][0], runs[1][0]).item()
        assert runs[0][1] == runs[1][1]


class TestRenoiseLoopBatch:
    V = 50

    def test_rows_finish_independently_and_stay_frozen(self):
        # Row 0 is confident from step 1 (stops at step 2), then its prediction keeps
        # flipping; row 1 becomes confident at step 3 (stops at step 4). The loop must
        # remember row 0's stop, freeze its step-2 argmax, and end at step 4.
        decode = _load_renoise_loop()
        V, calls = self.V, [0]

        def sharp(token: int) -> mx.array:
            return 40.0 * mx.equal(mx.arange(V), token).astype(mx.float32)

        def model(canvas: mx.array) -> mx.array:
            calls[0] += 1
            n = calls[0]
            row0 = sharp(3) if n <= 2 else sharp(10 + n % 2)
            row1 = mx.zeros((V,)) if n <= 2 else sharp(7)
            rows = mx.stack([row0, row1])[:, None, :]
            return mx.broadcast_to(rows, (2, canvas.shape[1], V))

        mx.random.seed(0)
        out, steps = decode(model, batch=2, canvas_len=4, vocab_size=V)
        assert steps == 4
        assert out.tolist() == [[3] * 4, [7] * 4]

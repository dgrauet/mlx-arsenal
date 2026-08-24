"""Shared state and validation for the step-aware cache controllers.

Four controllers (:class:`TeaCacheController`, :class:`PerLayerAttentionCache`,
:class:`PerHeadAttentionCache`, :class:`WindowResidualController`) share the
same skeleton: a step-index range check, a "previous payload" slot that raises
until seeded, and a scalar relative-L1 tracker over the previous input. This
module hosts the single implementation.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .._typing import item_float


def check_step(step_index: int, num_steps: int, *, what: str = "step_index") -> None:
    """Raise ``ValueError`` unless ``0 <= step_index < num_steps``."""
    if step_index < 0 or step_index >= num_steps:
        raise ValueError(f"{what} must be in [0, {num_steps}), got {step_index}")


def require_cached(value: Any, message: str) -> Any:
    """Return ``value``, raising ``RuntimeError(message)`` when it is ``None``."""
    if value is None:
        raise RuntimeError(message)
    return value


class RelL1State:
    """Previous-input tracker for the scalar relative-L1 recompute metric.

    ``seed`` stores the input at a boundary/degenerate step; ``delta`` returns
    ``mean(|x - prev|) / mean(|prev|)`` and advances the state, or ``None``
    when the previous summary is zero (degenerate — the caller must force a
    recompute). ``delta`` raises ``RuntimeError(seed_message)`` if called
    before the first ``seed``.
    """

    def __init__(self, seed_message: str):
        self._seed_message = seed_message
        self._prev_input: mx.array | None = None
        self._prev_summary: float | None = None

    def reset(self) -> None:
        self._prev_input = None
        self._prev_summary = None

    def seed(self, x: mx.array) -> None:
        self._prev_input = x
        self._prev_summary = item_float(mx.mean(mx.abs(x)))

    def delta(self, x: mx.array) -> float | None:
        prev = self._prev_input
        prev_summary = self._prev_summary
        if prev is None or prev_summary is None:
            raise RuntimeError(self._seed_message)
        if prev_summary == 0.0:
            self.seed(x)
            return None
        d = item_float(mx.mean(mx.abs(x - prev))) / prev_summary
        self.seed(x)
        return d

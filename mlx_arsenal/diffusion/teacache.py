"""TeaCache: timestep-aware residual caching for diffusion transformers.

TeaCache (Liu et al., *"Timestep Embedding Aware Cache"*) accelerates diffusion
inference by reusing a cached residual across timesteps when the modulated
input doesn't move much. The trade-off is governed by ``rel_l1_thresh``:
higher = more skipping = faster but lossier.

Architecture-agnostic mechanism — coefficients are model-specific and live
with each model's port (e.g. inside the LTX-2 / Hunyuan / Flux MLX
implementation). ``mlx-arsenal`` ships only the engine.

References:
    https://github.com/ali-vilab/TeaCache
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np


class TeaCacheController:
    """Stateful controller deciding when to skip a transformer forward.

    Usage per denoising step::

        if controller.should_compute(step_index, modulated_input):
            x_in = x
            x = transformer_blocks(x)
            controller.cache_residual(x - x_in)
        else:
            x = x + controller.previous_residual

    Boundary steps (``step_index == 0`` and ``step_index == num_steps - 1``)
    always compute and reset the accumulator.

    Args:
        num_steps: Total number of denoising steps in the generation.
        rel_l1_thresh: Skip threshold on the accumulated rescaled L1 distance.
        coefficients: Polynomial coefficients in ``numpy.poly1d`` order
            (highest degree first), calibrated to map raw L1 distances to a
            quality budget.
    """

    def __init__(
        self,
        num_steps: int,
        rel_l1_thresh: float,
        coefficients: Sequence[float],
    ):
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.coefficients = list(coefficients)
        self._rescale = np.poly1d(self.coefficients)
        self._prev_modulated_input: mx.array | None = None
        self._accumulated: float = 0.0
        self._prev_residual = None

    def reset(self) -> None:
        """Clear all state. Call at the start of each new generation."""
        self._prev_modulated_input = None
        self._accumulated = 0.0
        self._prev_residual = None

    def should_compute(self, step_index: int, modulated_input: mx.array) -> bool:
        """Decide whether to run the transformer at ``step_index``.

        Side-effects: advances the stored ``previous_modulated_input`` and the
        internal accumulator. Must be called once per step in order.
        """
        if step_index == 0 or step_index == self.num_steps - 1:
            self._accumulated = 0.0
            self._prev_modulated_input = modulated_input
            return True

        delta = (
            mx.mean(mx.abs(modulated_input - self._prev_modulated_input))
            / mx.mean(mx.abs(self._prev_modulated_input))
        ).item()
        self._accumulated += float(self._rescale(delta))
        self._prev_modulated_input = modulated_input

        if self._accumulated < self.rel_l1_thresh:
            return False
        self._accumulated = 0.0
        return True

    def cache_residual(self, residual) -> None:
        """Store the residual from the just-computed step for reuse on skip.

        ``residual`` is whatever the caller wants to retrieve later via
        ``previous_residual``. Single-tensor models pass an ``mx.array``;
        multi-stream models (e.g. LTX-2) pass a tuple or dict. The controller
        does not inspect or copy the value.
        """
        self._prev_residual = residual

    @property
    def previous_residual(self):
        """Last cached payload. Raises before the first ``cache_residual`` call."""
        if self._prev_residual is None:
            raise RuntimeError(
                "No residual cached yet — call cache_residual() after a computed step "
                "before reading previous_residual."
            )
        return self._prev_residual

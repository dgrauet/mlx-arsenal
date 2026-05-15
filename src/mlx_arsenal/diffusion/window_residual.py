"""WA-RS controller (DiTFastAttn Window Attention + Residual Sharing).

Decides when to refresh the cached ``full - window`` attention residual:

* :meth:`WindowResidualController.fixed` — refresh every ``K`` steps.
* :meth:`WindowResidualController.scheduled` — refresh on an explicit list.
* :meth:`WindowResidualController.adaptive` — refresh when the attention
  input has moved more than ``rel_l1_thresh`` since the previous step
  (same relative-L1 metric as :class:`mlx_arsenal.diffusion.TeaCacheController`).

Boundary steps (``0`` and ``num_steps - 1``) always refresh regardless of
mode. The cached residual itself is opaque — the controller does not look
inside ``mx.array`` shapes.

References:
    DiTFastAttn — *Window Attention with Residual Sharing* (WA-RS).
"""

from __future__ import annotations

import mlx.core as mx


class WindowResidualController:
    """Step-aware controller for the WA-RS residual cache.

    Construct via :meth:`fixed`, :meth:`scheduled`, or :meth:`adaptive` —
    the bare ``__init__`` is intentionally not part of the public API.
    """

    def __init__(self, num_steps: int):
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")
        self.num_steps = num_steps
        self._mode: str = ""
        self._refresh_every: int = 0
        self._refresh_set: frozenset[int] = frozenset()
        self._rel_l1_thresh: float = 0.0
        self._prev_input: mx.array | None = None
        self._prev_summary: float | None = None
        self._prev_residual: mx.array | None = None

    @classmethod
    def fixed(cls, num_steps: int, *, refresh_every: int) -> WindowResidualController:
        """Refresh on step 0, ``num_steps - 1``, and every ``refresh_every`` step."""
        if refresh_every <= 0:
            raise ValueError(f"refresh_every must be > 0, got {refresh_every}")
        obj = cls(num_steps)
        obj._mode = "fixed"
        obj._refresh_every = refresh_every
        return obj

    def reset(self) -> None:
        self._prev_input = None
        self._prev_summary = None
        self._prev_residual = None

    def should_refresh(self, step_index: int, attn_input: mx.array | None = None) -> bool:
        """Decide whether to recompute full attention at ``step_index``.

        ``attn_input`` is only consulted in ``adaptive`` mode; ignored
        otherwise. Boundary steps (0 and ``num_steps - 1``) always return
        ``True`` regardless of mode.
        """
        self._check_step(step_index)
        if step_index == 0 or step_index == self.num_steps - 1:
            return True
        if self._mode == "fixed":
            return step_index % self._refresh_every == 0
        raise RuntimeError(f"unhandled mode: {self._mode!r}")

    def cache_residual(self, residual: mx.array) -> None:
        self._prev_residual = residual

    @property
    def previous_residual(self) -> mx.array:
        if self._prev_residual is None:
            raise RuntimeError(
                "No residual cached yet — call cache_residual() after a "
                "refresh step before reading previous_residual."
            )
        return self._prev_residual

    def _check_step(self, step_index: int) -> None:
        if step_index < 0 or step_index >= self.num_steps:
            raise ValueError(f"step_index must be in [0, {self.num_steps}), got {step_index}")

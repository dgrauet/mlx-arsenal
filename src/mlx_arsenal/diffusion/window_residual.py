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

from collections.abc import Sequence

import mlx.core as mx

from ._cache_common import RelL1State, check_step, require_cached


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
        self._state = RelL1State(
            "adaptive mode: should_refresh called at a non-boundary step "
            "before the first boundary seeded the cache."
        )
        self._prev_residual: mx.array | None = None

    @classmethod
    def fixed(cls, num_steps: int, *, refresh_every: int) -> WindowResidualController:
        """Refresh on step 0, ``num_steps - 1``, and every ``refresh_every`` step."""
        obj = cls(num_steps)
        if refresh_every <= 0:
            raise ValueError(f"refresh_every must be > 0, got {refresh_every}")
        obj._mode = "fixed"
        obj._refresh_every = refresh_every
        return obj

    @classmethod
    def scheduled(cls, num_steps: int, *, refresh_steps: Sequence[int]) -> WindowResidualController:
        """Refresh on step 0, ``num_steps - 1``, and every step in ``refresh_steps``."""
        obj = cls(num_steps)
        if not refresh_steps:
            raise ValueError("refresh_steps must be non-empty")
        for s in refresh_steps:
            if s < 0 or s >= num_steps:
                raise ValueError(f"refresh_steps must be in [0, {num_steps}), got {s}")
        obj._mode = "scheduled"
        obj._refresh_set = frozenset(refresh_steps)
        return obj

    @classmethod
    def adaptive(cls, num_steps: int, *, rel_l1_thresh: float) -> WindowResidualController:
        """Refresh when relative-L1 input delta crosses ``rel_l1_thresh``.

        Mirrors :class:`TeaCacheController` semantics: at non-boundary
        step ``i`` with previous input ``p``, refresh iff
        ``mean(|input - p|) / mean(|p|) >= rel_l1_thresh``. A zero-norm
        previous input also forces a refresh.
        """
        obj = cls(num_steps)
        if rel_l1_thresh <= 0:
            raise ValueError(f"rel_l1_thresh must be > 0, got {rel_l1_thresh}")
        obj._mode = "adaptive"
        obj._rel_l1_thresh = rel_l1_thresh
        return obj

    def reset(self) -> None:
        """Clear all state. Call at the start of each new generation."""
        self._state.reset()
        self._prev_residual = None

    def should_refresh(self, step_index: int, attn_input: mx.array | None = None) -> bool:
        """Decide whether to recompute full attention at ``step_index``.

        ``attn_input`` is required in ``adaptive`` mode and ignored
        otherwise. Boundary steps (0 and ``num_steps - 1``) always
        return ``True``.
        """
        check_step(step_index, self.num_steps)
        is_boundary = step_index == 0 or step_index == self.num_steps - 1
        if self._mode == "fixed":
            return is_boundary or step_index % self._refresh_every == 0
        if self._mode == "scheduled":
            return is_boundary or step_index in self._refresh_set
        if self._mode == "adaptive":
            return self._adaptive_decision(is_boundary, attn_input)
        raise RuntimeError(f"unhandled mode: {self._mode!r}")

    def _adaptive_decision(self, is_boundary: bool, attn_input: mx.array | None) -> bool:
        if attn_input is None:
            raise RuntimeError("adaptive mode requires attn_input on every should_refresh call")
        if is_boundary:
            self._state.seed(attn_input)
            return True
        delta = self._state.delta(attn_input)
        return True if delta is None else delta >= self._rel_l1_thresh

    def cache_residual(self, residual: mx.array) -> None:
        """Store the ``full - window`` residual from the just-refreshed step for reuse."""
        self._prev_residual = residual

    @property
    def previous_residual(self) -> mx.array:
        """Last cached residual. Raises before the first ``cache_residual`` call."""
        return require_cached(
            self._prev_residual,
            "No residual cached yet — call cache_residual() after a "
            "refresh step before reading previous_residual.",
        )

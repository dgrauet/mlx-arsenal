"""CAT-Flow: curvature-adaptive step sizes for flow-matching Euler sampling.

CAT-Flow (arXiv 2609.01746) picks each Euler step size from the velocity the
model just returned, with no extra function evaluations: small steps where
the trajectory bends, large ones where it is straight. Two rules:

- **OT** ("over time"): ``dt = λ / ‖(u_k − u_{k−1}) / dt_{k−1}‖₂``, a finite
  difference of the velocity, i.e. the trajectory's acceleration.
- **OV** ("over values"): Adam-like moments of the scaled velocity
  ``(1 − t)·u``, ``dt = λ / sqrt(‖m2 − m1²‖₂)``.

The caller owns the sampling loop; :class:`CurvatureAdaptiveStepper` only
turns velocities into step sizes. See ``docs/research/adaptive-flow-steps.md``.

References:
    https://arxiv.org/abs/2609.01746
"""

from __future__ import annotations

import math
from typing import Literal

import mlx.core as mx

from .._typing import item_float


class CurvatureAdaptiveStepper:
    """Zero-NFE adaptive step-size controller for flow-matching Euler sampling.

    Time follows the paper: ``t`` runs from 0 (noise) to 1 (data) and the
    Euler update is ``x ← x + dt·u``. Both rules are invariant to the sign
    of the velocity, so a diffusers-convention output ``v`` (``σ = 1 − t``,
    ``x ← x + (σ_next − σ)·v``) can be passed as is. Usage::

        stepper = CurvatureAdaptiveStepper(1.75, mode="ov")
        while not stepper.done:
            sigma = 1.0 - stepper.t
            v = model(x, sigma)
            dt = stepper.step(v)
            x = x - dt * v  # σ decreases by dt

    Each step size is clipped to ``[dt_min, min(dt_max, 1 − t)]``; the upper
    bound wins, so the last step lands exactly on ``t = 1``. A zero norm (no
    curvature signal) gives the upper bound.

    Batches share one timeline: norms are taken per sample over all non-batch
    axes and the smallest step wins (at batch 1 this is the paper's rule).

    Args:
        scale: ``λ`` — larger means larger steps and fewer of them. The
            paper finds ``1.5``-``2`` best (≈15 steps on FLUX.1-dev).
        mode: ``"ov"`` (default, the paper's best) or ``"ot"``.
        beta: EMA factor of the OV moments (paper: ``0.3``).
        dt_min: Smallest step (paper: ``0.01``).
        dt_max: Optional largest step. ``None`` (paper) caps at ``1 − t``.
        warmup_steps: Leading steps forced to ``dt_min`` while the state
            still updates; the paper uses 2 (FLUX.1-dev), 3 (SD3.5, Krea)
            or 0 (FLUX.1-schnell) because early velocities are unreliable.
        t_start: Initial time, e.g. for image-to-image starting mid-way.
    """

    def __init__(
        self,
        scale: float,
        *,
        mode: Literal["ot", "ov"] = "ov",
        beta: float = 0.3,
        dt_min: float = 0.01,
        dt_max: float | None = None,
        warmup_steps: int = 0,
        t_start: float = 0.0,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if mode not in ("ot", "ov"):
            raise ValueError(f"mode must be 'ot' or 'ov', got {mode!r}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if not 0.0 < dt_min <= 1.0:
            raise ValueError(f"dt_min must be in (0, 1], got {dt_min}")
        if dt_max is not None and dt_max < dt_min:
            raise ValueError(f"dt_max must be >= dt_min ({dt_min}), got {dt_max}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if not 0.0 <= t_start < 1.0:
            raise ValueError(f"t_start must be in [0, 1), got {t_start}")
        self.scale = scale
        self.mode = mode
        self.beta = beta
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.warmup_steps = warmup_steps
        self.t_start = t_start
        self.reset()

    def reset(self) -> None:
        """Return to ``t_start`` and drop the velocity history."""
        self._t = self.t_start
        self._steps = 0
        self._dt_prev = 0.0
        self._u_prev: mx.array | None = None
        self._m1: mx.array | None = None
        self._m2: mx.array | None = None

    @property
    def t(self) -> float:
        """Current time: 0 = noise, 1 = data (diffusers ``σ = 1 − t``)."""
        return self._t

    @property
    def steps(self) -> int:
        """Number of :meth:`step` calls since the last reset."""
        return self._steps

    @property
    def done(self) -> bool:
        """``True`` once ``t`` has reached 1."""
        return self._t >= 1.0

    def step(self, velocity: mx.array) -> float:
        """Step size to take now from ``velocity`` (evaluated at :attr:`t`).

        Advances :attr:`t` by the returned value. ``velocity`` is
        ``(B, ...)``; its shape must not change during a trajectory. A NaN
        velocity raises ``ValueError`` (OT: from the step after it).
        """
        if self.done:
            raise RuntimeError("the trajectory is complete (t == 1); call reset() first")
        if velocity.ndim < 2:
            raise ValueError(
                f"velocity must be (B, ...) with a leading batch axis, got shape {velocity.shape}"
            )
        u = velocity.astype(mx.float32).reshape(velocity.shape[0], -1)
        prev = self._u_prev if self._u_prev is not None else self._m1
        if prev is not None and prev.shape != u.shape:
            raise ValueError(
                f"velocity shape changed mid-trajectory: {tuple(prev.shape)} -> {tuple(u.shape)}"
            )

        if self.mode == "ot":
            dt = self._ot(u)
        else:
            dt = self._ov(u)
        if self._steps < self.warmup_steps:
            dt = self.dt_min

        remaining = 1.0 - self._t
        upper = remaining if self.dt_max is None else min(self.dt_max, remaining)
        dt = min(max(dt, self.dt_min), upper)
        self._t = 1.0 if dt >= remaining else self._t + dt
        self._dt_prev = dt
        self._steps += 1
        return dt

    def _ot(self, u: mx.array) -> float:
        u_prev, self._u_prev = self._u_prev, u
        if u_prev is None:
            return self.dt_min
        accel = (u - u_prev) / self._dt_prev
        return self._scaled_inverse(mx.sqrt(mx.sum(accel * accel, axis=-1)))

    def _ov(self, u: mx.array) -> float:
        b = self.beta
        scaled = (1.0 - self._t) * u
        m1 = self._m1 if self._m1 is not None else mx.zeros_like(u)
        m2 = self._m2 if self._m2 is not None else mx.zeros_like(u)
        self._m2 = b * m2 + (1.0 - b) * scaled * scaled
        self._m1 = b * m1 + (1.0 - b) * scaled
        var = self._m2 - self._m1 * self._m1
        dt = self._scaled_inverse(mx.sqrt(mx.sqrt(mx.sum(var * var, axis=-1))))
        if self._steps == 0:
            dt *= math.sqrt(1.0 - b)  # bias correction of the zero-initialised moments
        return dt

    def _scaled_inverse(self, norms: mx.array) -> float:
        """``scale / norm`` for the largest per-sample norm (the smallest step)."""
        largest = item_float(mx.max(norms))
        if math.isnan(largest):
            raise ValueError("velocity contains NaN")
        return math.inf if largest == 0.0 else self.scale / largest

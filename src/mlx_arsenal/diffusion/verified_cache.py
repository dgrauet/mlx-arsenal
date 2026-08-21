"""Verified feature caching (SpeCa-style) for iterative generative models.

Forecast-then-verify cache: at each iterative step (denoising timestep,
autoregressive step, etc.) the controller can extrapolate a tracked feature
from previously observed anchors via Lagrange polynomial extrapolation,
then accept or reject the prediction by comparing it to a freshly computed
ground-truth feature (typically the output of a single inner layer — the
"verification layer").

The pattern is *lossy but bounded* — see the SpeCa convergence analysis
(Zou et al., 2025, arxiv:2509.11628). For bit-exact reproducibility, do
not use.

Architecture-agnostic mechanism. The caller decides:

- Which feature to track (the verification-layer output is a common pick).
- How to map the controller's abstract ``step_index`` to its own iterative
  axis (denoising step, diffusion timestep, etc.).
- The Lagrange ``order`` (number of anchors − 1) and the threshold schedule
  parameters ``tau_0`` and ``beta``.

References:
    SpeCa — https://arxiv.org/abs/2509.11628
    TaylorSeer — https://arxiv.org/abs/2503.06923
"""

from __future__ import annotations

from collections import deque

import mlx.core as mx

from .._typing import item_float


def geometric_threshold(
    step_index: int,
    num_steps: int,
    tau_0: float,
    beta: float,
) -> float:
    """SpeCa-style geometric threshold schedule.

    ``τ_t = τ₀ · β^((T - 1 - t) / max(T - 1, 1))``

    With ``beta < 1``: threshold grows from ``τ₀·β`` (strict early) to
    ``τ₀`` (tolerant late). With ``beta > 1``: the opposite. ``beta == 1``
    yields a constant threshold ``τ₀``. The "right" regime is empirical
    and depends on the schedule and model — see the research note at
    ``docs/research/verified-feature-caching.md``.
    """
    if num_steps <= 1:
        return tau_0
    exponent = (num_steps - 1 - step_index) / (num_steps - 1)
    return tau_0 * (beta**exponent)


class VerifiedFeatureCache:
    """Lagrange-extrapolate a tracked feature and verify before accepting.

    Usage per step::

        cache = VerifiedFeatureCache(num_steps=50, tau_0=0.1, beta=0.5)

        for step in range(num_steps):
            if cache.can_predict(step):
                predicted = cache.extrapolate(step)
                actual = run_verifier_layer(state)  # caller-supplied
                if cache.accept(step, predicted, actual):
                    # use predicted for downstream layers / skip full forward
                    cache.record(step, predicted)
                    continue
            feature = run_full_forward(state)  # fallback
            cache.record(step, feature)

    Boundary policy: ``can_predict`` returns ``False`` at ``step == 0`` and
    ``step == num_steps - 1``, forcing a full compute at both ends. This
    matches the convention used by other step-aware controllers in
    ``mlx_arsenal.diffusion``.

    Args:
        num_steps: Total number of iterative steps in the generation.
        tau_0: Base relative-L2² threshold for accepting a draft.
        beta: Geometric schedule base. ``beta < 1`` → strict early /
            tolerant late (SpeCa default regime).
        order: Lagrange polynomial order. Requires ``order + 1`` anchors
            before predictions become available. Typical values 1-3.
        epsilon: Numerical floor in the relative-L2 denominator.
    """

    def __init__(
        self,
        num_steps: int,
        tau_0: float,
        beta: float,
        *,
        order: int = 2,
        epsilon: float = 1e-8,
    ) -> None:
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")
        if tau_0 <= 0.0:
            raise ValueError(f"tau_0 must be > 0, got {tau_0}")
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")

        self.num_steps = num_steps
        self.tau_0 = tau_0
        self.beta = beta
        self.order = order
        self.epsilon = epsilon

        self._anchor_steps: deque[int] = deque(maxlen=order + 1)
        self._anchor_values: deque[mx.array] = deque(maxlen=order + 1)

    def reset(self) -> None:
        """Drop all anchors. Call at the start of each new generation."""
        self._anchor_steps.clear()
        self._anchor_values.clear()

    def threshold(self, step_index: int) -> float:
        """Adaptive threshold at ``step_index`` (relative-L2² units)."""
        return geometric_threshold(step_index, self.num_steps, self.tau_0, self.beta)

    def can_predict(self, step_index: int) -> bool:
        """``True`` if a draft is available for ``step_index``.

        Returns ``False`` at boundary steps (0 and ``num_steps - 1``) and
        whenever fewer than ``order + 1`` anchors have been recorded.
        """
        if step_index <= 0 or step_index >= self.num_steps - 1:
            return False
        if len(self._anchor_values) < self.order + 1:
            return False
        return step_index > self._anchor_steps[-1]

    def extrapolate(self, step_index: int) -> mx.array:
        """Lagrange-extrapolate the tracked feature at ``step_index``.

        Raises ``RuntimeError`` if called when ``can_predict`` would
        return ``False`` — the caller is expected to gate on it first.
        """
        if not self.can_predict(step_index):
            raise RuntimeError(
                "extrapolate called when can_predict is False — boundary step, "
                "insufficient anchors, or non-monotonic step_index."
            )

        steps = list(self._anchor_steps)
        values = list(self._anchor_values)
        result = mx.zeros_like(values[0])
        for i, s_i in enumerate(steps):
            num = 1.0
            den = 1.0
            for j, s_j in enumerate(steps):
                if i == j:
                    continue
                num *= step_index - s_j
                den *= s_i - s_j
            result = result + values[i] * (num / den)
        return result

    def accept(self, step_index: int, predicted: mx.array, actual: mx.array) -> bool:
        """Compare draft to ground truth on the verification layer.

        Uses squared relative-L2 to match the SpeCa formulation:
        ``e = ‖predicted − actual‖²₂ / (‖actual‖²₂ + ε)``.
        Returns ``True`` if ``e <= threshold(step_index)``.
        """
        diff_sq = item_float(mx.sum((predicted - actual) ** 2))
        actual_sq = item_float(mx.sum(actual**2))
        error = diff_sq / (actual_sq + self.epsilon)
        return error <= self.threshold(step_index)

    def record(self, step_index: int, feature: mx.array) -> None:
        """Append ``feature`` as a new anchor at ``step_index``.

        Anchors are stored in a fixed-capacity FIFO of size ``order + 1``.
        ``step_index`` must be strictly greater than the last recorded
        step (anchors are monotonic by construction).
        """
        if self._anchor_steps and step_index <= self._anchor_steps[-1]:
            raise ValueError(
                f"step_index {step_index} is not strictly greater than the last "
                f"recorded anchor at step {self._anchor_steps[-1]}."
            )
        self._anchor_steps.append(step_index)
        self._anchor_values.append(feature)

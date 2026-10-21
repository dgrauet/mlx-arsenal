"""SoftCache: adaptive step size from the unspent TeaCache / SeaCache budget.

SoftCache (Malnick et al., *"Adaptive Step Size Caching for Diffusion
Transformers"*, arXiv 2610.22369) keeps the TeaCache-style accumulator of
relative-L1 probe changes, but turns its unspent budget into a longer step:
when the accumulator ``A`` is well under the threshold ``tau``, the sampler
jumps ``1 + (J - 1)(1 - A / tau)`` scheduled steps at once and re-spaces
the rest of the schedule with fewer nodes. Reused steps extrapolate the
transformer residual along the secant of the last two computed anchors
("Residual Motion") instead of reusing it verbatim. No per-model polynomial
fit is needed when the probe is a :func:`~mlx_arsenal.diffusion.sea_filter`
output.

The caller keeps the sampling loop and the (mutable) sigma schedule::

    ctl = SoftCacheController(tau=0.3)
    i = 0
    while sigmas[i] > 0:
        probe = sea_filter(modulated, ...) if i > 0 else modulated
        d = ctl.decide(i, sigmas[i], sigmas[i + 1], probe)
        if d.action == "compute":
            r = blocks(h) - h
            ctl.record(r, sigmas[i])
        else:
            r = ctl.residual(sigmas[i])
        v = head(h + r)
        if d.action == "jump":
            sigmas = regrid_tail(sigmas, i, d.sigma_next)
        x = x + (d.sigma_next - sigmas[i]) * v
        i += 1

Ported from the authors' model-independent ``softcache/method.py`` (MIT,
https://github.com/ShimonMalnick/softcache, commit 49f00ab).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import mlx.core as mx

from ._cache_common import RelL1State, require_cached


class SoftCacheDecision(NamedTuple):
    """Outcome of :meth:`SoftCacheController.decide`.

    Attributes:
        action: ``"compute"`` (run the transformer, then
            :meth:`~SoftCacheController.record`), ``"reuse"`` (take the
            scheduled step with :meth:`~SoftCacheController.residual`) or
            ``"jump"`` (same as reuse, but step to ``sigma_next`` and
            re-space the schedule with :func:`regrid_tail`).
        sigma_next: Noise level to step to. Equals the scheduled next sigma
            unless ``action == "jump"``.
        jump_factor: Stride in scheduled steps (``1.0`` unless jumping).
    """

    action: Literal["compute", "reuse", "jump"]
    sigma_next: float
    jump_factor: float


class SoftCacheController:
    """Small-state controller for SoftCache (compute / reuse / jump per node).

    Sigmas follow the flow-matching convention (1 = noise, 0 = data).

    Args:
        tau: Refresh threshold on the accumulated relative-L1 probe change
            since the last computed node (the TeaCache / SeaCache
            ``rel_l1_thresh``). The paper sweeps 0.3–1.1 on FLUX.
        max_jump_factor: Cap ``J`` on the jump stride, reached with a full
            budget. The paper uses 8.
        beta: Residual Motion step. The paper uses 1.0 up to about 4x
            speedup and 0.5 above.
        jump_sigma_max: No jump while ``sigma`` is above this (the early,
            volatile part of the schedule).
        jump_probe_max: No jump when the probe change at this node exceeds
            this.
    """

    def __init__(
        self,
        tau: float,
        *,
        max_jump_factor: float = 8.0,
        beta: float = 1.0,
        jump_sigma_max: float = 0.9,
        jump_probe_max: float = 0.18,
    ):
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if max_jump_factor < 1:
            raise ValueError(f"max_jump_factor must be >= 1, got {max_jump_factor}")
        self.tau = tau
        self.max_jump_factor = max_jump_factor
        self.beta = beta
        self.jump_sigma_max = jump_sigma_max
        self.jump_probe_max = jump_probe_max
        self._probe = RelL1State(
            "decide called for step_index > 0 before step 0 — the first node must run first."
        )
        self.reset()

    def reset(self) -> None:
        """Clear all state. Call at the start of each new generation."""
        self._probe.reset()
        self._accumulated = 0.0
        self._anchor: mx.array | None = None
        self._anchor_sigma = 0.0
        self._prev: mx.array | None = None
        self._prev_sigma = 0.0

    @property
    def headroom(self) -> float:
        """Unspent fraction of the budget, ``max(0, 1 - accumulated / tau)``."""
        return max(0.0, 1.0 - self._accumulated / self.tau)

    def decide(
        self, step_index: int, sigma: float, sigma_next: float, probe: mx.array
    ) -> SoftCacheDecision:
        """Decide what to do at the node ``sigma`` (scheduled next: ``sigma_next``).

        ``probe`` is the staleness signal (typically the SeaCache-filtered,
        timestep-modulated input of the first block; the authors pass the
        unfiltered input at ``step_index == 0``, where the filter output is
        about zero). Call once per node, in order.
        """
        if step_index == 0:
            self._probe.seed(probe)
            return self._compute(sigma_next)
        raw = self._probe.delta(probe)
        if raw is None:
            # Degenerate (all-zero) previous probe: the change is undefined.
            return self._compute(sigma_next)
        self._accumulated += raw
        if self._accumulated >= self.tau:
            return self._compute(sigma_next)
        if sigma <= self.jump_sigma_max and raw <= self.jump_probe_max:
            factor = 1.0 + (self.max_jump_factor - 1.0) * self.headroom
            if factor > 1.01:  # a stride under 1.01 skips no scheduled node
                target = max(0.0, sigma + factor * (sigma_next - sigma))
                return SoftCacheDecision("jump", target, factor)
        return SoftCacheDecision("reuse", sigma_next, 1.0)

    def _compute(self, sigma_next: float) -> SoftCacheDecision:
        self._accumulated = 0.0
        return SoftCacheDecision("compute", sigma_next, 1.0)

    def record(self, residual: mx.array, sigma: float) -> None:
        """Store the residual (block-stack output minus input) of a computed node."""
        if self._anchor is not None:
            self._prev, self._prev_sigma = self._anchor, self._anchor_sigma
        self._anchor, self._anchor_sigma = residual, sigma

    def residual(self, sigma: float) -> mx.array:
        """Residual to use at a reused / jump node ``sigma`` (Residual Motion).

        ``r_a + beta * lam * (r_a - r_{a-1})`` with
        ``lam = (sigma - sigma_a) / (sigma_a - sigma_{a-1})``; the last anchor
        as is while only one exists.
        """
        anchor = require_cached(
            self._anchor, "No residual recorded yet — call record() after a computed node."
        )
        if self._prev is None:
            return anchor
        # The 1e-8 guard and the (beta * lam) grouping follow the reference code.
        lam = (sigma - self._anchor_sigma) / (self._anchor_sigma - self._prev_sigma - 1e-8)
        return anchor + (self.beta * lam) * (anchor - self._prev)


def regrid_tail(sigmas: Sequence[float], index: int, target: float) -> list[float]:
    """Re-space the schedule after jumping from node ``index`` to ``target``.

    The new tail goes linearly from ``target`` to 0 with as many nodes as
    the old schedule had below ``target`` (plus the terminal 0), never more
    than the old tail. The removed nodes are the compute the jump saves.
    Note the tail is linear in sigma even if the original schedule was
    shifted, as in the reference code.

    Args:
        sigmas: Current schedule, decreasing, ending at 0.
        index: Node the jump starts from.
        target: Noise level the jump lands on.

    Returns:
        ``sigmas[: index + 1]`` followed by the new tail.
    """
    head = list(sigmas[: index + 1])
    n_tail = len(sigmas) - (index + 1)
    n_below = sum(1 for s in sigmas if 0.0 < s < target) + 1
    n_new = min(n_tail - 1, n_below)
    if n_new <= 1:
        return head + ([target, 0.0] if target > 0 else [0.0])
    tail = [target * (1.0 - k / (n_new - 1)) for k in range(n_new)]
    tail[0], tail[-1] = target, 0.0
    return head + tail

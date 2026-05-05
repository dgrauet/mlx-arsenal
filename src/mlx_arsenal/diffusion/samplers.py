"""Stateless samplers and guidance primitives for diffusion denoising."""

from __future__ import annotations

import mlx.core as mx


def euler_step(x: mx.array, x0: mx.array, sigma: float, sigma_next: float) -> mx.array:
    """Single stateless Euler step on an ``x0``-prediction model.

    Implements ``x_{t-1} = x + (sigma_next - sigma) * (x - x0) / sigma``.
    When ``sigma == 0`` the current sample is already clean, so ``x0`` is
    returned directly.

    Args:
        x: Current noisy sample.
        x0: Predicted clean sample.
        sigma: Current noise level.
        sigma_next: Next noise level.

    Returns:
        Updated sample at ``sigma_next``.
    """
    if sigma == 0:
        return x0
    d = (x - x0) / sigma
    return x + (sigma_next - sigma) * d


def classifier_free_guidance(
    cond: mx.array,
    uncond: mx.array,
    scale: float,
) -> mx.array:
    """Apply classifier-free guidance.

    Returns ``uncond + scale * (cond - uncond)``. A ``scale`` of ``1.0``
    yields the conditioned prediction, ``0.0`` the unconditioned one, and
    values greater than ``1.0`` amplify the conditioning signal.

    Args:
        cond: Conditioned prediction.
        uncond: Unconditioned prediction (same shape as ``cond``).
        scale: Guidance scale.

    Returns:
        Guided prediction.
    """
    return uncond + scale * (cond - uncond)

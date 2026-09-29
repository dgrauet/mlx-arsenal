"""PMC-CFG: posterior-mean-capped classifier-free guidance for flow matching.

Plain CFG extrapolates ``v_u + λ(v_c − v_u)``. With large scales the implied
clean-sample estimate (the posterior mean) overshoots, which shows up as
saturation. PMC-CFG (arXiv 2609.24287) keeps the guidance direction but picks,
per sample and per step, the largest increment ``β ∈ [0, λ − 1]`` for which the
guided posterior mean stays within ``Γ`` times the norm of the conditional
one. The cap releases by itself late in sampling, when the conditional and
unconditional estimates agree. No extra model evaluations.

References:
    https://arxiv.org/abs/2609.24287
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .._typing import array_from_any


def posterior_mean_capped_guidance(
    cond: mx.array,
    uncond: mx.array,
    scale: float,
    *,
    x: mx.array,
    sigma: float | mx.array | Any,
    cap: float,
) -> mx.array:
    """Classifier-free guidance with PMC-CFG's per-sample posterior-mean cap.

    diffusers convention: ``σ = 1`` is noise, the velocity is
    ``v = noise − x0`` and the posterior mean (the ``x0`` estimate) is
    ``m = x − σ·v``. With ``Δ = m_c − m_u`` the guidance increment is::

        β = max{β ∈ [0, λ − 1] : ‖m_c + β·Δ‖ ≤ Γ·‖m_c‖}

    solved in closed form per sample (norms over all non-batch axes), and the
    result is ``cond + β·(cond − uncond)``. When the cap does not bind this is
    exactly ``classifier_free_guidance(cond, uncond, scale)``; ``σ = 0`` (no
    gap) gives nominal guidance.

    For the paper's convention (``t = 1`` is data, ``u = x0 − noise``) pass
    ``-u`` velocities and ``σ = 1 − t``, and negate the result.

    Args:
        cond: Conditional velocity ``(B, ...)``.
        uncond: Unconditional velocity, same shape.
        scale: Nominal guidance scale ``λ >= 1`` (as in
            ``classifier_free_guidance``).
        x: Current sample, same shape.
        sigma: Current noise level in ``[0, 1]``: a float, a 0-d array
            (e.g. ``scheduler.sigmas[i]``) or a ``(B,)`` array.
        cap: ``Γ >= 1``. The paper uses 1.05–1.10 (ImageNet, toy GMM);
            ``Γ < 1`` would make even the unguided ``m_c`` infeasible.

    Returns:
        The guided velocity, dtype of ``cond``.
    """
    if cond.ndim < 2:
        raise ValueError(f"inputs must be (B, ...) with a leading batch axis, got {cond.shape}")
    if uncond.shape != cond.shape or x.shape != cond.shape:
        raise ValueError(
            f"cond, uncond and x must share a shape, got {cond.shape}, {uncond.shape}, {x.shape}"
        )
    if not scale >= 1.0:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if not cap >= 1.0:
        raise ValueError(f"cap must be >= 1, got {cap}")
    B = cond.shape[0]
    s = array_from_any(sigma, dtype=mx.float32)
    if s.ndim == 1 and s.shape[0] == B:
        s = s.reshape(B, 1)
    elif s.ndim != 0:
        raise ValueError(f"sigma must be a scalar or have shape ({B},), got {s.shape}")
    if mx.any(mx.logical_or(s < 0.0, s > 1.0)).item():
        raise ValueError("sigma must be in [0, 1]")

    vc = cond.astype(mx.float32).reshape(B, -1)
    vu = uncond.astype(mx.float32).reshape(B, -1)
    mc = x.astype(mx.float32).reshape(B, -1) - s * vc
    gap = s * (vu - vc)  # m_c - m_u
    a = mx.sum(gap * gap, axis=-1)
    b = mx.sum(mc * gap, axis=-1)
    q = mx.sum(mc * mc, axis=-1)
    # Γ >= 1 and Cauchy-Schwarz (b² <= a·q) give disc >= Γ²·b² >= 0 and a
    # non-negative root; the clip only guards float rounding.
    disc = b * b + (cap * cap - 1.0) * a * q
    nominal = scale - 1.0
    safe_a = mx.where(a > 0, a, 1.0)
    beta_cap = mx.clip((-b + mx.sqrt(mx.maximum(disc, 0.0))) / safe_a, 0.0, nominal)
    beta = mx.where(a > 0, beta_cap, nominal)
    guided = vc + beta[:, None] * (vc - vu)
    return guided.reshape(cond.shape).astype(cond.dtype)

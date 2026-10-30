"""Guidance rules: PMC-CFG, TrajCFG token-wise scales and normalized attention guidance (NAG).

PMC-CFG: posterior-mean-capped classifier-free guidance for flow matching.

Plain CFG extrapolates ``v_u + λ(v_c − v_u)``. With large scales the implied
clean-sample estimate (the posterior mean) overshoots, which shows up as
saturation. PMC-CFG (arXiv 2609.24287) keeps the guidance direction but picks,
per sample and per step, the largest increment ``β ∈ [0, λ − 1]`` for which the
guided posterior mean stays within ``Γ`` times the norm of the conditional
one. The cap releases by itself late in sampling, when the conditional and
unconditional estimates agree. No extra model evaluations.

TrajCFG: token-wise CFG scales from the guided-velocity history. Tokens
whose guided velocity keeps its direction from step to step get a weaker
scale, tokens whose direction turns get a stronger one
(:func:`trajectory_guidance_scale`). No extra model evaluations.

References:
    https://arxiv.org/abs/2609.24287
    https://arxiv.org/abs/2610.14153
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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


def _stable_cosine(a: mx.array, b: mx.array, eps: float) -> mx.array:
    """Per-token cosine over the last axis, norms floored at ``eps``, clipped to [-1, 1]."""
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    dot = mx.sum(a32 * b32, axis=-1, keepdims=True)
    na = mx.maximum(mx.sqrt(mx.sum(a32 * a32, axis=-1, keepdims=True)), eps)
    nb = mx.maximum(mx.sqrt(mx.sum(b32 * b32, axis=-1, keepdims=True)), eps)
    return mx.clip(dot / (na * nb), -1.0, 1.0)


def trajectory_guidance_scale(
    history: Sequence[mx.array],
    sigmas: Sequence[float],
    *,
    min_scale: float,
    max_scale: float,
    rho_lo: float = 0.95,
    long_weight: float = 0.2,
    eps: float = 1e-6,
) -> mx.array:
    """Token-wise CFG scale from the directions of past guided velocities (TrajCFG).

    TrajCFG (arXiv 2610.14153) gives each token its own guidance scale. The
    signal is the cosine ``q`` between the guided velocities a token received
    at the two previous steps, mapped linearly to a scale::

        g(q) = min_scale + (max_scale - min_scale) * clip((1 - q) / (1 - rho_lo), 0, 1)

    so a token whose direction did not change gets ``min_scale`` and one whose
    cosine fell to ``rho_lo`` or below gets ``max_scale``. With three past
    velocities, the stride-2 cosine (oldest vs latest) also votes: its angle
    is first shrunk to the adjacent interval,
    ``cos(arccos(q2) * |σ[-1] - σ[-2]| / |σ[-1] - σ[-3]|)``, and the two
    scales are blended ``(1 - long_weight) * g(q1) + long_weight * g(q2')``.

    The caller keeps the history (the last two or three guided velocities
    actually passed to the solver, oldest first, with their sigmas), uses
    its base scale for the first two steps, and guides with
    ``uncond + scale * (cond - uncond)``. The paper adapts every later step.

    A token is a vector along the **last** axis: a latent pixel for a
    channels-last ``(B, H, W, C)`` latent, a patch for a packed
    ``(B, L, C·p²)`` one. The paper uses the latent sites the backbone's
    pipeline exposes and does not compare the two.

    Args:
        history: Two or three past guided velocities, oldest first, same
            shape ``(..., C)``.
        sigmas: Their noise levels (or any time variable affine in them),
            same length, strictly monotonic.
        min_scale: Scale for a token whose direction did not change, ``>= 0``.
        max_scale: Scale at or below ``rho_lo``, ``> min_scale``.
        rho_lo: Cosine at which the scale saturates, in ``[-1, 1)``. The
            paper uses 0.95 (0.90 for some video models).
        long_weight: Weight of the stride-2 term when three velocities are
            given, in ``[0, 1]``. Paper: 0.2.
        eps: Floor on velocity norms in the cosine, ``> 0``.

    Returns:
        ``(..., 1)`` float32 scales, broadcasting against the velocity.

    References:
        https://arxiv.org/abs/2610.14153 (reimplemented from the paper; the
        reference code has no licence)
    """
    if len(history) not in (2, 3):
        raise ValueError(f"history must hold 2 or 3 velocities, got {len(history)}")
    if len(sigmas) != len(history):
        raise ValueError(f"sigmas must match history: {len(sigmas)} vs {len(history)}")
    shapes = {tuple(v.shape) for v in history}
    if len(shapes) != 1:
        raise ValueError(f"history velocities must share one shape, got {sorted(shapes)}")
    if history[0].ndim < 1:
        raise ValueError("history velocities need a channel axis")
    if not 0.0 <= min_scale < max_scale or not math.isfinite(max_scale):
        raise ValueError(f"need 0 <= min_scale < max_scale < inf, got {min_scale} and {max_scale}")
    if not -1.0 <= rho_lo < 1.0:
        raise ValueError(f"rho_lo must be in [-1, 1), got {rho_lo}")
    if not 0.0 <= long_weight <= 1.0:
        raise ValueError(f"long_weight must be in [0, 1], got {long_weight}")
    if not eps > 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if not all(math.isfinite(s) for s in sigmas):
        raise ValueError(f"sigmas must be finite, got {list(sigmas)}")
    steps = [b - a for a, b in zip(sigmas[:-1], sigmas[1:], strict=True)]
    if not (all(d < 0 for d in steps) or all(d > 0 for d in steps)):
        raise ValueError(f"sigmas must be strictly monotonic, got {list(sigmas)}")
    short = abs(steps[-1])

    def g(q: mx.array) -> mx.array:
        ramp = mx.clip((1.0 - q) / (1.0 - rho_lo), 0.0, 1.0)
        return min_scale + (max_scale - min_scale) * ramp

    scale = g(_stable_cosine(history[-2], history[-1], eps))
    if len(history) == 3:
        ratio = short / abs(sigmas[-1] - sigmas[-3])
        q_long = _stable_cosine(history[-3], history[-1], eps)
        q_norm = mx.cos(ratio * mx.arccos(q_long))
        scale = (1.0 - long_weight) * scale + long_weight * g(q_norm)
    return scale


def normalized_attention_guidance(
    positive: mx.array,
    negative: mx.array,
    *,
    scale: float,
    tau: float = 2.5,
    alpha: float = 0.5,
    axis: int | tuple[int, ...] = -1,
) -> mx.array:
    """Normalized attention guidance (NAG) on attention outputs.

    NAG (Chen et al., arXiv 2505.21179) guides in attention-feature space
    instead of on the model output, which works for few-step and distilled
    models where CFG fails. The same queries attend to the positive and the
    negative context. The two outputs are extrapolated, the L1 norm of the
    result is clipped per token to ``tau`` times that of the positive output,
    and the result is blended back with the positive output::

        z = scale * positive - (scale - 1) * negative
        z = z * min(|z|_1 / |positive|_1, tau) / (|z|_1 / |positive|_1)
        out = alpha * z + (1 - alpha) * positive

    Apply it to cross-attention outputs before the output projection; the
    extra cost is one attention call against the negative context. Computed
    in float32; a token whose positive output is all zero gets a zero
    guidance instead of NaN.

    Args:
        positive: Attention output against the positive context.
        negative: Attention output against the negative context, same shape.
        scale: Extrapolation factor (``>= 1``; ``1`` returns ``positive``).
            Typical values: 5 for Flux (reference README), 11 for distilled
            LTX (ltx-2-mlx).
        tau: Clip on the per-token L1-norm ratio (``> 0``).
        alpha: Blend between the clipped guidance and ``positive``, in
            ``[0, 1]``.
        axis: Feature axes of the L1 norm. ``-1`` for a ``(B, N, H*D)``
            output; ``(1, 3)`` for ``(B, H, N, D)`` (the norm spans all heads,
            as in the reference, which normalizes the merged output).

    Returns:
        Guided output in ``positive``'s dtype.

    References:
        https://github.com/ChenDarYen/Normalized-Attention-Guidance (MIT)
    """
    if tuple(positive.shape) != tuple(negative.shape):
        raise ValueError(
            f"positive and negative must share one shape, got {tuple(positive.shape)} and "
            f"{tuple(negative.shape)}"
        )
    if scale < 1.0:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    pos = positive.astype(mx.float32)
    guided = pos * scale - negative.astype(mx.float32) * (scale - 1.0)
    norm_pos = mx.sum(mx.abs(pos), axis=axis, keepdims=True)
    norm_guided = mx.sum(mx.abs(guided), axis=axis, keepdims=True)
    # min(r, tau) / r with r = norm_guided / norm_pos, written so that r <= tau
    # (including 0 / 0) leaves the guidance untouched and r = inf zeroes it.
    clip = mx.where(norm_guided > tau * norm_pos, tau * norm_pos / norm_guided, 1.0)
    out = alpha * (guided * clip) + (1.0 - alpha) * pos
    return out.astype(positive.dtype)

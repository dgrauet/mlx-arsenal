"""Head-wise temporal reuse of sparse attention masks (HEART).

Content-dependent sparse masks (e.g.
:func:`~mlx_arsenal.attention.top_p_block_mask` over
:func:`~mlx_arsenal.attention.antidiagonal_block_scores`) change slowly
across denoising steps, and some heads change much less than others. HEART
(Temporal Mask Reuse, arXiv 2605.14513) keeps, per head, an *anchor*: the
token-mean of Q and K at the step where the head's mask was last rebuilt.
At every step the head's drift is the L1 distance between the current
pooled Q/K and its anchor; heads whose drift exceeds ``δ`` rebuild their
mask and move their anchor, the others reuse the anchored mask. Because the
anchor only moves on refresh, small drifts accumulate until they cross
``δ``.

Caller-side, as in the paper: the dense warm-up steps (simply do not use the
cache during them), one cache per layer and per CFG branch, the choice of
``δ``, and the mask predictor itself.
"""

from __future__ import annotations

import mlx.core as mx

_NORM_FLOOR = 1e-12


def pooled_qk(q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
    """Token-mean of queries and keys per head (HEART's drift summary).

    Args:
        q: ``(B, H, Nq, D)`` queries.
        k: ``(B, H, Nk, D)`` keys, same ``(B, H, ·, D)``.

    Returns:
        ``(qbar, kbar)``, each ``(B, H, D)`` float32.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must have rank 4, got {q.ndim} and {k.ndim}")
    if q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q and k shapes are incompatible: {tuple(q.shape)} vs {tuple(k.shape)}")
    return mx.mean(q.astype(mx.float32), axis=2), mx.mean(k.astype(mx.float32), axis=2)


def qk_drift(
    qbar_a: mx.array,
    kbar_a: mx.array,
    qbar_b: mx.array,
    kbar_b: mx.array,
    *,
    relative: bool = False,
) -> mx.array:
    """Per-head drift between two pooled Q/K summaries.

    ``‖q̄_a − q̄_b‖₁ + ‖k̄_a − k̄_b‖₁`` over the feature axis (HEART Eq. 4).
    This raw distance scales with the magnitude of Q and K, so HEART's
    threshold is model-specific (8 or 30 in the paper). ``relative=True``
    divides by ``‖q̄_a‖₁ + ‖k̄_a‖₁`` (the reference summary), which makes the
    drift scale-free — an extension, not part of the paper.

    Args:
        qbar_a, kbar_a: ``(B, H, D)`` reference (anchor) summaries.
        qbar_b, kbar_b: ``(B, H, D)`` current summaries.
        relative: Normalize by the reference summary's L1 norm.

    Returns:
        ``(B, H)`` float32 drift.
    """
    shapes = {tuple(x.shape) for x in (qbar_a, kbar_a, qbar_b, kbar_b)}
    if len(shapes) != 1 or qbar_a.ndim != 3:
        raise ValueError(f"all summaries must share one (B, H, D) shape, got {sorted(shapes)}")
    d = mx.sum(mx.abs(qbar_a - qbar_b), axis=-1) + mx.sum(mx.abs(kbar_a - kbar_b), axis=-1)
    if relative:
        ref = mx.sum(mx.abs(qbar_a), axis=-1) + mx.sum(mx.abs(kbar_a), axis=-1)
        d = d / mx.maximum(ref, _NORM_FLOOR)
    return d.astype(mx.float32)

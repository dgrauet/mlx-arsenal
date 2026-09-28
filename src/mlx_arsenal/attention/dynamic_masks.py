"""Content-dependent block-sparse attention masks (XAttention-style).

The shipped video masks are static patterns. Dynamic predictors instead
score every (query block, key block) pair from the actual Q/K and keep the
blocks carrying most of the attention mass. This module implements the
XAttention estimator (Xu et al., arXiv 2503.16428) as dense MLX array math,
written from the paper's description:

1. :func:`antidiagonal_block_scores` — cheap per-block attention mass
   estimate from strided anti-diagonal sums (~1/stride of ``QKᵀ`` FLOPs).
2. :func:`top_p_block_mask` — per query block, keep the smallest set of key
   blocks covering a fraction ``τ`` of the mass; ``τ`` may be per head.

The output is a block-level additive mask, consumable as ``block_mask`` by
:func:`~mlx_arsenal.attention.centroid_compensated_attention` with
``labels = mx.arange(N) // block_size``. For reuse across denoising steps
see :class:`mlx_arsenal.diffusion.HeadMaskCache`.
"""

from __future__ import annotations

import math

import mlx.core as mx


def antidiagonal_block_scores(
    q: mx.array,
    k: mx.array,
    *,
    block_size: int = 128,
    stride: int = 16,
    scale: float | None = None,
) -> mx.array:
    """Estimate the attention mass of every (query block, key block) pair.

    Tokens are grouped into contiguous blocks of ``block_size``. Each
    ``stride × stride`` sub-block of ``QKᵀ`` is summarized by the sum along
    its anti-diagonal, ``Σ_r q_{iS+S-1-r} · k_{jS+r}``, which touches every
    query and key of the sub-block once. These strided logits (scaled by
    ``scale / stride``) are softmaxed per row in float32, summed over each
    ``(block_size/stride)²`` tile, and normalized so every query-block row
    sums to 1. XAttention keeps the raw tile sums; the normalization does not
    change :func:`top_p_block_mask`, which is relative to the row total.

    The strided matmul costs about ``1/stride`` of a full ``QKᵀ`` in FLOPs;
    the logits and softmax are ``1/stride²`` of the full attention matrix.
    All ``B·H`` heads are scored at once: transient memory is roughly
    ``2 · B·H · (Nq/stride)·(Nk/stride) · 4`` bytes (logits + softmax) —
    ≈16 MB per head at ``N = 32k``, ``stride = 16``, but ≈14 GB for a Wan
    720p layer (``N ≈ 75.8k``, ``H = 40``, CFG ``B = 2``). For large models,
    call it per head slice (``q[:, h0:h1]``) and concatenate.

    Sequence lengths must be multiples of ``block_size``: pad beforehand.
    Zero-padded keys still take softmax mass (as in XAttention's non-causal
    mode); mask their blocks out afterwards if that matters. Permute tokens
    first (e.g. :func:`~mlx_arsenal.attention.block_contiguous_permutation`)
    if blocks should follow another order.

    Args:
        q: ``(B, H, Nq, D)`` queries, ``Nq`` a multiple of ``block_size``.
        k: ``(B, H, Nk, D)`` keys with the same ``(B, H)`` (GQA not
            supported), ``Nk`` a multiple of ``block_size``.
        block_size: Block size in tokens, a multiple of ``stride``.
        stride: Anti-diagonal sampling stride, ``>= 1``.
        scale: Logit scale. Defaults to ``1 / sqrt(D)``.

    Returns:
        ``(B, H, Nq // block_size, Nk // block_size)`` float32 block masses,
        each row summing to 1.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must have rank 4, got {q.ndim} and {k.ndim}")
    if k.shape[:2] != q.shape[:2]:
        raise ValueError(f"q and k must share (batch, heads), got {q.shape[:2]} and {k.shape[:2]}")
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    if k.shape[3] != D:
        raise ValueError(f"q and k head dim differ: {D} vs {k.shape[3]}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if block_size % stride:
        raise ValueError(f"block_size ({block_size}) must be a multiple of stride ({stride})")
    if Nq % block_size or Nk % block_size:
        raise ValueError(f"Nq ({Nq}) and Nk ({Nk}) must be a multiple of block_size ({block_size})")

    S = stride
    s = 1.0 / math.sqrt(D) if scale is None else scale
    # Slot r of a strided row holds k_{jS+r} and q_{iS+S-1-r}: their dot
    # products summed over r give the anti-diagonal of each S×S sub-block.
    k_r = k.astype(mx.float32).reshape(B, H, Nk // S, S * D)
    q_r = (
        q.astype(mx.float32)
        .reshape(B, H, Nq // S, S, D)[:, :, :, ::-1, :]
        .reshape(B, H, Nq // S, S * D)
    )
    probs = mx.softmax((q_r @ k_r.swapaxes(-1, -2)) * (s / S), axis=-1)
    t = block_size // S
    tiles = probs.reshape(B, H, Nq // block_size, t, Nk // block_size, t).sum(axis=(3, 5))
    return tiles / t


def top_p_block_mask(scores: mx.array, threshold: float | mx.array) -> mx.array:
    """Keep, per query block, the key blocks covering a fraction ``τ`` of the mass.

    Key blocks are ranked by score (descending, stable: lower index first on
    ties) and a block is kept while the mass ranked *before* it is below
    ``τ · row_total`` — so the block crossing the threshold is kept, and every
    row keeps at least one block (also when a row is all zeros). This is
    XAttention's non-causal selection rule, except that XAttention's scatter
    also re-marks key block 0 in every row as a side effect; this function
    keeps only the top-p set.

    ``threshold`` may be one value per head, e.g. the per-head table produced
    by an offline calibration such as HEART's EBC.

    Args:
        scores: ``(B, H, Cq, Ck)`` non-negative block scores, e.g. from
            :func:`antidiagonal_block_scores`.
        threshold: ``τ`` in ``(0, 1]``, or an ``(H,)`` array of per-head
            values in ``(0, 1]``.

    Returns:
        ``(B, H, Cq, Ck)`` float32 additive mask: ``0`` for kept blocks,
        ``-inf`` for skipped ones. For token-level attention without
        compensation, expand it with ``mx.repeat`` along the last two axes
        by ``block_size``.
    """
    if scores.ndim != 4:
        raise ValueError(f"scores must have rank 4 (B, H, Cq, Ck), got shape {tuple(scores.shape)}")
    H = scores.shape[1]
    if isinstance(threshold, mx.array):
        if tuple(threshold.shape) != (H,):
            raise ValueError(
                f"threshold array must have shape ({H},), got {tuple(threshold.shape)}"
            )
        tau = threshold.astype(mx.float32)
        if not mx.all(mx.logical_and(tau > 0, tau <= 1)).item():
            raise ValueError("threshold values must be in (0, 1]")
        tau = tau.reshape(1, H, 1, 1)
    else:
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        tau = mx.array(threshold, dtype=mx.float32)

    s = scores.astype(mx.float32)
    order = mx.argsort(-s, axis=-1)
    ranked = mx.take_along_axis(s, order, axis=-1)
    before = mx.cumsum(ranked, axis=-1) - ranked
    total = mx.sum(s, axis=-1, keepdims=True)
    keep_ranked = mx.logical_or(before < tau * total, mx.arange(s.shape[-1]) == 0)
    keep = mx.take_along_axis(keep_ranked, mx.argsort(order, axis=-1), axis=-1)
    return mx.where(keep, 0.0, float("-inf")).astype(mx.float32)

"""Content-dependent block-sparse attention masks (XAttention-style).

The shipped video masks are static patterns. Dynamic predictors instead
score every (query block, key block) pair from the actual Q/K and keep the
blocks carrying most of the attention mass. This module implements the
XAttention estimator (Xu et al., arXiv 2503.16428) as dense MLX array math,
written from the paper's description:

1. :func:`antidiagonal_block_scores` — cheap per-block attention mass
   estimate from strided anti-diagonal sums (~1/stride² of ``QKᵀ``).
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

    Cost is about ``1/stride²`` of a full ``QKᵀ``; memory is one float32
    ``(Nq/stride, Nk/stride)`` matrix per head (≈16 MB per head at
    ``N = 32k``, ``stride = 16``). Permute tokens first (e.g.
    :func:`~mlx_arsenal.attention.block_contiguous_permutation`) if blocks
    should follow another order.

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

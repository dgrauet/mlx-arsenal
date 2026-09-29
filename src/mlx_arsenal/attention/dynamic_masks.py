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

SPADE (Liu et al., arXiv 2608.03335) adds input-adaptive blocking:
:func:`block_self_similarity` (query cohesion per block),
:func:`select_tiling` (per-head choice among candidate 3D tilings),
:func:`minmax_block_scores` (min/max block summaries) and
:func:`top_k_block_mask` (fixed per-row budget).

The masks are block-level and additive, consumable as ``block_mask`` by
:func:`~mlx_arsenal.attention.centroid_compensated_attention` with
``labels = mx.arange(N) // block_size``. For reuse across denoising steps
see :class:`mlx_arsenal.diffusion.HeadMaskCache`.

RBS-Attention (Song et al., arXiv 2609.20971) guards centroid estimates
against mean dilution: :func:`radius_bounded_block_scores` (centroid and
radius-bounded log scores) and :func:`relative_block_mask` (keep blocks
within ``α`` of the row maximum).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mlx.core as mx

from mlx_arsenal.attention.compensation import tile_labels


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

    Scores must be finite: a NaN makes its row total NaN, and only the forced
    top-ranked block survives. Validating a ``threshold`` array reads it on
    the host (one sync per call), negligible next to the attention it gates.

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


_ZERO_NORM = 1e-8


def block_self_similarity(x: mx.array, *, block_size: int) -> mx.array:
    """Mean pairwise cosine similarity of the tokens inside each contiguous block.

    SPADE's query-cohesion summary (arXiv 2608.03335, "SICS"): for each block
    of ``block_size`` consecutive tokens, the mean cosine similarity over all
    pairs of distinct tokens, ignoring zero-norm (padding) tokens. Computed
    in ``O(block_size · D)`` as ``(‖Σ x̂‖² − Σ‖x̂‖²) / (n(n−1))`` with ``x̂`` the
    normalized valid tokens and ``n`` their count; blocks with fewer than two
    valid tokens score 0. High values mean the block's queries point the same
    way, so a per-block summary (e.g. :func:`minmax_block_scores`) represents
    them well.

    Args:
        x: ``(B, H, N, D)`` tokens (typically queries) in block order.
        block_size: Tokens per block; ``N`` must be a multiple.

    Returns:
        ``(B, H, N // block_size)`` float32 in ``[-1, 1]``.
    """
    if x.ndim != 4:
        raise ValueError(f"x must have rank 4 (B, H, N, D), got shape {tuple(x.shape)}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    B, H, N, D = x.shape
    if N % block_size:
        raise ValueError(f"N ({N}) must be a multiple of block_size ({block_size})")
    xb = x.astype(mx.float32).reshape(B, H, N // block_size, block_size, D)
    norm = mx.linalg.norm(xb, axis=-1, keepdims=True)
    valid = norm > _ZERO_NORM
    unit = mx.where(valid, xb / mx.maximum(norm, _ZERO_NORM), 0.0)
    total = mx.sum(mx.square(mx.sum(unit, axis=-2)), axis=-1)
    self_terms = mx.sum(mx.square(unit), axis=(-1, -2))
    n = mx.sum(valid.astype(mx.float32), axis=(-1, -2))
    pairs = n * (n - 1)
    return mx.where(pairs > 0, (total - self_terms) / mx.maximum(pairs, 1.0), 0.0)


def select_tiling(
    q: mx.array,
    grid: tuple[int, int, int],
    tiles: Sequence[tuple[int, int, int]],
) -> mx.array:
    """Pick, per head, the 3D tiling under which the queries are most cohesive.

    SPADE's input-adaptive blocking: for each candidate ``(tt, th, tw)`` tile,
    tokens (T-major, ``N = T·H·W``) are reordered so every tile is contiguous
    and the tiles' :func:`block_self_similarity` is averaged. Each (batch,
    head) takes the candidate with the highest mean cohesion (the first one
    on ties). To use a choice, reorder q/k/v with
    ``mx.argsort(tile_labels(T, H, W, tile=tiles[c]))`` along the token axis
    and build block masks with ``block_size = tt·th·tw``.

    Args:
        q: ``(B, H, N, D)`` queries in T-major token order.
        grid: ``(T, H, W)`` latent grid with ``T·H·W == N``.
        tiles: Candidate tile shapes, each dividing ``grid``.

    Returns:
        ``(B, H)`` int32 index into ``tiles``.
    """
    if q.ndim != 4:
        raise ValueError(f"q must have rank 4 (B, H, N, D), got shape {tuple(q.shape)}")
    T, Hg, W = grid
    if T * Hg * W != q.shape[2]:
        raise ValueError(f"grid {grid} has {T * Hg * W} tokens, q has {q.shape[2]}")
    if not tiles:
        raise ValueError("tiles must list at least one candidate")
    cohesion = []
    for tile in tiles:
        order = mx.argsort(tile_labels(T, Hg, W, tile=tile))
        blocked = mx.take(q, order, axis=2)
        size = tile[0] * tile[1] * tile[2]
        cohesion.append(mx.mean(block_self_similarity(blocked, block_size=size), axis=-1))
    return mx.argmax(mx.stack(cohesion), axis=0).astype(mx.int32)


def minmax_block_scores(q: mx.array, k: mx.array, *, block_size: int) -> mx.array:
    """Cheap per-block attention estimate from element-wise min/max summaries.

    SPADE's estimator ("DSA"): each block of ``block_size`` consecutive
    tokens is summarized by its element-wise max and min over tokens, and the
    (query block, key block) score is
    ``max((q_max + q_min) · k_maxᵀ, (q_max + q_min) · k_minᵀ)`` — a
    ``(Nq/bs) × (Nk/bs)`` matmul pair instead of ``Nq × Nk``. Scores are
    unscaled logit estimates (as in the reference): rank them with
    :func:`top_k_block_mask`, or softmax each row before
    :func:`top_p_block_mask`. Reorder tokens first (e.g. by the tiling from
    :func:`select_tiling`) so blocks are meaningful.

    Args:
        q: ``(B, H, Nq, D)`` queries in block order.
        k: ``(B, H, Nk, D)`` keys in block order, same ``(B, H)``.
        block_size: Tokens per block; ``Nq`` and ``Nk`` must be multiples.

    Returns:
        ``(B, H, Nq // block_size, Nk // block_size)`` float32 scores.
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
    if Nq % block_size or Nk % block_size:
        raise ValueError(f"Nq ({Nq}) and Nk ({Nk}) must be a multiple of block_size ({block_size})")
    qb = q.astype(mx.float32).reshape(B, H, Nq // block_size, block_size, D)
    kb = k.astype(mx.float32).reshape(B, H, Nk // block_size, block_size, D)
    q_mid = mx.max(qb, axis=3) + mx.min(qb, axis=3)
    with_max = q_mid @ mx.max(kb, axis=3).swapaxes(-1, -2)
    with_min = q_mid @ mx.min(kb, axis=3).swapaxes(-1, -2)
    return mx.maximum(with_max, with_min)


def top_k_block_mask(scores: mx.array, k: int) -> mx.array:
    """Keep the ``k`` highest-scoring key blocks of every query-block row.

    SPADE's selection rule (a fixed block budget, e.g. 17 % of the key
    blocks on Wan 2.1). Ties keep the lower block index; ``k`` larger than
    the number of key blocks keeps them all.

    Args:
        scores: ``(B, H, Cq, Ck)`` block scores, e.g. from
            :func:`minmax_block_scores` or :func:`antidiagonal_block_scores`.
        k: Blocks kept per row, ``>= 1``.

    Returns:
        ``(B, H, Cq, Ck)`` float32 additive mask (``0`` kept, ``-inf`` skipped).
    """
    if scores.ndim != 4:
        raise ValueError(f"scores must have rank 4 (B, H, Cq, Ck), got shape {tuple(scores.shape)}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rank = mx.argsort(mx.argsort(-scores.astype(mx.float32), axis=-1), axis=-1)
    return mx.where(rank < k, 0.0, float("-inf")).astype(mx.float32)


def _quantile(x: mx.array, p: float) -> mx.array:
    """Linear-interpolated quantile ``p`` along the last axis (numpy's default)."""
    n = x.shape[-1]
    s = mx.sort(x, axis=-1)
    pos = p * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[..., lo] * (1.0 - frac) + s[..., hi] * frac


def radius_bounded_block_scores(
    q: mx.array,
    k: mx.array,
    *,
    block_size: int,
    low: float = 0.5,
    high: float = 0.9,
) -> tuple[mx.array, mx.array]:
    """RBS-Attention block scores: centroid and radius-bounded ("rescue").

    A block centroid ``c_b`` can hide one strongly matching key among keys that
    cancel it ("mean dilution"). RBS (arXiv 2609.20971) adds the key block's
    radius ``r_b = max ‖k − c_b‖₂``: by Cauchy-Schwarz
    ``qᵀk ≤ qᵀc_b + ‖q‖·r_b`` for every key of the block. Per query token::

        ℓ_base   = qᵀc_b / √D
        ℓ_rescue = (qᵀc_b + ‖q‖·r_b·β_b) / √D

    with ``β_b = clip((r_b − r_low) / (r_high − r_low), 0, 1)`` from the
    ``low`` / ``high`` quantiles of the radii of each (batch, head), so only
    unusually spread blocks get the bound. Each (query block, key block) score
    is the logsumexp of the logits over the query block's tokens — the log of
    the paper's ``Σ exp(ℓ)``. Select with :func:`relative_block_mask` per
    branch and take the union (``mx.maximum`` of the additive masks), plus any
    forced blocks (sinks, diagonal band).

    ``β_b`` is 1 above ``r_high`` and 0 below ``r_low``; when the two
    quantiles coincide it is ``1`` for blocks strictly wider than ``r_low``.

    Unlike the other estimators it scores every query token: an
    ``Nq × Nk / block_size`` matmul, cheaper than ``QKᵀ`` by ``block_size``
    in FLOPs, but it holds about three ``(B, H, Nq, Nk / block_size)``
    float32 tensors at once — ≈1.1 GB per head for a Wan 720p layer
    (``N ≈ 75.8k``, ``block_size = 64``). Call it per head (or per group of
    heads) at video scale.

    Args:
        q: ``(B, H, Nq, D)`` queries in block order.
        k: ``(B, H, Nk, D)`` keys in block order, same ``(B, H)``.
        block_size: Tokens per block; ``Nq`` and ``Nk`` must be multiples.
        low: Radius quantile below which ``β_b = 0`` (paper: 0.5).
        high: Radius quantile above which ``β_b = 1`` (paper: 0.9).

    Returns:
        ``(base, rescue)``, each ``(B, H, Nq // block_size, Nk // block_size)``
        float32 **log** scores. Use them with :func:`relative_block_mask` or
        :func:`top_k_block_mask` (order-only); :func:`top_p_block_mask` needs
        masses, i.e. ``mx.softmax(scores, axis=-1)`` first.
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
    if Nq % block_size or Nk % block_size:
        raise ValueError(f"Nq ({Nq}) and Nk ({Nk}) must be a multiple of block_size ({block_size})")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"need 0 <= low < high <= 1, got low={low}, high={high}")
    kb = k.astype(mx.float32).reshape(B, H, Nk // block_size, block_size, D)
    centroid = mx.mean(kb, axis=3)  # (B, H, Ck, D)
    radius = mx.max(mx.linalg.norm(kb - centroid[:, :, :, None], axis=-1), axis=-1)  # (B, H, Ck)
    r_low = _quantile(radius, low)[..., None]
    r_high = _quantile(radius, high)[..., None]
    span = r_high - r_low
    beta = mx.where(
        span > 0,
        mx.clip((radius - r_low) / mx.where(span > 0, span, 1.0), 0.0, 1.0),
        (radius > r_low).astype(mx.float32),
    )
    qf = q.astype(mx.float32) * (1.0 / math.sqrt(D))  # scale once: the bound's ‖q‖ inherits it
    dot = qf @ centroid.swapaxes(-1, -2)  # (B, H, Nq, Ck) base logits
    Cq, Ck = Nq // block_size, Nk // block_size

    def pool(logits: mx.array) -> mx.array:
        return mx.logsumexp(logits.reshape(B, H, Cq, block_size, Ck), axis=3)

    base = pool(dot)
    bound = mx.linalg.norm(qf, axis=-1, keepdims=True) * (radius * beta)[:, :, None, :]
    return base, pool(dot + bound)


def relative_block_mask(log_scores: mx.array, alpha: float) -> mx.array:
    """Keep key blocks whose score is at least ``alpha`` times the row maximum.

    RBS-Attention's selection rule, on log scores (as returned by
    :func:`radius_bounded_block_scores`): keep ``b`` when
    ``log S_b ≥ max_b' log S_b' + log α``. The paper applies it to each branch
    separately (``α = 0.22`` base, ``0.18`` rescue, tuned offline on LLM
    prompts) and unions the masks with ``mx.maximum``.

    Scores must be in the log domain (RBS, or logit-like estimates such as
    :func:`minmax_block_scores`); take ``mx.log`` of mass estimates such as
    :func:`antidiagonal_block_scores` first. A row whose scores are all equal
    (including all ``-inf``) keeps every block; a NaN row keeps none.

    Args:
        log_scores: ``(B, H, Cq, Ck)`` log block scores.
        alpha: Relative threshold in ``(0, 1]``; ``1`` keeps only the maximum.

    Returns:
        ``(B, H, Cq, Ck)`` float32 additive mask (``0`` kept, ``-inf`` skipped).
    """
    if log_scores.ndim != 4:
        raise ValueError(
            f"log_scores must have rank 4 (B, H, Cq, Ck), got shape {tuple(log_scores.shape)}"
        )
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    s = log_scores.astype(mx.float32)
    keep = s >= mx.max(s, axis=-1, keepdims=True) + math.log(alpha)
    return mx.where(keep, 0.0, float("-inf")).astype(mx.float32)

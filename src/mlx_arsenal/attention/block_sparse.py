"""Block-sparse attention over contiguous token blocks (ADR-0002).

:func:`block_sparse_attention` is the call site a block-sparse kernel plugs
into. Its route is the additive block mask the selectors already return
(:func:`~mlx_arsenal.attention.top_p_block_mask`,
:func:`~mlx_arsenal.attention.top_k_block_mask`,
:func:`~mlx_arsenal.attention.relative_block_mask`,
:func:`~mlx_arsenal.attention.centroid_threshold_block_mask`), over blocks of
``block_size`` consecutive tokens.

No Metal kernel ships yet: :func:`block_sparse_kernel_available` is
``False`` and every call runs the dense backend, which is also the numerical
reference for a future kernel. ADR-0002 lists the gates a kernel must pass
before it moves here.
"""

from __future__ import annotations

import math

import mlx.core as mx

from mlx_arsenal.attention.compensation import _centroid_attention


def block_sparse_kernel_available(dtype: mx.Dtype, head_dim: int) -> bool:
    """Whether :func:`block_sparse_attention` runs a block-sparse kernel for this case.

    Always ``False`` for now: no kernel has passed ADR-0002's gates, so every
    call runs dense. Ports can branch on it today (e.g. to skip a routing
    pass that only pays off with a kernel) and pick up a kernel later
    without code changes.

    Args:
        dtype: Dtype of ``q``, ``k`` and ``v``.
        head_dim: Head dim ``D``.
    """
    del dtype, head_dim
    return False


def block_sparse_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    block_mask: mx.array,
    block_size: int = 64,
    compensate: bool = True,
    scale: float | None = None,
) -> mx.array:
    """Attention restricted to the (query block, key block) pairs a block mask keeps.

    Tokens form contiguous blocks of ``block_size`` along each sequence, in
    the order given; the last block may be partial. To cluster tokens by
    content or by video tile, permute them first (for instance with
    :func:`~mlx_arsenal.attention.block_contiguous_permutation`) and undo the
    permutation on the output.

    For a kept block, attention is exact. A skipped key block is, with
    ``compensate=True``, replaced by one key: its centroid, with a
    ``log L`` logit bias (``L`` its length) and its mean value. This is the
    SVG-EAR / Sol-Attn correction, and the result is exactly
    :func:`~mlx_arsenal.attention.centroid_compensated_attention` with
    ``labels = mx.arange(S) // block_size``. With ``compensate=False``, skipped
    blocks are dropped; a query block that keeps no key block then has an
    empty softmax and an unspecified output.

    The call never reads an array on the host: only shapes and dtypes are
    checked, so it stays lazy and works inside :func:`mx.compile`. The mask
    must be floating point; an entry of exactly ``0`` keeps its block and
    any other value (``-inf``, NaN, a finite bias) skips it. Boolean or
    integer masks are rejected: convert a "1 = keep" route with
    ``mx.where(route, 0.0, -mx.inf)``.

    The dense backend materializes a ``(1|B, 1|H, Sq, Sk + Ck)`` mask in
    ``q.dtype`` (``(1|B, 1|H, Sq, Sk)`` booleans without compensation), per
    batch and head only when ``block_mask`` varies over them. At video
    lengths, call it per head slice.

    Args:
        q: ``(B, H, Sq, D)`` queries.
        k: ``(B, H, Sk, D)`` keys, same ``(B, H)`` (GQA not supported).
        v: ``(B, H, Sk, Dv)`` values.
        block_mask: ``(1|B, 1|H, ceil(Sq / block_size), ceil(Sk / block_size))``
            floating additive block mask: ``0`` keeps a block.
        block_size: Tokens per block, ``>= 1``. A kernel, once available,
            will require 64.
        compensate: Replace skipped key blocks by their centroids (``True``)
            or drop them (``False``).
        scale: Logit scale. Defaults to ``1 / sqrt(D)``.

    Returns:
        ``(B, H, Sq, Dv)`` attention output, in the dtype
        ``mx.fast.scaled_dot_product_attention`` returns for ``q, k, v``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q, k, v must have rank 4, got {q.ndim}, {k.ndim}, {v.ndim}")
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    if k.shape[:2] != q.shape[:2] or v.shape[:2] != q.shape[:2]:
        raise ValueError(
            f"q, k, v must share (batch, heads), got {q.shape[:2]}, {k.shape[:2]}, {v.shape[:2]}"
        )
    if v.shape[2] != Sk:
        raise ValueError(f"k and v key length differ: {Sk} vs {v.shape[2]}")
    if k.shape[3] != D:
        raise ValueError(f"q and k head dim differ: {D} vs {k.shape[3]}")
    if Sq == 0 or Sk == 0:
        raise ValueError(f"query and key sequences must be non-empty, got Sq={Sq}, Sk={Sk}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    Cq, Ck = -(-Sq // block_size), -(-Sk // block_size)
    if not mx.issubdtype(block_mask.dtype, mx.floating):
        # A boolean or 0/1 "keep" mask would read inverted: False == 0 keeps.
        raise ValueError(
            f"block_mask must be a floating additive mask (0 keeps), got {block_mask.dtype}"
        )
    if block_mask.ndim != 4:
        raise ValueError(
            f"block_mask must have rank 4 (B, H, Cq, Ck), got shape {tuple(block_mask.shape)}"
        )
    mb, mh, mq, mk = block_mask.shape
    if mb not in (1, B) or mh not in (1, H) or (mq, mk) != (Cq, Ck):
        raise ValueError(
            f"block_mask shape {tuple(block_mask.shape)} does not broadcast to "
            f"{(B, H, Cq, Ck)} (block_size={block_size})"
        )
    s = 1.0 / math.sqrt(D) if scale is None else scale
    keep = mx.equal(block_mask, 0)

    if compensate:
        ql = (mx.arange(Sq, dtype=mx.int32) // block_size).reshape(1, 1, Sq)
        kl = (mx.arange(Sk, dtype=mx.int32) // block_size).reshape(1, 1, Sk)
        return _centroid_attention(q, k, v, ql, kl, keep, s)
    tok = mx.repeat(mx.repeat(keep, block_size, axis=-2), block_size, axis=-1)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=s, mask=tok[..., :Sq, :Sk])

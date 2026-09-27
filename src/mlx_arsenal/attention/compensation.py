"""Sparse-block compensation: dense reference implementations.

Block-sparse attention drops (query cluster, key cluster) blocks. Two
training-free methods recover part of the dropped signal:

- **SVG-EAR** (arXiv 2603.08982): each skipped key is replaced by its
  cluster centroid, so the output stays a softmax average over the full
  key set. See :func:`centroid_compensated_attention`.
- **SparsePR** (arXiv 2608.18484): a few exact "probe" rows are computed
  densely and a ridge regression predicts the residual on the other rows.
  See :func:`probe_residual_correction` and :func:`select_probe_rows`.

MLX has no block-sparse kernel, so everything here runs dense: these are
quality tools and numerical references, not speedups. Clusters are
caller-provided integer labels; :func:`tile_labels` gives the labels that
match the shipped Sliding Tile Attention masks.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import cast

import mlx.core as mx

from mlx_arsenal._typing import item_int
from mlx_arsenal.attention._thw import thw_coords as _thw_coords
from mlx_arsenal.attention._thw import validate_thw as _validate_thw


def tile_labels(T: int, H: int, W: int, *, tile: tuple[int, int, int]) -> mx.array:
    """Cluster label of every token for a `(tt, th, tw)` tiling of a video grid.

    The label of token `(t, h, w)` is the row-major index of its tile in the
    `(T // tt, H // th, W // tw)` tile grid; tokens are in T-major order.
    Labels line up with :func:`~mlx_arsenal.attention.sliding_tile_block_mask`
    evaluated at tile resolution, which yields the matching cluster-level
    block mask:

    ```python
    labels = tile_labels(T, H, W, tile=(tt, th, tw))
    block_mask = sliding_tile_block_mask(T // tt, H // th, W // tw, tile=(1, 1, 1), window=window)
    ```

    Args:
        T: Number of frames. Must be divisible by `tile[0]`.
        H: Latent height. Must be divisible by `tile[1]`.
        W: Latent width. Must be divisible by `tile[2]`.
        tile: `(tt, th, tw)` tile dims, all positive.

    Returns:
        `(T*H*W,)` int32 labels in `[0, (T//tt) * (H//th) * (W//tw))`.
    """
    _validate_thw(T, H, W)
    tt, th, tw = tile
    if tt <= 0 or th <= 0 or tw <= 0:
        raise ValueError(f"tile dims must be positive, got {tile}")
    if T % tt or H % th or W % tw:
        raise ValueError(f"(T, H, W)={(T, H, W)} not divisible by tile={tile}")
    t_flat, h_flat, w_flat = _thw_coords(T, H, W)
    gh, gw = H // th, W // tw
    t_tile = mx.floor_divide(t_flat, tt)
    h_tile = mx.floor_divide(h_flat, th)
    w_tile = mx.floor_divide(w_flat, tw)
    return ((t_tile * gh + h_tile) * gw + w_tile).astype(mx.int32)


def _cluster_labels(labels: mx.array, B: int, H: int, S: int, C: int, name: str) -> mx.array:
    """Validate `(S,)` / `(B, H, S)` integer labels in `[0, C)`; return `(B, H, S)` int32."""
    if not mx.issubdtype(labels.dtype, mx.integer):
        raise ValueError(f"{name} must have an integer dtype, got {labels.dtype}")
    if tuple(labels.shape) not in ((S,), (B, H, S)):
        raise ValueError(f"{name} must have shape ({S},) or {(B, H, S)}, got {tuple(labels.shape)}")
    lo, hi = item_int(mx.min(labels)), item_int(mx.max(labels))
    if lo < 0 or hi >= C:
        raise ValueError(f"{name} values must lie in [0, {C}), got range [{lo}, {hi}]")
    return mx.broadcast_to(labels.astype(mx.int32), (B, H, S))


def centroid_compensated_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    q_labels: mx.array,
    k_labels: mx.array,
    block_mask: mx.array,
    scale: float | None = None,
) -> mx.array:
    """Block-sparse attention with skipped blocks compensated by key centroids (SVG-EAR).

    Queries and keys are grouped into clusters by integer labels. For every
    (query cluster, key cluster) block kept by `block_mask`, attention is
    exact. For a skipped block, each key and value is replaced by the mean
    key and mean value of its cluster. The result is exactly dense attention
    over that modified key set, so every output row is still a convex
    softmax average — skipped blocks are approximated, not dropped.

    Keys of one cluster share a single centroid logit, so the skipped part
    collapses to one extra key per cluster carrying a `log n_c` bias
    (`n_c` = cluster size). The implementation therefore runs one
    `mx.fast.scaled_dot_product_attention` over `[K; K̄]` / `[V; V̄]` with an
    additive `(Sq, Sk + Ck)` mask. It is dense — O(Sq·(Sk+Ck)) mask memory,
    no speedup — and meant as a quality tool and a reference for a future
    block-sparse kernel.

    With `block_mask` all `0` this is exact dense attention; with it all
    `-inf`, attention over the cluster centroids only.

    Args:
        q: `(B, H, Sq, D)` queries.
        k: `(B, H, Sk, D)` keys. Same `(B, H)` as `q` (GQA not supported).
        v: `(B, H, Sk, Dv)` values.
        q_labels: `(Sq,)` or `(B, H, Sq)` integer query-cluster labels in `[0, Cq)`.
        k_labels: `(Sk,)` or `(B, H, Sk)` integer key-cluster labels in `[0, Ck)`.
            A cluster with no member contributes nothing.
        block_mask: `(..., Cq, Ck)` additive mask with entries `0` (block
            computed exactly) or `-inf` (block skipped and compensated),
            broadcastable to `(B, H, Cq, Ck)`. For Sliding Tile Attention,
            see :func:`tile_labels`.
        scale: Logit scale. Defaults to `1 / sqrt(D)`.

    Returns:
        `(B, H, Sq, Dv)` attention output in `q.dtype`.
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
    if block_mask.ndim < 2:
        raise ValueError(f"block_mask must have rank >= 2, got shape {tuple(block_mask.shape)}")
    Cq, Ck = block_mask.shape[-2:]
    ql = _cluster_labels(q_labels, B, H, Sq, Cq, "q_labels")
    kl = _cluster_labels(k_labels, B, H, Sk, Ck, "k_labels")

    bm = block_mask.astype(mx.float32)
    kept = mx.equal(bm, 0.0)
    if not mx.all(mx.logical_or(kept, mx.isneginf(bm))).item():
        raise ValueError("block_mask entries must be 0 or -inf")
    try:
        keep = mx.broadcast_to(kept, (B, H, Cq, Ck))
    except ValueError as e:
        raise ValueError(
            f"block_mask shape {tuple(block_mask.shape)} does not broadcast to {(B, H, Cq, Ck)}"
        ) from e

    onehot = mx.equal(mx.expand_dims(kl, -1), mx.arange(Ck)).astype(mx.float32)  # (B, H, Sk, Ck)
    counts = onehot.sum(axis=-2)  # (B, H, Ck)
    inv_counts = mx.expand_dims(1.0 / mx.maximum(counts, 1.0), -1)
    onehot_t = onehot.swapaxes(-1, -2)
    k_bar = (onehot_t @ k.astype(mx.float32)) * inv_counts
    v_bar = (onehot_t @ v.astype(mx.float32)) * inv_counts

    keep_rows = mx.take_along_axis(keep, mx.expand_dims(ql, -1), axis=-2)  # (B, H, Sq, Ck)
    keep_tok = mx.take_along_axis(keep_rows, mx.expand_dims(kl, -2), axis=-1)  # (B, H, Sq, Sk)
    neg_inf = mx.array(float("-inf"), dtype=mx.float32)
    tok_mask = mx.where(keep_tok, 0.0, neg_inf)
    log_n = mx.expand_dims(mx.log(counts), -2)  # -inf for empty clusters
    cent_mask = mx.where(keep_rows, neg_inf, log_n)
    mask = mx.concatenate([tok_mask, cent_mask], axis=-1).astype(q.dtype)

    k_ext = mx.concatenate([k, k_bar.astype(k.dtype)], axis=2)
    v_ext = mx.concatenate([v, v_bar.astype(v.dtype)], axis=2)
    s = 1.0 / math.sqrt(D) if scale is None else scale
    return mx.fast.scaled_dot_product_attention(q, k_ext, v_ext, scale=s, mask=mask)


def _middle_out(n: int) -> list[int]:
    """Indices of `range(n)` from the middle outward: mid, mid+1, mid-1, mid+2, ..."""
    mid = (n - 1) // 2
    order = [mid]
    for d in range(1, n):
        if mid + d < n:
            order.append(mid + d)
        if mid - d >= 0:
            order.append(mid - d)
    return order


def select_probe_rows(q_labels: mx.array, num_probes: int) -> tuple[mx.array, mx.array]:
    """Pick probe query rows for :func:`probe_residual_correction`, spread across clusters.

    Round-robin over the non-empty clusters in increasing label order: pass
    `r` takes, from every cluster that still has unused members, its member
    at middle-out rank `r` (positions sorted, middle first, then alternating
    outward), until `num_probes` rows are taken.

    Deviation from SparsePR, which takes the row nearest to each query-group
    centroid: that needs `q` and per-head groups. This selection depends only
    on the labels, so a single `probe_idx` serves every head. Pass your own
    `probe_idx` to :func:`probe_residual_correction` for another policy.

    Args:
        q_labels: `(Sq,)` non-negative integer query-cluster labels.
        num_probes: Number of rows to pick, in `[1, Sq]`.

    Returns:
        `(probe_idx, weights)`, both `(num_probes,)`: int32 row indices and
        float32 weights `|G_a| / m_a` (cluster size over probes taken from
        that cluster). When every cluster gets at least one probe, the
        weights sum to `Sq`.
    """
    if q_labels.ndim != 1:
        raise ValueError(f"q_labels must be 1D, got shape {tuple(q_labels.shape)}")
    if not mx.issubdtype(q_labels.dtype, mx.integer):
        raise ValueError(f"q_labels must have an integer dtype, got {q_labels.dtype}")
    S = q_labels.shape[0]
    if not 1 <= num_probes <= S:
        raise ValueError(f"num_probes must be in [1, {S}], got {num_probes}")
    labels = cast(list[int], q_labels.tolist())
    if min(labels) < 0:
        raise ValueError("q_labels must be non-negative")

    members: dict[int, list[int]] = {}
    for pos, lab in enumerate(labels):
        members.setdefault(lab, []).append(pos)
    groups = sorted(members)
    orders = {g: [members[g][i] for i in _middle_out(len(members[g]))] for g in groups}

    picks: list[tuple[int, int]] = []
    rank = 0
    while len(picks) < num_probes:
        for g in groups:
            if rank < len(orders[g]):
                picks.append((orders[g][rank], g))
                if len(picks) == num_probes:
                    break
        rank += 1

    taken = Counter(g for _, g in picks)
    idx = [pos for pos, _ in picks]
    weights = [len(members[g]) / taken[g] for _, g in picks]
    return mx.array(idx, dtype=mx.int32), mx.array(weights, dtype=mx.float32)

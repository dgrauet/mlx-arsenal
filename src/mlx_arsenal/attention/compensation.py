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

import mlx.core as mx

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

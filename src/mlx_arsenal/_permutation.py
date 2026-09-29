"""Scatter-based inverse of a permutation (private helper)."""

from __future__ import annotations

import mlx.core as mx


def invert_permutation_last_axis(perm: mx.array) -> mx.array:
    """Inverse of the permutations along the last axis, as int32.

    Equal to ``mx.argsort(perm, axis=-1)`` for valid permutations, but a
    scatter (``inv[perm[i]] = i``) instead of a second sort: 4-5x faster on
    MLX 0.32 (the mlx-lm #1825 pattern). ``perm`` is not validated.
    """
    if perm.ndim < 1:
        raise ValueError("perm must have at least one axis")
    n = perm.shape[-1]
    positions = mx.broadcast_to(mx.arange(n, dtype=mx.int32), perm.shape)
    return mx.put_along_axis(mx.zeros(perm.shape, dtype=mx.int32), perm, positions, axis=-1)

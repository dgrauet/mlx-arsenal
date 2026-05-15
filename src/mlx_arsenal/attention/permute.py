"""Block-contiguous token permutation (SVG2 semantic permutation).

Reorders a sequence of tokens so that high-importance ones fall into the
first contiguous blocks, which is what block-sparse attention kernels
actually need to realize their savings. Pair with ``mx.take(x, perm, axis=...)``
to permute Q/K/V tensors and ``mx.take(y, inv_perm, axis=...)`` to undo.
"""

from __future__ import annotations

import mlx.core as mx


def block_contiguous_permutation(
    scores: mx.array,
    *,
    block_size: int,
    descending: bool = True,
) -> tuple[mx.array, mx.array]:
    """Sort tokens by score so high-importance ones cluster into early blocks.

    Args:
        scores: ``(S,)`` per-token importance score.
        block_size: Block size of the downstream sparse kernel.
            **Informational only** — this function does not pad ``S`` to a
            multiple of ``block_size``. The caller is responsible if they
            need exact alignment.
        descending: If ``True`` (default), highest scores land at position 0.
            If ``False``, lowest first.

    Returns:
        ``(perm, inv_perm)`` of shape ``(S,)``, dtype ``int32``.

        - ``perm[i]`` is the original index of the token now at position ``i``.
        - ``inv_perm[i]`` is the new position of the token originally at ``i``.

        Tie-breaking among equal scores is stable (preserves original order),
        which is the MLX ``argsort`` contract.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {scores.shape}")
    if scores.shape[0] == 0:
        raise ValueError("scores must be non-empty")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    keys = scores.astype(mx.float32)
    if descending:
        keys = -keys
    perm = mx.argsort(keys).astype(mx.int32)
    inv_perm = mx.argsort(perm).astype(mx.int32)
    return perm, inv_perm


def invert_permutation(perm: mx.array) -> mx.array:
    """Compute the inverse of a 1D permutation.

    Equivalent to ``mx.argsort(perm)``. Caller is responsible for ensuring
    ``perm`` is a valid permutation of ``[0, S)``; misuse silently produces
    wrong results.

    Args:
        perm: ``(S,)`` int permutation array.

    Returns:
        ``(S,)`` int32 inverse permutation.
    """
    if perm.ndim != 1:
        raise ValueError(f"perm must be 1D, got shape {perm.shape}")
    return mx.argsort(perm).astype(mx.int32)

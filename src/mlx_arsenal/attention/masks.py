"""Attention mask utilities."""

import mlx.core as mx


def causal_mask(
    seq_len: int,
    offset: int = 0,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create a causal (lower-triangular) attention mask.

    Args:
        seq_len: Sequence length.
        offset: Offset for KV cache (total KV length = offset + seq_len).
        dtype: Output dtype. Masked positions are -inf.

    Returns:
        Mask of shape (1, 1, seq_len, offset + seq_len).

    Raises:
        ValueError: if ``seq_len`` is not positive or ``offset`` is negative.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    total = offset + seq_len
    mask = mx.full((seq_len, total), float("-inf"), dtype=dtype)
    rows = mx.arange(seq_len)
    cols = mx.arange(total)
    # Position i can attend to positions <= i + offset
    valid = mx.expand_dims(cols, 0) <= mx.expand_dims(rows + offset, 1)
    mask = mx.where(valid, mx.zeros_like(mask), mask)
    return mask.reshape(1, 1, seq_len, total)


def block_causal_mask(
    seq_len: int,
    block_len: int,
    offset: int = 0,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create a block-causal attention mask (block diffusion LLMs).

    Positions are grouped into blocks of ``block_len`` aligned on absolute
    positions (``position // block_len``). Attention is bidirectional inside
    a block and causal across blocks: the query at absolute position
    ``p = offset + i`` sees key ``j`` iff ``j // block_len <= p // block_len``.
    This is the mask block-diffusion models (LLaDA2.x, SDAR, Nemotron-Labs
    Diffusion) are trained with, and it makes a prefix KV cache exact for
    them. ``block_len=1`` reduces to :func:`causal_mask`.

    The prompt is split into blocks like the rest of the sequence. A model
    that attends to its whole prompt bidirectionally needs its own mask.

    Args:
        seq_len: Number of query positions.
        block_len: Block size in tokens.
        offset: Offset for KV cache (total KV length = offset + seq_len).
        dtype: Output dtype. Masked positions are -inf.

    Returns:
        Mask of shape (1, 1, seq_len, offset + seq_len).

    Raises:
        ValueError: if ``seq_len`` or ``block_len`` is not positive, or
            ``offset`` is negative.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if block_len <= 0:
        raise ValueError(f"block_len must be > 0, got {block_len}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    total = offset + seq_len
    mask = mx.full((seq_len, total), float("-inf"), dtype=dtype)
    row_block = mx.floor_divide(mx.arange(seq_len) + offset, block_len)
    col_block = mx.floor_divide(mx.arange(total), block_len)
    valid = mx.expand_dims(col_block, 0) <= mx.expand_dims(row_block, 1)
    mask = mx.where(valid, mx.zeros_like(mask), mask)
    return mask.reshape(1, 1, seq_len, total)


def sliding_window_mask(
    seq_len: int,
    window_size: int,
    offset: int = 0,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Create a sliding window causal attention mask.

    Each position can attend to at most `window_size` previous positions
    (including itself).

    Args:
        seq_len: Sequence length.
        window_size: Size of the attention window.
        offset: Offset for KV cache.
        dtype: Output dtype. Masked positions are -inf.

    Returns:
        Mask of shape (1, 1, seq_len, offset + seq_len).

    Raises:
        ValueError: if ``seq_len`` or ``window_size`` is not positive, or
            ``offset`` is negative (a non-positive window would produce
            all ``-inf`` rows and NaN softmax downstream).
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    total = offset + seq_len
    mask = mx.full((seq_len, total), float("-inf"), dtype=dtype)
    rows = mx.arange(seq_len)
    cols = mx.arange(total)
    row_pos = rows + offset
    # Can attend to positions in [row_pos - window_size + 1, row_pos]
    valid = (mx.expand_dims(cols, 0) <= mx.expand_dims(row_pos, 1)) & (
        mx.expand_dims(cols, 0) > mx.expand_dims(row_pos - window_size, 1)
    )
    mask = mx.where(valid, mx.zeros_like(mask), mask)
    return mask.reshape(1, 1, seq_len, total)

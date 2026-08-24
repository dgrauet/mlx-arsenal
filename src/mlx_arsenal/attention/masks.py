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

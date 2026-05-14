"""Rotary Position Embeddings (RoPE) — N-axis composable primitives.

RoPE rotates pairs of features in each query/key vector by an angle
that depends on the token's position, encoding position into the
attention dot product. The math is identical across most models; the
practical pitfalls are:

1. **Pair layout.** Two conventions are in the wild:

   - *Interleaved* (Llama2 paper, Matrix-Game, RoFormer original):
     pairs are adjacent — pair k consists of features ``(x[2k], x[2k+1])``.
   - *Half-rotated* (HuggingFace Transformers default for Llama, GPT-NeoX):
     pairs span halves — pair k consists of ``(x[k], x[k + d/2])``.

   The two compute identical attention scores, but the cos/sin layout
   and ``rotate_half`` form differ. This module supports both via
   ``interleaved=True`` (default, RoPE-paper convention) or
   ``interleaved=False``.

2. **Multi-axis (N-D) composition.** Video / image models split the
   head dim across spatial axes — e.g. CogVideoX uses ``[t, h, w]``
   with ``rope_dim_list = [16, 56, 56]``. Each axis gets a 1-D RoPE
   over its own position grid; the resulting per-axis ``(cos, sin)``
   are concatenated along the feature axis. See
   :func:`rope_frequencies_nd`.

3. **mx.fast.rope.** MLX ships ``mx.fast.rope`` for the standard 1-D
   case but does NOT cover N-axis composition or arbitrary frequency
   schedules (Megatron non-interleaved, log-spaced SPLIT). This module
   fills that gap with portable, slow-but-correct primitives.

Model-specific variants (ERNIE-Image Megatron, LTX SPLIT log-spaced)
should stay in their ports — arsenal only covers the "standard"
case. The shapes returned here mirror the most common HF /
Matrix-Game convention.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

__all__ = [
    "apply_rotary_emb",
    "meshgrid_nd",
    "rope_frequencies_1d",
    "rope_frequencies_nd",
    "rotate_half",
]


def rope_frequencies_1d(
    dim: int,
    positions: mx.array,
    theta: float = 10000.0,
    *,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Compute 1-axis RoPE ``(cos, sin)`` for the given positions.

    The output has shape ``(S, dim // 2)`` and represents the per-pair
    rotation angle ``θ_k = pos / theta^(2k/dim)`` for ``k ∈ [0, d/2)``.

    Args:
        dim: Head dimension this axis covers (must be even).
        positions: ``(S,)`` float array of token positions.
        theta: RoPE base. ``10000.0`` is the de-facto standard; some
            recent models use larger bases for longer contexts.
        theta_rescale_factor: NTK-aware rescaling (Reddit user bloc97):
            ``theta *= rescale_factor ** (dim / (dim - 2))``. Use
            ``1.0`` to disable.
        interpolation_factor: Position-interpolation factor for context
            extension (Chen et al., 2023). Positions are scaled by
            this factor before frequency computation. Use ``1.0`` to
            disable.

    Returns:
        ``(cos, sin)`` of shape ``(S, dim // 2)``. Pass to
        :func:`apply_rotary_emb` along with the target tensor.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    half = dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, half, dtype=mx.float32) * 2.0 / dim))
    # outer product: positions[:, None] * freqs[None, :] → (S, half)
    angles = positions.astype(mx.float32)[:, None] * interpolation_factor * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def rope_frequencies_nd(
    dims_per_axis: Sequence[int],
    position_grids: Sequence[mx.array],
    theta: float = 10000.0,
    *,
    theta_rescale_factor: float | Sequence[float] = 1.0,
    interpolation_factor: float | Sequence[float] = 1.0,
) -> tuple[mx.array, mx.array]:
    """Compose multi-axis RoPE by concatenating per-axis frequencies.

    For a video transformer with ``head_dim = sum(dims_per_axis)`` and
    each token addressed by a tuple of positions ``(t, h, w)``, each
    axis is RoPE'd independently and the resulting half-angles are
    concatenated along the last dim.

    Args:
        dims_per_axis: Per-axis head-dim allocation, must sum to the
            attention head dim. All entries must be even.
        position_grids: One ``(S,)`` array of positions per axis.
            ``S`` must be the same across axes (one position tuple
            per token).
        theta: Base frequency, broadcast across axes.
        theta_rescale_factor: Either a scalar (broadcast) or one
            factor per axis. See :func:`rope_frequencies_1d`.
        interpolation_factor: Same as above.

    Returns:
        ``(cos, sin)`` of shape ``(S, sum(dims_per_axis) // 2)``.
    """
    n = len(dims_per_axis)
    if len(position_grids) != n:
        raise ValueError(f"len(dims_per_axis)={n} != len(position_grids)={len(position_grids)}")

    rescales = _broadcast_to_axes(theta_rescale_factor, n, "theta_rescale_factor")
    interps = _broadcast_to_axes(interpolation_factor, n, "interpolation_factor")

    cos_parts: list[mx.array] = []
    sin_parts: list[mx.array] = []
    for axis_dim, grid, rescale, interp in zip(
        dims_per_axis, position_grids, rescales, interps, strict=True
    ):
        c, s = rope_frequencies_1d(
            axis_dim,
            grid,
            theta=theta,
            theta_rescale_factor=rescale,
            interpolation_factor=interp,
        )
        cos_parts.append(c)
        sin_parts.append(s)

    return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)


def _broadcast_to_axes(value: float | Sequence[float], n: int, name: str) -> list[float]:
    if isinstance(value, int | float):
        return [float(value)] * n
    if len(value) == 1:
        return [float(value[0])] * n
    if len(value) != n:
        raise ValueError(f"{name} must be a scalar or length {n}, got length {len(value)}")
    return [float(v) for v in value]


def rotate_half(x: mx.array, *, interleaved: bool = True) -> mx.array:
    """Rotate halves of ``x`` for RoPE application.

    For ``interleaved=True`` (RoPE-paper / Matrix-Game): pairs are
    adjacent; transforms ``[x0, x1, x2, x3, ...]`` into
    ``[-x1, x0, -x3, x2, ...]``.

    For ``interleaved=False`` (HuggingFace Llama / GPT-NeoX): pairs
    span halves; transforms ``[x_left, x_right]`` into
    ``[-x_right, x_left]``.

    Args:
        x: Tensor with even last dim.
        interleaved: Selects the convention.

    Returns:
        Tensor of the same shape with paired elements rotated.
    """
    if interleaved:
        # (..., D) → (..., D/2, 2)
        pairs = mx.reshape(x, (*x.shape[:-1], -1, 2))
        x_even = pairs[..., 0]
        x_odd = pairs[..., 1]
        rotated = mx.stack([-x_odd, x_even], axis=-1)
        return mx.reshape(rotated, x.shape)

    half = x.shape[-1] // 2
    x_left = x[..., :half]
    x_right = x[..., half:]
    return mx.concatenate([-x_right, x_left], axis=-1)


def apply_rotary_emb(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    *,
    interleaved: bool = True,
) -> mx.array:
    """Apply RoPE rotation to ``x``.

    Computes ``x * cos + rotate_half(x) * sin`` where ``cos`` and
    ``sin`` are expanded from the half-dim shape returned by
    :func:`rope_frequencies_1d` / :func:`rope_frequencies_nd` to match
    the full last-dim of ``x``.

    Args:
        x: ``(..., S, D)`` tensor to rotate. The trailing two axes
            must be the sequence and the head dim respectively
            (broadcasts otherwise).
        cos: ``(S, D // 2)`` cosines.
        sin: ``(S, D // 2)`` sines.
        interleaved: See :func:`rotate_half`. Defaults to ``True``
            (RoPE-paper convention).

    Returns:
        Rotated tensor, same shape as ``x``.
    """
    full_cos, full_sin = _expand_freqs(cos, sin, interleaved=interleaved)
    # Reshape (S, D) → broadcast against x. x has ``S`` on some axis
    # before the last — broadcasting handles any (..., S, ..., D) shape
    # as long as S matches.
    full_cos = _broadcast_for(full_cos, x)
    full_sin = _broadcast_for(full_sin, x)

    x_f = x.astype(mx.float32)
    out = x_f * full_cos + rotate_half(x_f, interleaved=interleaved) * full_sin
    return out.astype(x.dtype)


def _expand_freqs(cos: mx.array, sin: mx.array, *, interleaved: bool) -> tuple[mx.array, mx.array]:
    """Expand half-shape ``(S, D/2)`` cos/sin to full-shape ``(S, D)``."""
    if interleaved:
        # Duplicate each entry: [θ0, θ1, ...] → [θ0, θ0, θ1, θ1, ...]
        return _interleave_duplicate(cos), _interleave_duplicate(sin)
    # Concatenate: [θ0, ..., θ_{d/2-1}] → [θ0, ..., θ_{d/2-1}, θ0, ..., θ_{d/2-1}]
    return mx.concatenate([cos, cos], axis=-1), mx.concatenate([sin, sin], axis=-1)


def _interleave_duplicate(t: mx.array) -> mx.array:
    """Duplicate each element along the last axis: ``[..., D/2] → [..., D]``."""
    # repeat_interleave(2, axis=-1) equivalent
    expanded = mx.broadcast_to(t[..., :, None], (*t.shape, 2))
    return mx.reshape(expanded, (*t.shape[:-1], -1))


def _broadcast_for(freq: mx.array, x: mx.array) -> mx.array:
    """Reshape ``(S, D)`` freq to broadcast against the seq + head dims of ``x``.

    Inserts singleton dims for every non-seq, non-head axis. The
    convention is: ``S`` is the axis of ``x`` whose length matches
    ``freq.shape[0]``; ``D`` is the last axis of ``x``.
    """
    s, d = freq.shape
    if x.shape[-1] != d:
        raise ValueError(f"expected x.shape[-1]={d}, got {x.shape[-1]}")
    # Find the axis matching the sequence length.
    candidates = [i for i, dim in enumerate(x.shape[:-1]) if dim == s]
    if not candidates:
        raise ValueError(f"no axis of x with shape={s} (got x.shape={tuple(x.shape)})")
    seq_axis = candidates[0]
    target_shape = [1] * x.ndim
    target_shape[seq_axis] = s
    target_shape[-1] = d
    return mx.reshape(freq, target_shape)


def meshgrid_nd(sizes: Sequence[int]) -> list[mx.array]:
    """Build an ``"ij"``-indexed N-D meshgrid of integer positions.

    Returns one ``(S,)`` flattened position grid per axis, where
    ``S = prod(sizes)``. Use these grids as the ``position_grids``
    argument to :func:`rope_frequencies_nd`.

    Args:
        sizes: Number of positions along each axis (e.g.
            ``[T, H, W]`` for a video).

    Returns:
        List of ``len(sizes)`` flattened position arrays, each of
        shape ``(prod(sizes),)`` and dtype ``float32``.
    """
    n = len(sizes)
    axis_arrays = [mx.arange(s, dtype=mx.float32) for s in sizes]
    grids: list[mx.array] = []
    for i, axis in enumerate(axis_arrays):
        shape = [1] * n
        shape[i] = sizes[i]
        broadcast = mx.broadcast_to(mx.reshape(axis, shape), tuple(sizes))
        grids.append(mx.reshape(broadcast, (-1,)))
    return grids

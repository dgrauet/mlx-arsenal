"""Head-pattern profiler for video DiTs.

Classify each attention head as ``SPATIAL`` (mass concentrated on same-frame
keys), ``TEMPORAL`` (same-position cross-frame), or ``OTHER`` (neither).

All functions assume T-major token flattening — same convention as
``mlx_arsenal.attention.video_masks``: tokens flatten as
``[t0(h0w0..hHwW), t1(...), ..., tT(...)]`` to a sequence of length
``S = T*H*W``.
"""

from enum import Enum
from typing import cast

import mlx.core as mx

from mlx_arsenal.attention._thw import thw_coords as _thw_ids
from mlx_arsenal.attention._thw import validate_thw as _validate_thw


class Kind(Enum):
    """Discrete head-pattern label."""

    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    OTHER = "other"


def classify(
    scores: mx.array,
    *,
    spatial_threshold: float = 0.5,
    temporal_threshold: float = 0.5,
) -> list[Kind]:
    """Convert raw per-head mass scores to discrete labels.

    Args:
        scores: ``(num_heads, 2)`` array. Column 0 = mass on same-frame keys,
            column 1 = mass on same-spatial-position keys.
        spatial_threshold: Min column-0 mass to label a head ``SPATIAL``.
        temporal_threshold: Min column-1 mass to label a head ``TEMPORAL``.

    Returns:
        List of ``Kind`` of length ``num_heads``. Tie-break: if both columns
        exceed their threshold, ``SPATIAL`` wins.
    """
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError(f"scores must have shape (num_heads, 2), got {scores.shape}")
    if not 0.0 <= spatial_threshold <= 1.0:
        raise ValueError(f"spatial_threshold must be in [0, 1], got {spatial_threshold}")
    if not 0.0 <= temporal_threshold <= 1.0:
        raise ValueError(f"temporal_threshold must be in [0, 1], got {temporal_threshold}")
    rows = cast(list[list[float]], scores.tolist())
    out: list[Kind] = []
    for row in rows:
        s, t = float(row[0]), float(row[1])
        if s >= spatial_threshold:
            out.append(Kind.SPATIAL)
        elif t >= temporal_threshold:
            out.append(Kind.TEMPORAL)
        else:
            out.append(Kind.OTHER)
    return out


def classify_heads_from_probs(
    probs: mx.array,
    T: int,
    H: int,
    W: int,
) -> mx.array:
    """Per-head attention-mass fractions on same-frame and same-position keys.

    Uses all queries (no sampling) — assumes the caller has already paid the
    cost of materializing ``(B, num_heads, S, S)`` softmaxed probabilities.

    Args:
        probs: ``(B, num_heads, S, S)`` attention probabilities. The caller is
            responsible for ensuring these are valid (non-negative, sum-1 along
            the key axis); not validated here because it would cost ``O(S²)``.
        T: Number of frames.
        H: Latent height.
        W: Latent width.

    Returns:
        ``(num_heads, 2)`` float array. Column 0 = mass on same-frame keys,
        column 1 = mass on same-spatial-position keys. Averaged over batch
        and queries.
    """
    _validate_thw(T, H, W)
    if probs.ndim != 4:
        raise ValueError(f"probs must be 4D (B, nH, S, S), got shape {probs.shape}")
    S = T * H * W
    if probs.shape[2] != S or probs.shape[3] != S:
        raise ValueError(f"probs last two dims must be {S}, got {probs.shape[2:]}")
    t_flat, h_flat, w_flat = _thw_ids(T, H, W)
    same_frame = mx.equal(mx.expand_dims(t_flat, 0), mx.expand_dims(t_flat, 1))
    same_h = mx.equal(mx.expand_dims(h_flat, 0), mx.expand_dims(h_flat, 1))
    same_w = mx.equal(mx.expand_dims(w_flat, 0), mx.expand_dims(w_flat, 1))
    same_pos = mx.logical_and(same_h, same_w)
    same_frame_f = same_frame.astype(probs.dtype)
    same_pos_f = same_pos.astype(probs.dtype)
    mass_frame = mx.sum(probs * same_frame_f, axis=(2, 3)) / S  # (B, nH)
    mass_pos = mx.sum(probs * same_pos_f, axis=(2, 3)) / S
    mass_frame = mx.mean(mass_frame, axis=0)  # (nH,)
    mass_pos = mx.mean(mass_pos, axis=0)
    return mx.stack([mass_frame, mass_pos], axis=1)


def classify_heads_from_qk(
    q: mx.array,
    k: mx.array,
    T: int,
    H: int,
    W: int,
    *,
    n_samples: int = 64,
    key: mx.array | None = None,
) -> mx.array:
    """Per-head attention-mass fractions, sampled from Q,K.

    Avoids materializing the full ``(B, num_heads, S, S)`` attention by
    sampling ``n_samples`` queries uniformly per call. Reproducible: with a
    fixed ``key``, returns identical results.

    Args:
        q: ``(B, num_heads, S, D)`` query tensor.
        k: ``(B, num_heads, S, D)`` key tensor, same shape as ``q``.
        T: Number of frames.
        H: Latent height.
        W: Latent width.
        n_samples: How many queries to sample uniformly per call. Must satisfy
            ``0 < n_samples <= S``.
        key: ``mx.random`` key for the sampler. If ``None``, uses
            ``mx.random.key(0)`` so default behavior is deterministic across
            calls. Callers who want variance must pass their own key.

    Returns:
        ``(num_heads, 2)`` float array. Column 0 = mass on same-frame keys,
        column 1 = mass on same-spatial-position keys.
    """
    _validate_thw(T, H, W)
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            f"q and k must be 4D (B, nH, S, D), got q.shape={q.shape}, k.shape={k.shape}"
        )
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} vs {k.shape}")
    S = T * H * W
    if q.shape[2] != S:
        raise ValueError(f"q.shape[2] must equal T*H*W={S}, got {q.shape[2]}")
    if n_samples <= 0 or n_samples > S:
        raise ValueError(f"n_samples must be in (0, {S}], got {n_samples}")
    if key is None:
        key = mx.random.key(0)
    perm = mx.random.permutation(S, key=key)
    idx = perm[:n_samples]
    q_sub = mx.take(q, idx, axis=2)
    D = q.shape[3]
    scale = 1.0 / mx.sqrt(mx.array(D, dtype=q.dtype))
    logits = mx.matmul(q_sub, mx.swapaxes(k, 2, 3)) * scale
    probs = mx.softmax(logits, axis=-1)  # (B, nH, n_samples, S)
    t_flat, h_flat, w_flat = _thw_ids(T, H, W)
    t_q = mx.take(t_flat, idx)
    h_q = mx.take(h_flat, idx)
    w_q = mx.take(w_flat, idx)
    same_frame = mx.equal(mx.expand_dims(t_q, 1), mx.expand_dims(t_flat, 0))
    same_h = mx.equal(mx.expand_dims(h_q, 1), mx.expand_dims(h_flat, 0))
    same_w = mx.equal(mx.expand_dims(w_q, 1), mx.expand_dims(w_flat, 0))
    same_pos = mx.logical_and(same_h, same_w)
    same_frame_f = same_frame.astype(probs.dtype)
    same_pos_f = same_pos.astype(probs.dtype)
    mass_frame = mx.sum(probs * same_frame_f, axis=-1)  # (B, nH, n_samples)
    mass_pos = mx.sum(probs * same_pos_f, axis=-1)
    mass_frame = mx.mean(mass_frame, axis=(0, 2))
    mass_pos = mx.mean(mass_pos, axis=(0, 2))
    return mx.stack([mass_frame, mass_pos], axis=1)

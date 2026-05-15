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


def _thw_ids(T: int, H: int, W: int) -> tuple[mx.array, mx.array, mx.array]:
    """Per-token (t, h, w) coordinate vectors of length S=T*H*W (T-major).

    Mirror of ``video_masks._thw_coords``. Kept private here so ``profile``
    has no internal dependency on ``video_masks``.
    """
    t_idx = mx.arange(T).reshape(T, 1, 1)
    h_idx = mx.arange(H).reshape(1, H, 1)
    w_idx = mx.arange(W).reshape(1, 1, W)
    t_flat = mx.broadcast_to(t_idx, (T, H, W)).reshape(-1)
    h_flat = mx.broadcast_to(h_idx, (T, H, W)).reshape(-1)
    w_flat = mx.broadcast_to(w_idx, (T, H, W)).reshape(-1)
    return t_flat, h_flat, w_flat


def _validate_thw(T: int, H: int, W: int) -> None:
    """Same contract as ``video_masks._validate_thw``."""
    if T <= 0 or H <= 0 or W <= 0:
        raise ValueError(f"T, H, W must be positive, got T={T}, H={H}, W={W}")


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

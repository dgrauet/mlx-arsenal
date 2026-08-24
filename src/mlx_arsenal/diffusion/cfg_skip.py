"""CFG-skip (Attention Sharing across CFG, DiTFastAttn ASC).

Profiles per-head similarity between the conditional and unconditional CFG
branches, builds a static schedule, and applies that schedule at runtime to
skip the unconditional branch on heads where cond and uncond outputs are
near-identical.

Two metrics:

* ``cosine`` — dot/norm of flattened ``(B, S, D)`` per head, in ``[-1, 1]``.
  Skip when score is **at least** the threshold. Default; matches the
  DiTFastAttn ASC literature.
* ``relative_l1`` — ``mean(|c - u|) / mean(|c|)`` per head, ≥ 0. Skip when
  score is **at most** the threshold. Same family as
  :class:`~mlx_arsenal.diffusion.TeaCacheController` and the attention
  output caches, so existing thresholds carry over.

The runtime apply step (:meth:`CFGSkipController.apply`) bundles a
:func:`mlx_arsenal.diffusion.splice_heads` call so callers do not have to
import the splice helper directly.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx

from .attention_cache import splice_heads

Metric = Literal["cosine", "relative_l1"]

_ALLOWED_METRICS: tuple[Metric, ...] = ("cosine", "relative_l1")


def _validate_metric(metric: str) -> None:
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {_ALLOWED_METRICS}, got {metric!r}")


def cfg_head_similarity(
    cond: mx.array,
    uncond: mx.array,
    *,
    metric: Metric = "cosine",
) -> mx.array:
    """Per-head similarity between conditional and unconditional outputs.

    Args:
        cond: ``(B, num_heads, S, D)`` conditional attention output.
        uncond: ``(B, num_heads, S, D)`` unconditional output, same shape.
        metric: ``"cosine"`` (default, ``[-1, 1]``) or ``"relative_l1"``
            (``≥ 0``).

    Returns:
        ``(num_heads,)`` float array. Reduced over batch, sequence, and
        feature axes.
    """
    _validate_metric(metric)
    if cond.ndim < 4:
        raise ValueError(f"cond must be at least 4D, got shape {cond.shape}")
    if cond.shape != uncond.shape:
        raise ValueError(
            f"cond and uncond must have matching shapes, got {cond.shape} vs {uncond.shape}"
        )
    num_heads = cond.shape[1]
    # Flatten to (num_heads, K) where K = B * S * D (and any trailing axes).
    # Reduce in float32: fp16 accumulation overflows for realistic
    # activation magnitudes (sum of B*S*D values).
    cond_h = mx.swapaxes(cond, 0, 1).reshape(num_heads, -1).astype(mx.float32)
    uncond_h = mx.swapaxes(uncond, 0, 1).reshape(num_heads, -1).astype(mx.float32)
    if metric == "cosine":
        dot = mx.sum(cond_h * uncond_h, axis=1)
        c_norm = mx.sqrt(mx.sum(cond_h * cond_h, axis=1))
        u_norm = mx.sqrt(mx.sum(uncond_h * uncond_h, axis=1))
        denom = c_norm * u_norm
        # Zero-norm heads → similarity = 0 (no skip under either threshold convention).
        zero = mx.equal(denom, 0.0)
        safe_denom = mx.where(zero, mx.ones_like(denom), denom)
        sim = mx.where(zero, mx.zeros_like(dot), dot / safe_denom)
        return sim
    # relative_l1
    diff = mx.mean(mx.abs(cond_h - uncond_h), axis=1)
    base = mx.mean(mx.abs(cond_h), axis=1)
    zero = mx.equal(base, 0.0)
    same = mx.equal(diff, 0.0)
    # 0/0 → 0.0; nonzero/0 → +inf
    safe_base = mx.where(zero, mx.ones_like(base), base)
    raw = diff / safe_base
    inf = mx.full(raw.shape, float("inf"), dtype=raw.dtype)
    zero_score = mx.zeros_like(raw)
    return mx.where(zero, mx.where(same, zero_score, inf), raw)


def cfg_skip_mask(
    scores: mx.array,
    threshold: float,
    *,
    metric: Metric = "cosine",
) -> mx.array:
    """Convert similarity scores to a skip-uncond mask.

    Args:
        scores: ``(num_heads,)`` or ``(num_blocks, num_heads)`` scores from
            :func:`cfg_head_similarity` or
            :class:`CFGSimilarityProfiler.scores`.
        threshold: Cut-off. With ``cosine``, skip when ``scores >= threshold``;
            with ``relative_l1``, skip when ``scores <= threshold``.
        metric: Which inversion to apply.

    Returns:
        Bool array with the same shape as ``scores``. ``True`` means "this
        head can skip the uncond branch".
    """
    _validate_metric(metric)
    if metric == "cosine":
        return mx.greater_equal(scores, threshold)
    return mx.less_equal(scores, threshold)


class CFGSimilarityProfiler:
    """Per-(block, head) running-mean similarity accumulator.

    Records the per-head similarity of cond and uncond attention outputs
    during a warmup pass and produces a static ``(num_blocks, num_heads)``
    skip schedule.
    """

    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        *,
        metric: Metric = "cosine",
    ):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        _validate_metric(metric)
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.metric: Metric = metric
        self._sum_scores = mx.zeros((num_blocks, num_heads), dtype=mx.float32)
        self._counts = mx.zeros((num_blocks,), dtype=mx.int32)

    def reset(self) -> None:
        """Clear accumulated scores and counts. Call before a new warmup pass."""
        self._sum_scores = mx.zeros((self.num_blocks, self.num_heads), dtype=mx.float32)
        self._counts = mx.zeros((self.num_blocks,), dtype=mx.int32)

    def record(self, block_idx: int, cond: mx.array, uncond: mx.array) -> None:
        """Accumulate per-head cond/uncond similarity for block ``block_idx``."""
        if block_idx < 0 or block_idx >= self.num_blocks:
            raise ValueError(f"block_idx must be in [0, {self.num_blocks}), got {block_idx}")
        if cond.ndim < 4 or cond.shape[1] != self.num_heads:
            raise ValueError(
                f"cond must have head axis 1 of size {self.num_heads}, got shape {cond.shape}"
            )
        if cond.shape != uncond.shape:
            raise ValueError(
                f"cond and uncond shapes must match, got {cond.shape} vs {uncond.shape}"
            )
        sim = cfg_head_similarity(cond, uncond, metric=self.metric)  # (num_heads,)
        # Update only row block_idx via gather + scatter-like recomposition.
        prev_row = self._sum_scores[block_idx]
        new_row = (prev_row + sim).astype(mx.float32)
        # Reassemble _sum_scores with the updated row.
        before = self._sum_scores[:block_idx]
        after = self._sum_scores[block_idx + 1 :]
        self._sum_scores = mx.concatenate(
            [before, new_row.reshape(1, self.num_heads), after], axis=0
        )
        # Update count.
        cnt_before = self._counts[:block_idx]
        cnt_after = self._counts[block_idx + 1 :]
        new_cnt = mx.array([self._counts[block_idx].item() + 1], dtype=mx.int32)
        self._counts = mx.concatenate([cnt_before, new_cnt, cnt_after], axis=0)

    @property
    def scores(self) -> mx.array:
        """``(num_blocks, num_heads)`` running mean. Zero-count blocks → 0.0."""
        counts_f = self._counts.astype(mx.float32).reshape(self.num_blocks, 1)
        zero = mx.equal(counts_f, 0.0)
        safe_counts = mx.where(zero, mx.ones_like(counts_f), counts_f)
        return mx.where(
            mx.broadcast_to(zero, self._sum_scores.shape),
            mx.zeros_like(self._sum_scores),
            self._sum_scores / safe_counts,
        )

    @property
    def call_counts(self) -> mx.array:
        """``(num_blocks,)`` int32 count of :meth:`record` calls per block."""
        return self._counts

    def build_skip_schedule(self, threshold: float) -> mx.array:
        """Threshold :attr:`scores` into a ``(num_blocks, num_heads)`` skip mask."""
        return cfg_skip_mask(self.scores, threshold=threshold, metric=self.metric)


class CFGSkipController:
    """Wraps a static ``(num_blocks, num_heads)`` bool skip schedule.

    ``True`` at ``(b, h)`` means: for block ``b``, head ``h`` skips the
    unconditional branch and reuses the conditional output. ``False`` means
    the head computes uncond normally.
    """

    def __init__(self, schedule: mx.array):
        if schedule.ndim != 2:
            raise ValueError(
                f"schedule must be 2D (num_blocks, num_heads), got shape {schedule.shape}"
            )
        if schedule.dtype != mx.bool_:
            raise ValueError(f"schedule must be bool dtype, got {schedule.dtype}")
        self._schedule = schedule

    @classmethod
    def from_profiler(
        cls,
        profiler: CFGSimilarityProfiler,
        threshold: float,
    ) -> "CFGSkipController":  # noqa: UP037
        """Build a controller from a profiler by thresholding its scores."""
        return cls(profiler.build_skip_schedule(threshold))

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks in the schedule."""
        return int(self._schedule.shape[0])

    @property
    def num_heads(self) -> int:
        """Number of attention heads in the schedule."""
        return int(self._schedule.shape[1])

    def should_skip_uncond(self, block_idx: int) -> mx.array:
        """``(num_heads,)`` bool mask — ``True`` heads skip uncond for this block."""
        self._check_block(block_idx)
        return self._schedule[block_idx]

    def apply(
        self,
        block_idx: int,
        cond_output: mx.array,
        uncond_output: mx.array,
    ) -> mx.array:
        """Apply the cached schedule to splice cond/uncond outputs (wraps :func:`splice_heads`)."""
        self._check_block(block_idx)
        if cond_output.shape != uncond_output.shape:
            raise ValueError(
                f"cond_output and uncond_output must match in shape, "
                f"got {cond_output.shape} vs {uncond_output.shape}"
            )
        return splice_heads(cond_output, uncond_output, self._schedule[block_idx])

    def _check_block(self, block_idx: int) -> None:
        if block_idx < 0 or block_idx >= self.num_blocks:
            raise ValueError(f"block_idx must be in [0, {self.num_blocks}), got {block_idx}")

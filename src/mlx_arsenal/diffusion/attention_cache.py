"""Attention output cache (AST-style) for diffusion transformers.

Caches attention sub-layer output across denoising steps and reuses it on
the next step when the input has barely changed. Two granularities:

* :class:`PerLayerAttentionCache` — scalar similarity, one decision per layer
  per step. Simpler, mirrors :class:`mlx_arsenal.diffusion.TeaCacheController`
  but at the attention sub-layer instead of a whole transformer block.

Decision rule, at step ``i``:

1. If ``i == 0`` or ``i == num_steps - 1`` → recompute (boundary).
2. If ``mean(abs(prev_input))`` is zero → recompute (degenerate).
3. If ``mean(abs(input - prev_input)) / mean(abs(prev_input)) >= rel_l1_thresh``
   → recompute.

References:
    DiTFastAttn — *Attention Sharing across Timesteps* (AST).
"""

from __future__ import annotations

import mlx.core as mx

from .._scalar import item_float


class PerLayerAttentionCache:
    """Stateful per-layer attention output cache."""

    def __init__(self, num_steps: int, rel_l1_thresh: float):
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")
        if rel_l1_thresh <= 0:
            raise ValueError(f"rel_l1_thresh must be > 0, got {rel_l1_thresh}")
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self._prev_input: mx.array | None = None
        self._prev_summary: float | None = None
        self._prev_output: mx.array | None = None

    def reset(self) -> None:
        """Clear all state. Call at the start of each new generation."""
        self._prev_input = None
        self._prev_summary = None
        self._prev_output = None

    def should_compute(self, step_index: int, attn_input: mx.array) -> bool:
        """Decide whether to recompute attention at ``step_index``.

        Side-effects: advances the stored previous input and summary. Must be
        called once per step in order.
        """
        self._check_step(step_index)
        if step_index == 0 or step_index == self.num_steps - 1:
            self._prev_input = attn_input
            self._prev_summary = item_float(mx.mean(mx.abs(attn_input)))
            return True
        prev = self._prev_input
        prev_summary = self._prev_summary
        if prev is None or prev_summary is None:
            raise RuntimeError(
                "should_compute called for a non-boundary step before step 0 — "
                "boundary steps must run first to seed the cache."
            )
        if prev_summary == 0.0:
            self._prev_input = attn_input
            self._prev_summary = item_float(mx.mean(mx.abs(attn_input)))
            return True
        delta = item_float(mx.mean(mx.abs(attn_input - prev))) / prev_summary
        self._prev_input = attn_input
        self._prev_summary = item_float(mx.mean(mx.abs(attn_input)))
        return delta >= self.rel_l1_thresh

    def should_compute_from_summary(self, step_index: int, summary: float) -> bool:
        """Decide using a caller-supplied scalar summary instead of a tensor.

        ``summary`` is the analogue of
        ``mean(abs(input - prev_input)) / mean(abs(prev_input))`` — the caller
        has already done the math.

        Do not interleave with :meth:`should_compute` within a single
        denoising run: the two methods write semantically different values
        into the internal previous-summary slot. Pick one mode per run.
        """
        self._check_step(step_index)
        if step_index == 0 or step_index == self.num_steps - 1:
            self._prev_summary = summary
            return True
        if self._prev_summary is None:
            raise RuntimeError(
                "should_compute_from_summary called for a non-boundary step "
                "before step 0 — boundary steps must run first to seed."
            )
        out = summary >= self.rel_l1_thresh
        self._prev_summary = summary
        return out

    def cache_output(self, output: mx.array) -> None:
        """Store the attention output from the just-computed step for reuse on skip."""
        self._prev_output = output

    @property
    def previous_output(self) -> mx.array:
        """Last cached attention output. Raises before the first ``cache_output`` call."""
        if self._prev_output is None:
            raise RuntimeError(
                "No output cached yet — call cache_output() after a computed "
                "step before reading previous_output."
            )
        return self._prev_output

    def _check_step(self, step_index: int) -> None:
        if step_index < 0 or step_index >= self.num_steps:
            raise ValueError(f"step_index must be in [0, {self.num_steps}), got {step_index}")


class PerHeadAttentionCache:
    """Stateful per-head attention output cache.

    Returns a ``(num_heads,)`` bool decision per step. Inputs are assumed to
    have the head axis at position 1 — i.e. shape ``(B, num_heads, ...)``.
    """

    def __init__(self, num_heads: int, num_steps: int, rel_l1_thresh: float):
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")
        if rel_l1_thresh <= 0:
            raise ValueError(f"rel_l1_thresh must be > 0, got {rel_l1_thresh}")
        self.num_heads = num_heads
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self._prev_input: mx.array | None = None
        self._prev_summary: mx.array | None = None
        self._prev_output: mx.array | None = None

    def reset(self) -> None:
        """Clear all state. Call at the start of each new generation."""
        self._prev_input = None
        self._prev_summary = None
        self._prev_output = None

    def should_compute(self, step_index: int, attn_input: mx.array) -> mx.array:
        """Per-head decide whether to recompute attention at ``step_index``.

        Side-effects: advances the stored previous input and per-head summary.
        Must be called once per step in order.
        """
        self._check_step(step_index)
        head_summary = self._reduce_per_head(attn_input)
        if step_index == 0 or step_index == self.num_steps - 1:
            self._prev_input = attn_input
            self._prev_summary = head_summary
            return mx.ones((self.num_heads,), dtype=mx.bool_)
        prev = self._prev_input
        prev_summary = self._prev_summary
        if prev is None or prev_summary is None:
            raise RuntimeError(
                "should_compute called for a non-boundary step before step 0 — "
                "boundary steps must run first to seed the cache."
            )
        diff = self._reduce_per_head(attn_input - prev)
        zero = mx.equal(prev_summary, 0.0)
        safe_prev = mx.where(zero, mx.ones_like(prev_summary), prev_summary)
        delta = diff / safe_prev
        recompute = mx.logical_or(zero, mx.greater_equal(delta, self.rel_l1_thresh))
        self._prev_input = attn_input
        self._prev_summary = head_summary
        return recompute

    def should_compute_from_summary(self, step_index: int, summary: mx.array) -> mx.array:
        """Decide per head using a caller-supplied ``(num_heads,)`` summary.

        ``summary[h]`` is the analogue of the per-head delta ratio. Do not
        interleave with :meth:`should_compute` within a single denoising
        run: the two methods write semantically different values into the
        internal previous-summary slot. Pick one mode per run.
        """
        self._check_step(step_index)
        if summary.ndim != 1 or summary.shape[0] != self.num_heads:
            raise ValueError(
                f"summary must be 1D of length num_heads={self.num_heads}, "
                f"got shape {summary.shape}"
            )
        if step_index == 0 or step_index == self.num_steps - 1:
            self._prev_summary = summary
            return mx.ones((self.num_heads,), dtype=mx.bool_)
        if self._prev_summary is None:
            raise RuntimeError(
                "should_compute_from_summary called for a non-boundary step "
                "before step 0 — boundary steps must run first to seed."
            )
        out = mx.greater_equal(summary, self.rel_l1_thresh)
        self._prev_summary = summary
        return out

    def cache_output(self, output: mx.array) -> None:
        """Store the full ``(B, num_heads, ...)`` attention output for per-head splicing on skip."""
        self._prev_output = output

    @property
    def previous_output(self) -> mx.array:
        """Last cached attention output. Raises before the first ``cache_output`` call."""
        if self._prev_output is None:
            raise RuntimeError(
                "No output cached yet — call cache_output() after a computed "
                "step before reading previous_output."
            )
        return self._prev_output

    def _check_step(self, step_index: int) -> None:
        if step_index < 0 or step_index >= self.num_steps:
            raise ValueError(f"step_index must be in [0, {self.num_steps}), got {step_index}")

    def _reduce_per_head(self, x: mx.array) -> mx.array:
        if x.ndim < 2 or x.shape[1] != self.num_heads:
            raise ValueError(
                f"input must have head axis 1 of size {self.num_heads}, got shape {x.shape}"
            )
        abs_x = mx.abs(x)
        reduce_axes = tuple(i for i in range(x.ndim) if i != 1)
        return mx.mean(abs_x, axis=reduce_axes)


def splice_heads(
    new_output: mx.array,
    cached_output: mx.array,
    recompute_mask: mx.array,
) -> mx.array:
    """Per-head select-between-tensors.

    Where ``recompute_mask[h]`` is ``True``, take ``new_output[:, h]``;
    where ``False``, take ``cached_output[:, h]``. The head axis is assumed
    to be axis 1.

    Args:
        new_output: ``(B, num_heads, ...)`` newly computed attention output.
        cached_output: ``(B, num_heads, ...)`` previously cached output, same
            shape as ``new_output``.
        recompute_mask: 1D bool array of length ``num_heads``.

    Returns:
        Spliced tensor with the same shape as ``new_output``.
    """
    if new_output.shape != cached_output.shape:
        raise ValueError(
            f"new_output and cached_output must match in shape, "
            f"got {new_output.shape} vs {cached_output.shape}"
        )
    if recompute_mask.ndim != 1:
        raise ValueError(f"recompute_mask must be 1D, got shape {recompute_mask.shape}")
    if recompute_mask.dtype != mx.bool_:
        raise ValueError(f"recompute_mask must be bool, got dtype {recompute_mask.dtype}")
    if new_output.ndim < 2 or recompute_mask.shape[0] != new_output.shape[1]:
        raise ValueError(
            f"recompute_mask length must equal head axis (axis 1) of new_output, "
            f"got mask {recompute_mask.shape} vs new_output {new_output.shape}"
        )
    shape = [1] * new_output.ndim
    shape[1] = recompute_mask.shape[0]
    mask_bc = recompute_mask.reshape(tuple(shape))
    return mx.where(mask_bc, new_output, cached_output)

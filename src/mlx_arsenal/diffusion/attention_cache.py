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
        self._prev_input = None
        self._prev_summary = None
        self._prev_output = None

    def should_compute(self, step_index: int, attn_input: mx.array) -> bool:
        self._check_step(step_index)
        if step_index == 0 or step_index == self.num_steps - 1:
            self._prev_input = attn_input
            self._prev_summary = float(mx.mean(mx.abs(attn_input)).item())
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
            self._prev_summary = float(mx.mean(mx.abs(attn_input)).item())
            return True
        delta = float(mx.mean(mx.abs(attn_input - prev)).item()) / prev_summary
        self._prev_input = attn_input
        self._prev_summary = float(mx.mean(mx.abs(attn_input)).item())
        return delta >= self.rel_l1_thresh

    def should_compute_from_summary(self, step_index: int, summary: float) -> bool:
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
        self._prev_output = output

    @property
    def previous_output(self) -> mx.array:
        if self._prev_output is None:
            raise RuntimeError(
                "No output cached yet — call cache_output() after a computed "
                "step before reading previous_output."
            )
        return self._prev_output

    def _check_step(self, step_index: int) -> None:
        if step_index < 0 or step_index >= self.num_steps:
            raise ValueError(f"step_index must be in [0, {self.num_steps}), got {step_index}")

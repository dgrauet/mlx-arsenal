"""Head-wise temporal reuse of sparse attention masks (HEART).

Content-dependent sparse masks (e.g.
:func:`~mlx_arsenal.attention.top_p_block_mask` over
:func:`~mlx_arsenal.attention.antidiagonal_block_scores`) change slowly
across denoising steps, and some heads change much less than others. HEART
(Temporal Mask Reuse, arXiv 2605.14513) keeps, per head, an *anchor*: the
token-mean of Q and K at the step where the head's mask was last rebuilt.
At every step the head's drift is the L1 distance between the current
pooled Q/K and its anchor; heads whose drift exceeds ``δ`` rebuild their
mask and move their anchor, the others reuse the anchored mask. Because the
anchor only moves on refresh, small drifts accumulate until they cross
``δ``.

Caller-side, as in the paper: the dense warm-up steps (simply do not use the
cache during them), one cache per layer and per CFG branch, the choice of
``δ``, and the mask predictor itself.
"""

from __future__ import annotations

import mlx.core as mx

_NORM_FLOOR = 1e-12


def pooled_qk(q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
    """Token-mean of queries and keys per head (HEART's drift summary).

    Args:
        q: ``(B, H, Nq, D)`` queries.
        k: ``(B, H, Nk, D)`` keys, same ``(B, H, ·, D)``.

    Returns:
        ``(qbar, kbar)``, each ``(B, H, D)`` float32.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must have rank 4, got {q.ndim} and {k.ndim}")
    if q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q and k shapes are incompatible: {tuple(q.shape)} vs {tuple(k.shape)}")
    return mx.mean(q.astype(mx.float32), axis=2), mx.mean(k.astype(mx.float32), axis=2)


def qk_drift(
    qbar_a: mx.array,
    kbar_a: mx.array,
    qbar_b: mx.array,
    kbar_b: mx.array,
    *,
    relative: bool = False,
) -> mx.array:
    """Per-head drift between two pooled Q/K summaries.

    ``‖q̄_a − q̄_b‖₁ + ‖k̄_a − k̄_b‖₁`` over the feature axis (HEART Eq. 4).
    This raw distance scales with the magnitude of Q and K, so HEART's
    threshold is model-specific (8 or 30 in the paper). ``relative=True``
    divides by ``‖q̄_a‖₁ + ‖k̄_a‖₁`` (the reference summary), which makes the
    drift scale-free — an extension, not part of the paper.

    Args:
        qbar_a, kbar_a: ``(B, H, D)`` reference (anchor) summaries.
        qbar_b, kbar_b: ``(B, H, D)`` current summaries.
        relative: Normalize by the reference summary's L1 norm.

    Returns:
        ``(B, H)`` float32 drift.
    """
    shapes = {tuple(x.shape) for x in (qbar_a, kbar_a, qbar_b, kbar_b)}
    if len(shapes) != 1 or qbar_a.ndim != 3:
        raise ValueError(f"all summaries must share one (B, H, D) shape, got {sorted(shapes)}")
    d = mx.sum(mx.abs(qbar_a - qbar_b), axis=-1) + mx.sum(mx.abs(kbar_a - kbar_b), axis=-1)
    if relative:
        ref = mx.sum(mx.abs(qbar_a), axis=-1) + mx.sum(mx.abs(kbar_a), axis=-1)
        d = d / mx.maximum(ref, _NORM_FLOOR)
    return d.astype(mx.float32)


class HeadMaskCache:
    """Per-head sparse-mask cache refreshed on pooled Q/K drift (HEART TMR).

    Two-phase protocol, once per denoising step and per layer (and per CFG
    branch — use one cache each):

    ```python
    refresh = cache.should_refresh(q, k)             # (B, H) bool
    new_mask = predict(q, k)                         # caller's predictor
    mask = cache.update(new_mask, refresh)           # merged per head
    ```

    The first call refreshes every head. Afterwards a head refreshes when the
    drift of its pooled Q/K from its anchor (the step of its last refresh)
    exceeds ``delta``; refreshed heads take the new mask and move their
    anchor, the others keep their cached mask. The predictor may compute the
    new mask for every head (simplest, same cost in dense MLX) or only for
    refreshed ones — values for reused heads are ignored.

    ``layer_gate=(low, high)`` applies HEART's layer-level override per batch
    row: if the fraction ``r`` of heads marked for refresh is below ``low``
    no head refreshes, above ``high`` every head does (the paper uses
    ``(0.4, 0.8)`` to amortize per-layer mask-construction overhead).

    Args:
        delta: Drift threshold ``δ >= 0``; a head refreshes when its drift is
            strictly greater. Raw drift is model-scale dependent (the paper
            uses 8 or 30); see ``relative``.
        relative: Use :func:`qk_drift` with ``relative=True`` (drift divided
            by the anchor's L1 norm), making ``delta`` scale-free.
        layer_gate: Optional ``(low, high)`` with ``0 <= low <= high <= 1``.
    """

    def __init__(
        self,
        delta: float,
        *,
        relative: bool = False,
        layer_gate: tuple[float, float] | None = None,
    ):
        if not delta >= 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        if layer_gate is not None:
            low, high = layer_gate
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(f"layer_gate must satisfy 0 <= low <= high <= 1, got {layer_gate}")
        self.delta = delta
        self.relative = relative
        self.layer_gate = layer_gate
        self.reset()

    def reset(self) -> None:
        """Clear anchors and masks. Call at the start of each new generation."""
        self._qbar: mx.array | None = None
        self._kbar: mx.array | None = None
        self._mask: mx.array | None = None
        self._pending: tuple[mx.array, mx.array] | None = None

    @property
    def mask(self) -> mx.array:
        """The current merged mask (after the last :meth:`update`)."""
        if self._mask is None:
            raise RuntimeError("no mask cached yet: call should_refresh then update first")
        return self._mask

    def should_refresh(self, q: mx.array, k: mx.array) -> mx.array:
        """Decide which heads rebuild their mask at this step.

        Args:
            q: ``(B, H, Nq, D)`` queries of this step (the tensors the mask
                predictor sees).
            k: ``(B, H, Nk, D)`` keys.

        Returns:
            ``(B, H)`` bool, True where the head must use a freshly predicted
            mask. Must be followed by :meth:`update` before the next call.
        """
        if self._pending is not None:
            raise RuntimeError("should_refresh called twice: call update with the new mask first")
        qbar, kbar = pooled_qk(q, k)
        if self._qbar is None or self._kbar is None:
            refresh = mx.ones(qbar.shape[:2], dtype=mx.bool_)
        else:
            if tuple(qbar.shape) != tuple(self._qbar.shape):
                raise ValueError(
                    f"pooled (B, H, D) shape changed: {tuple(qbar.shape)} vs cached "
                    f"{tuple(self._qbar.shape)}; call reset() for a new configuration"
                )
            drift = qk_drift(self._qbar, self._kbar, qbar, kbar, relative=self.relative)
            refresh = drift > self.delta
            if self.layer_gate is not None:
                low, high = self.layer_gate
                r = mx.mean(refresh.astype(mx.float32), axis=-1, keepdims=True)
                refresh = mx.where(r < low, False, mx.where(r > high, True, refresh))
        self._pending = (qbar, kbar)
        return refresh

    def update(self, new_mask: mx.array, refresh: mx.array) -> mx.array:
        """Merge freshly predicted masks into the cache and move anchors.

        Args:
            new_mask: Mask with leading ``(B, H)`` axes (any trailing shape,
                e.g. a ``(B, H, Cq, Ck)`` block mask). Only refreshed heads'
                entries are used.
            refresh: The ``(B, H)`` bool returned by :meth:`should_refresh`.

        Returns:
            The merged mask, same shape and dtype as ``new_mask``.
        """
        if self._pending is None:
            raise RuntimeError("update called without a pending should_refresh")
        qbar, kbar = self._pending
        if refresh.dtype != mx.bool_ or tuple(refresh.shape) != tuple(qbar.shape[:2]):
            raise ValueError(
                f"refresh must be a bool array of shape {tuple(qbar.shape[:2])}, "
                f"got {refresh.dtype} {tuple(refresh.shape)}"
            )
        if tuple(new_mask.shape[:2]) != tuple(qbar.shape[:2]):
            raise ValueError(
                f"new_mask must have leading shape {tuple(qbar.shape[:2])}, "
                f"got {tuple(new_mask.shape)}"
            )
        if self._mask is None:
            if not mx.all(refresh).item():
                raise RuntimeError("first update must refresh every head (no cached mask yet)")
            merged = new_mask
        else:
            if tuple(new_mask.shape) != tuple(self._mask.shape):
                raise ValueError(
                    f"new_mask shape {tuple(new_mask.shape)} differs from the cached mask "
                    f"{tuple(self._mask.shape)}; call reset() for a new configuration"
                )
            sel = refresh.reshape(*refresh.shape, *([1] * (new_mask.ndim - 2)))
            merged = mx.where(sel, new_mask, self._mask).astype(new_mask.dtype)
        sel_d = mx.expand_dims(refresh, -1)
        self._qbar = qbar if self._qbar is None else mx.where(sel_d, qbar, self._qbar)
        self._kbar = kbar if self._kbar is None else mx.where(sel_d, kbar, self._kbar)
        self._mask = merged
        self._pending = None
        return merged

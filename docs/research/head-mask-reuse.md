# Dynamic block masks and head-wise mask reuse

Research notes on content-dependent sparse attention for video DiTs and
on reusing those masks across denoising steps, and how `mlx-arsenal`
exposes them. This page documents a pattern; it does not open an ADR. It
extends the sparse-attention work of
[ADR-0001](../adr/0001-sparse-attention-roadmap.md) and the
[sparse-block compensation](sparse-block-compensation.md) note.

## Static vs dynamic masks

The masks shipped with ADR-0001 (sliding tile, radial, spatial/temporal)
are fixed patterns: they do not depend on the content and never change
across steps. Dynamic predictors instead score every (query block, key
block) pair from the actual Q and K and keep the blocks carrying most of
the attention mass, per head and per step.

### XAttention — anti-diagonal block scoring

*XAttention: Block Sparse Attention with Antidiagonal Scoring* (Xu et al.,
arXiv 2503.16428). Each `stride × stride` sub-block of `QKᵀ` is summarized
by the sum along its anti-diagonal, which touches every query and key of
the sub-block once at `1/stride` of the cost. Strided logits are
softmaxed per row, summed per `block × block` tile, and each query block
keeps the smallest set of key blocks whose mass reaches a fraction `τ` of
its total (top-p). HEART runs it with block 128, stride 16, `τ = 0.9`.

### HEART — head-wise temporal mask reuse

*HEART: Exploiting Head Heterogeneity in Sparse Attention for Video
Diffusion* (arXiv 2605.14513, no public code). Two parts:

- **Temporal Mask Reuse (TMR).** Per head, keep an *anchor*: the token-mean
  of Q and K (`q̄`, `k̄ ∈ R^D`) at the step where the head's mask was last
  built, and that mask. At each step the drift is
  `‖q̄_anchor − q̄_t‖₁ + ‖k̄_anchor − k̄_t‖₁`. If it exceeds a global `δ`
  (8 or 30 depending on model and base method), rebuild the mask and move
  the anchor; otherwise reuse. A layer-level override refreshes nothing when
  under 40 % of a layer's heads ask for it and everything above 80 %. The
  first 5 of 50 steps run dense.
- **Error-guided Budgeted Calibration (EBC).** An offline per-head choice of
  `τ ∈ {0.85, 0.90, 0.95}` from single-head sparsification probes (a
  weighted 3D-FFT error against dense velocities), solved as an integer
  program under a global sparsity budget.

## What `mlx-arsenal` ships

| Function | Module | Role |
|---|---|---|
| `antidiagonal_block_scores` | `attention` | XAttention block-mass estimate, rows normalized to 1 |
| `top_p_block_mask` | `attention` | per-row top-p block selection, scalar or per-head `τ`, additive block mask |
| `pooled_qk`, `qk_drift` | `diffusion` | HEART's pooled summary and L1 drift (optionally relative) |
| `HeadMaskCache` | `diffusion` | two-phase per-head mask cache with anchors and optional layer gate |

The block mask plugs straight into `centroid_compensated_attention` with
`labels = mx.arange(N) // block_size`, so skipped blocks can be
compensated instead of dropped.

## Recipe

One cache per layer and per CFG branch; skip it during the dense warm-up
steps. The test suite executes this exact block.

<!-- reference-recipe -->
```python
import mlx.core as mx

from mlx_arsenal.attention import (
    antidiagonal_block_scores,
    centroid_compensated_attention,
    top_p_block_mask,
)
from mlx_arsenal.diffusion import HeadMaskCache


def sparse_attention(q, k, v, cache: HeadMaskCache, *, block_size=128, stride=16, tau=0.9):
    """One layer, one step: reuse or rebuild each head's block mask, then attend."""
    refresh = cache.should_refresh(q, k)  # all True on the first call
    if mx.any(refresh).item():
        scores = antidiagonal_block_scores(q, k, block_size=block_size, stride=stride)
        new_mask = top_p_block_mask(scores, tau)
    else:
        new_mask = cache.mask  # every head reuses: skip the predictor
    block_mask = cache.update(new_mask, refresh)
    labels = mx.arange(q.shape[2]) // block_size
    return centroid_compensated_attention(
        q, k, v, q_labels=labels, k_labels=labels, block_mask=block_mask
    )
```

Pass a per-head `tau` array (shape `(H,)`) to use a calibrated table such as
HEART's EBC output. For attention without compensation, expand the block
mask to tokens with `mx.repeat(mx.repeat(block_mask, block_size, -2),
block_size, -1)` and pass it to `mx.fast.scaled_dot_product_attention`.

## Dense by design

MLX has no block-sparse kernel, so attention itself runs dense. Reusing a
mask saves only the predictor (`≈ 1/stride` of the `QKᵀ` FLOPs), not the
attention.
These functions are quality tools and a reference for a future kernel: they
let a port measure what dynamic sparsity and mask reuse cost in quality.

## Deviations from the papers

- **Normalized scores.** `antidiagonal_block_scores` divides each row by its
  total (rows sum to 1); XAttention keeps raw tile sums. Top-p decisions
  are identical, since they are relative to the row total.
- **No forced key block 0.** In non-causal mode, XAttention's scatter
  re-marks key block 0 in every row as a side effect; `top_p_block_mask`
  keeps only the top-p set, with at least one block per row.
- **Relative drift option.** HEART's raw L1 drift makes `δ` depend on the
  model's Q/K scale. `relative=True` divides by the anchor's L1 norm, an
  extension not evaluated in the paper.
- **Independent implementation.** The XAttention repository ships without a
  license, so these functions are written from the paper's description; no
  code was reused.

## Out of scope, and why

- **EBC calibration**: per-model tooling (probes, video FFT, ILP solver,
  prompt sets) — the doctrine rules it out. Pass its `τ` table instead.
- **SVG2 k-means predictor**: stateful clustering across steps; any
  caller-built mask works with `HeadMaskCache`.
- **Causal XAttention options** (sink / recent blocks): not needed for
  bidirectional video DiTs.
- **Chunked scoring** inside the function. All `B·H` heads are scored at
  once, using roughly `2 · B·H · (N/stride)² · 4` bytes of transient memory
  (≈16 MB per head at `N = 32k`, `stride = 16`, but ≈14 GB for a Wan 720p
  layer with CFG). Large models should call it per head slice.
- **Padding.** Sequence lengths must be multiples of `block_size`; real
  token counts often are not (e.g. 75 600). The caller pads, and
  zero-padded keys still take softmax mass, as in XAttention's non-causal
  mode.

## References

- XAttention: <https://arxiv.org/abs/2503.16428>
- HEART: <https://arxiv.org/abs/2605.14513>

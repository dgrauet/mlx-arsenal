# ADR-0001: Sparse-attention roadmap for video DiTs

- **Status**: accepted
- **Date**: 2026-05-15
- **Stacks affected**: `python` / `mlx_arsenal.attention`, `mlx_arsenal.diffusion`

## Context

Video diffusion transformers (LTX-Video, CogVideoX, Wan2,
Hunyuan-Video, etc.) spend most of their compute in 3D self-attention
over a spatiotemporal sequence of length `S = T·H·W`. The full attention
matrix is `S²`, but recent literature (DiTFastAttn, Sparse VideoGen,
Sliding Tile Attention, Radial Attention, SVG2) shows that most heads
behave as *spatial-locality* or *temporal-locality* heads, and that
structured masks restricted to these patterns recover near-full quality
at a fraction of the compute.

`mlx-arsenal` must provide the **mid-level layer**: the reusable
primitives that let a port bring these techniques into an MLX DiT
without reimplementing them in every port. The kernel layer (Metal
block-sparse) stays out of scope — it is a project of its own.

## Decision

Ship six complementary steps across `attention/` and `diffusion/`,
merged as separate PRs (#27 to #32) into `main`:

| Step | Module | Symbols | PR |
|---|---|---|---|
| 1 | `attention.video_masks` | `spatial_only_mask`, `temporal_only_mask`, `sliding_tile_block_mask`, `sliding_tile_centered_mask`, `radial_box_mask`, `radial_gaussian_mask` | #27 |
| 2 | `attention.profile` | `Kind`, `classify`, `classify_heads_from_qk`, `classify_heads_from_probs` | #28 |
| 3 | `diffusion.attention_cache` | `PerLayerAttentionCache`, `PerHeadAttentionCache`, `splice_heads` | #29 |
| 4 | `diffusion.cfg_skip` | `cfg_head_similarity`, `cfg_skip_mask`, `CFGSimilarityProfiler`, `CFGSkipController` | #30 |
| 5 | `diffusion.window_residual` | `WindowResidualController` (3 modes) | #31 |
| 6 | `attention.permute` | `block_contiguous_permutation`, `invert_permutation` | #32 |

### Shared conventions

- **Token order:** T-major flattening (`[t0(h0w0..hHwW), t1(...), ...]`).
  Aligned with LTX, CogVideoX and `mlx_arsenal.spatial.patchify`.
- **Mask format:** additive float `0.0` / `-inf`, shape `(1, 1, S, S)`.
  Broadcasts natively to `mx.fast.scaled_dot_product_attention`.
- **Step-to-step similarity metric:** relative-L1
  (`mean(|x - prev|) / mean(|prev|)`) for every step-aware controller
  (`TeaCacheController`, `PerLayerAttentionCache`,
  `PerHeadAttentionCache`, `WindowResidualController.adaptive`).
  API consistency → a threshold calibrated for TeaCache carries over
  directly.
- **Cond/uncond metric:** `cosine` by default (ASC literature),
  `relative_l1` configurable.
- **Boundary policy:** step 0 and `num_steps - 1` always force the
  refresh / recompute, in every step-aware controller.
- **Validation:** `ValueError` for invalid arguments, `RuntimeError` for
  invariant violations (uninitialized state). No `assert` in `src/`
  (Python `-O` strips them).

### Layering

Each step depends only on MLX primitives and (at most) on an earlier
step:

- Steps 4–5 reuse `splice_heads` (step 3) for the cond/uncond splice and
  the window+residual composition.
- Step 2 (profiler) consumes Q/K but depends on no other step.
- Step 6 (permutation) is fully standalone.

This independence lets a caller consume only what it needs (for
example WA-RS without ASC, or ASC without TeaCache).

## Consequences

- 22 new public symbols, +104 tests, 364 tests green in CI.
- MLX ports (LTX-2, CogVideoX, Wan2, etc.) can now compose these
  primitives instead of reimplementing them.
- The caller orchestrates the TeaCache × AttentionCache × CFGSkip ×
  WA-RS combination itself — arsenal offers no unified orchestrator.
- The kernel layer is still to be done (Metal block-sparse); the step-1
  masks and the step-6 permutation are its direct prerequisites.

## Alternatives considered

- **Bundle a single orchestrator** (a `SparseAttentionRunner` consuming
  every cache). Rejected: too opinionated about the attention function
  signature, multiplies test paths, and every port has its own
  denoising loop.
- **Implement the Metal block-sparse kernel in parallel**. Rejected: it
  is a project in its own right (a Metal-FlashAttention equivalent),
  out of scope for mid-level. The primitives here are designed to plug
  into an external kernel once one is available.
- **Offline schedule / calibration heuristics** to pick thresholds
  automatically. Rejected: too model-dependent; left to the caller.

## Exit / revision

- If a Metal block-sparse kernel becomes available (philipturner/MFA or
  other), step 6 (permutation) must be validated against that kernel's
  effective block granularity — no API change expected, but it needs an
  integration test.
- If several ports end up writing the same orchestration loop, consider
  a `SparseAttentionRunner` extracted from the most mature ports.

## References

- [DiTFastAttn — Attention Sharing across Timesteps / CFG / Spatial](https://arxiv.org/abs/2406.08552)
- [Sparse VideoGen (ICML 2025)](https://arxiv.org/abs/2502.01776)
- [Sliding Tile Attention (ICML 2025)](https://arxiv.org/abs/2502.04507)
- [Radial Attention](https://arxiv.org/abs/2506.19852)
- [SVG2 — Semantic Permutation](https://arxiv.org/abs/2505.18875)
- TeaCache (Liu et al.) — already implemented in `mlx_arsenal.diffusion.TeaCacheController`.

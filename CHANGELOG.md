# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-04-18

### Added
- `diffusion` — flow-matching diffusion primitives shared across ports of
  LTX-2, Hunyuan3D-2.1, Matrix-Game, and VideoX-Fun:
  - `get_timestep_embedding`, `TimestepEmbedding` — sinusoidal embeddings + MLP projection.
  - `get_sampling_sigmas`, `dynamic_shift_schedule` — flow-matching sigma schedules
    (fixed-shift and token-count-dependent).
  - `FlowMatchEulerScheduler` — stateful scheduler with diffusers-style
    `set_timesteps` / `step` / `add_noise`.
  - `euler_step` — stateless Euler step for ``x0``-prediction models.
  - `classifier_free_guidance` — CFG combinator.

## [0.1.0] — 2026-04-18

### Added
- Initial release on PyPI as `mlx-arsenal` (renamed from `mlx-ops`).
- `spatial` — `interpolate_nearest`, `interpolate_3d`, `avg_pool1d`, `replicate_pad`,
  `upsample_nearest/bilinear`, `pixel_shuffle/unshuffle`, `patchify/unpatchify`,
  `PatchEmbed2d/3d`.
- `layout` — `to_channels_last/first`, `channels_last` context manager,
  `convert_conv_weights`, `load_safetensors`.
- `conv` — `weight_norm`, `WeightNorm`.
- `attention` — `causal_mask`, `sliding_window_mask`.
- `norm` — `PixelNorm`, `ScaleNorm`.
- `encoding` — `FourierEmbedder`.
- `moe` — `MoEGate`, `MoELayer` with top-k routing.
- `rasterize` — Metal triangle rasterizer with z-buffering, `rasterize_triangles`,
  `interpolate`.
- `tiling` — `tiled_process`, `temporal_slice_process` for memory-efficient
  large-tensor processing.

[Unreleased]: https://github.com/dgrauet/mlx-arsenal/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dgrauet/mlx-arsenal/releases/tag/v0.2.0
[0.1.0]: https://github.com/dgrauet/mlx-arsenal/releases/tag/v0.1.0

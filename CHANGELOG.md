# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] — 2026-05-14

### Added
- `mlx_arsenal.loader` — state-dict loading primitives extracted from
  `ltx-2-mlx`. `SDOps` is an immutable chain of rename / match / k-v
  transform operations (`.with_replacement`, `.with_matching`,
  `.with_additional_allowed_keys`, `.with_kv_operation`) that lets
  ports rewrite upstream weight-key naming (diffusers / ComfyUI /
  Megatron / ...) at load time. `SafetensorsStateDictLoader` consumes
  an `SDOps` chain, merges multi-file shards, and returns a
  `StateDict` container with size + dtype accounting. Also exposes
  `read_safetensors_metadata(path)` — a pure-stdlib reader for the
  optional `__metadata__` block (no `safetensors` dep added).

## [0.5.0] — 2026-05-14

### Added
- `mlx_arsenal.ffn` — feed-forward / MLP blocks for transformer ports.
  Exposes `FeedForward` (vanilla 2-Linear with configurable activation:
  gelu / gelu_approx / silu / relu), `GatedFFN` (3-Linear gated;
  selectable gate activation), and the `GeGLU` / `SwiGLU` convenience
  wrappers. Weight keys follow the public conventions:
  `proj_in/proj_out` for `FeedForward` (LTX-style),
  `gate_proj/up_proj/down_proj` for `GatedFFN` (LLaMA / HF style).

## [0.4.0] — 2026-05-14

### Added
- `mlx_arsenal.modulation` — AdaLN modulation primitives for DiT-style
  models. Exposes `AdaLNModulation` (`SiLU → Linear → N chunks`,
  configurable chunk count for 1 / 2 / 4 / 6 / 9-param variants),
  `ScaleShiftTable` (final-layer learnable scale/shift table), and the
  `modulate()` / `gated_residual()` functional helpers. Extracted from
  patterns shared by `ltx-2-mlx` (9-param self-attn, 2-param cross-attn,
  4-param AV) and `ernie-image-mlx` (6-param shared AdaLN).

## [0.3.0] — 2026-05-14

### Added
- `mlx_arsenal.streaming` — block streaming for low-RAM transformer
  inference on Apple Silicon. Stream block weights from memory-mapped
  safetensors into a single shared `nn.Module`, so peak resident memory
  stays at `~1 block` instead of `~num_blocks`. Exposes `BlockStreamer`
  (mmap + bind), `BlockLoraSource` (per-block LoRA A/B indexing with
  optional `key_mapper` for upstream-naming remaps), and an injected
  `lora_fuser` callable hook so quantization-aware fusion strategies
  stay out of arsenal. Extracted from the `ltx-2-mlx` port; generic
  enough for any model whose state-dict has `f"{prefix}{i}.{rest}"`
  block keys.

## [0.2.5] — 2026-05-14

### Changed
- Migrated package to `src/` layout (PYTHON_LO001).
- Adopted Intendant governance (renamed from Suzerain); `.intendant.toml`
  in advisory mode.
- Hardened CI: `ty` strict type-check, coverage config, `commitlint` for
  Conventional Commits.

### Fixed
- Resolved all `ty` diagnostics across source and tests (annotations,
  `not-None` asserts, narrow casts).

## [0.2.4] — 2026-04-27

> Note: 0.2.3 was bumped in pyproject for an unreleased pixel_shuffle fix
> branch but never published to PyPI; this release skips that number to
> avoid ambiguity with the dangling v0.2.3 tag in git.

### Added
- `diffusion.TeaCacheController` — timestep-aware residual caching for
  diffusion transformers (Liu et al., *Timestep Embedding Aware Cache*).
  Skips a transformer forward when the polyfit-rescaled L1 distance of the
  modulated input stays below a threshold, reusing the previous residual.
  Boundary steps (first/last) always compute. ``cache_residual`` /
  ``previous_residual`` accept any payload, so multi-stream / multi-pass
  models (e.g. LTX-2) can cache a tuple or dict of residuals. Coefficients
  and thresholds are model-specific and live with each model's port.

## [0.2.2] — 2026-04-18

### Fixed
- `mlx_arsenal.__version__` is now read from installed package metadata
  via `importlib.metadata`, so it stays in sync with `pyproject.toml`
  (previously hard-coded and left stale at 0.2.0 through the 0.2.1 release).

## [0.2.1] — 2026-04-18

### Changed
- Renamed `FlowMatchEulerScheduler` → `FlowMatchEulerDiscreteScheduler` to match
  the diffusers convention. No behavior change.

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

[Unreleased]: https://github.com/dgrauet/mlx-arsenal/compare/v0.2.4...HEAD
[0.2.4]: https://github.com/dgrauet/mlx-arsenal/releases/tag/v0.2.4
[0.2.2]: https://github.com/dgrauet/mlx-arsenal/releases/tag/v0.2.2
[0.2.1]: https://github.com/dgrauet/mlx-arsenal/releases/tag/v0.2.1
[0.2.0]: https://github.com/dgrauet/mlx-arsenal/releases/tag/v0.2.0
[0.1.0]: https://github.com/dgrauet/mlx-arsenal/releases/tag/v0.1.0

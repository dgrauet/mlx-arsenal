# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0](https://github.com/dgrauet/mlx-arsenal/compare/v0.8.0...v0.9.0) (2026-05-15)


### Features

* **attention:** block-contiguous token permutation (SVG2) ([#32](https://github.com/dgrauet/mlx-arsenal/issues/32)) ([eaf329e](https://github.com/dgrauet/mlx-arsenal/commit/eaf329ee9544793bb28a538b0e3a6e008b460782))
* **attention:** head-pattern profiler for video DiTs ([#28](https://github.com/dgrauet/mlx-arsenal/issues/28)) ([af784aa](https://github.com/dgrauet/mlx-arsenal/commit/af784aa7fc6f0c9a086ed8924627a4bd09078911))
* **attention:** video DiT spatiotemporal masks ([#27](https://github.com/dgrauet/mlx-arsenal/issues/27)) ([7eb93f3](https://github.com/dgrauet/mlx-arsenal/commit/7eb93f38a5734a11fe2b3019fd988c858666dbda))
* **diffusion:** attention output cache (AST-style) ([#29](https://github.com/dgrauet/mlx-arsenal/issues/29)) ([c8faf5e](https://github.com/dgrauet/mlx-arsenal/commit/c8faf5e5f25de1e7894e21385e2ed4e903859110))
* **diffusion:** cfg-skip (ASC) for video DiTs ([#30](https://github.com/dgrauet/mlx-arsenal/issues/30)) ([8f91d3d](https://github.com/dgrauet/mlx-arsenal/commit/8f91d3dc7e54bb96e130233f7597e0ce3c244f46))
* **diffusion:** window-residual controller (WA-RS) ([#31](https://github.com/dgrauet/mlx-arsenal/issues/31)) ([89eea40](https://github.com/dgrauet/mlx-arsenal/commit/89eea408705ce2802097c75258e0b9cee079c3a1))


### Reverts

* codecov coverage upload and README badge ([#21](https://github.com/dgrauet/mlx-arsenal/issues/21)) ([#22](https://github.com/dgrauet/mlx-arsenal/issues/22)) ([d5d69a8](https://github.com/dgrauet/mlx-arsenal/commit/d5d69a8afdacd8135efaefcfcd239d03bf377b36))


### Documentation

* refresh CLAUDE.md submodule list ([#26](https://github.com/dgrauet/mlx-arsenal/issues/26)) ([fd799f6](https://github.com/dgrauet/mlx-arsenal/commit/fd799f613d29899378ae6439ddaa3c5c8c924751))

## [Unreleased]

### Added — sparse-attention roadmap for video DiTs (étapes 1–6)

- `mlx_arsenal.attention` video-DiT spatiotemporal masks
  (#27): `spatial_only_mask`, `temporal_only_mask`,
  `sliding_tile_block_mask` and `sliding_tile_centered_mask` (STA),
  `radial_box_mask`, `radial_gaussian_mask`. All operate on T-major
  flattened token sequences of length `T*H*W` and return additive
  `(1, 1, S, S)` masks broadcastable into
  `mx.fast.scaled_dot_product_attention`. Convention matches LTX,
  CogVideoX, and `mlx_arsenal.spatial.patchify`.

- `mlx_arsenal.attention` head-pattern profiler (#28):
  `Kind` enum + `classify`, `classify_heads_from_qk`,
  `classify_heads_from_probs`. Returns per-head fractions of attention
  mass on same-frame / same-position keys, then converts to discrete
  labels. The `from_qk` path samples queries to avoid materializing the
  full `(S, S)` attention. Companion to the étape-1 masks for
  Sparse-VideoGen-style head selection.

- `mlx_arsenal.diffusion` attention output cache (AST,
  #29): `PerLayerAttentionCache`, `PerHeadAttentionCache`,
  `splice_heads`. Caches the attention sub-layer output across denoising
  steps and reuses it on the next step when the input has barely
  changed. Mirrors the `TeaCacheController` shape (`reset`,
  `should_compute`, `cache_output`, `previous_output`) but at the
  attention sub-layer instead of a whole transformer block.

- `mlx_arsenal.diffusion` CFG-skip / ASC (#30):
  `cfg_head_similarity`, `cfg_skip_mask`, `CFGSimilarityProfiler`,
  `CFGSkipController`. Profiles per-head cond/uncond similarity during
  warmup, builds a static skip schedule, applies it at runtime via
  `splice_heads`. Two metrics — `cosine` (literature default) and
  `relative_l1` (consistent with TeaCache / AttentionCache).

- `mlx_arsenal.diffusion.WindowResidualController` (WA-RS,
  #31). Step-aware controller for the
  `full − window` attention residual cache. Three classmethod
  constructors: `.fixed(refresh_every=K)`,
  `.scheduled(refresh_steps=[…])`, and `.adaptive(rel_l1_thresh=t)` —
  the adaptive variant reuses the same relative-L1 input-similarity
  metric as `TeaCacheController`.

- `mlx_arsenal.attention` block-contiguous token
  permutation (SVG2, #32):
  `block_contiguous_permutation`, `invert_permutation`. Sorts tokens by
  importance score so high-importance ones cluster into the first
  contiguous blocks (what block-sparse attention kernels actually need
  to realize their savings). Pair with `mx.take(x, perm, axis=…)` to
  permute Q/K/V and the inverse to unpermute the output.

### Fixed

- `docs/api/*.md` — H1 headings had a `u`-prefix typo on 11 of 15 module
  pages (`# uattention`, `# uconv`, etc.); harmonized all titles to
  Title case to match `mkdocs.yml` nav.

### Changed (internal / docs only)

- Added one-line docstrings to all 22 public methods and properties of
  the new cache, profiler, and controller classes for consistent
  mkdocstrings rendering.
- Added ADR-0001 covering the sparse-attention roadmap.
- Wired up release-please: synced `.release-please-manifest.json` to
  the current released version, added the missing workflow, and
  configured `extra-files` so pyproject version bumps on release-PR
  merge.

## [0.8.0] — 2026-05-14

### Added
- `mlx_arsenal.rope` — Rotary Position Embedding primitives.
  `rope_frequencies_1d` / `rope_frequencies_nd` compute the
  `(cos, sin)` half-angle pairs for one or many axes (CogVideoX-style
  `[t, h, w]` allocation works directly). `apply_rotary_emb` rotates
  a tensor with support for both *interleaved* (RoPE-paper / Matrix-Game)
  and *half-rotated* (HuggingFace / Llama) pair conventions.
  `meshgrid_nd` builds the integer position grids. NTK rescaling and
  position-interpolation factors are first-class. Fills the gap that
  `mx.fast.rope` (1-D interleaved only) leaves open for video and
  image diffusion ports. Model-specific variants (ERNIE-Image
  Megatron, LTX SPLIT log-spaced) stay in their ports — arsenal
  covers the standard case.

## [0.7.0] — 2026-05-14

### Added
- `mlx_arsenal.diffusion.DDIMScheduler` — deterministic DDIM scheduler
  (`eta=0`) matching the diffusers `DDIMScheduler` /
  `CogVideoXDDIMScheduler` behaviour. Supports both `epsilon` and
  `v_prediction` model outputs, `leading` / `trailing` timestep
  spacing, optional zero-terminal-SNR rescaling, `scaled_linear` /
  `linear` beta schedules. Extracted from the `VideoX-Fun-mlx` port;
  complements the existing `FlowMatchEulerDiscreteScheduler` for
  DDPM-trained models (CogVideoX, Stable Diffusion 1.x / 2.x).

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

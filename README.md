# mlx-arsenal

[![PyPI version](https://img.shields.io/pypi/v/mlx-arsenal.svg)](https://pypi.org/project/mlx-arsenal/)
[![CI](https://github.com/dgrauet/mlx-arsenal/actions/workflows/ci.yml/badge.svg)](https://github.com/dgrauet/mlx-arsenal/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/mlx-arsenal.svg)](https://pypi.org/project/mlx-arsenal/)
[![License](https://img.shields.io/pypi/l/mlx-arsenal.svg)](https://github.com/dgrauet/mlx-arsenal/blob/main/LICENSE)

Low-level operations and reusable building blocks missing from [MLX](https://github.com/ml-explore/mlx) core — the toolbox you want when porting PyTorch models to Apple Silicon.

> **Tip:** if you use Claude Code for MLX ports, the [`mlx-porting`](https://github.com/dgrauet/claude-skill-mlx-porting) skill teaches Claude to reach for `mlx-arsenal` submodules (`diffusion`, `spatial`, `attention`, `norm`, `encoding`, `moe`, `tiling`, etc.) before hand-rolling ops.

## Install

```bash
pip install mlx-arsenal
```

Or directly from source:

```bash
pip install git+https://github.com/dgrauet/mlx-arsenal.git
```

## Modules

| Module | Components | Replaces (PyTorch) |
|--------|-----------|-------------------|
| `mlx_arsenal.spatial` | `interpolate_nearest`, `interpolate_3d`, `avg_pool1d`, `replicate_pad`, `upsample_nearest/bilinear`, `pixel_shuffle/unshuffle`, `patchify/unpatchify`, `PatchEmbed2d/3d` | `F.interpolate`, `F.avg_pool1d`, `F.pad(mode="replicate")`, `F.pixel_shuffle` |
| `mlx_arsenal.layout` | `to_channels_last/first`, `channels_last` ctx manager, `convert_conv_weights`, `load_safetensors` | NCHW ↔ NHWC conversion, weight transposition |
| `mlx_arsenal.conv` | `weight_norm`, `WeightNorm` | `nn.utils.weight_norm` |
| `mlx_arsenal.attention` | `causal_mask`, `sliding_window_mask`, `spatial_only_mask`, `temporal_only_mask`, `sliding_tile_block_mask`, `sliding_tile_centered_mask`, `radial_box_mask`, `radial_gaussian_mask`, `frame_stride_diagonal_mask`, `vertical_stripe_mask`, `classify_heads_from_qk`, `classify_heads_from_probs`, `classify`, `Kind`, `block_contiguous_permutation`, `invert_permutation` | Attention mask creation (LLM + sparse video DiT), head-pattern profiler, SVG2 block-contiguous token permutation |
| `mlx_arsenal.norm` | `PixelNorm`, `ScaleNorm` | Custom normalization layers |
| `mlx_arsenal.encoding` | `FourierEmbedder` | Sinusoidal positional encoding |
| `mlx_arsenal.diffusion` | `get_timestep_embedding`, `TimestepEmbedding`, `get_sampling_sigmas`, `dynamic_shift_schedule`, `FlowMatchEulerDiscreteScheduler`, `DDIMScheduler`, `euler_step`, `classifier_free_guidance`, `TeaCacheController`, `PerLayerAttentionCache`, `PerHeadAttentionCache`, `splice_heads`, `cfg_head_similarity`, `cfg_skip_mask`, `CFGSimilarityProfiler`, `CFGSkipController`, `WindowResidualController`, `VerifiedFeatureCache`, `geometric_threshold` | Flow-matching + DDIM diffusion primitives, TeaCache, AST attention caches, ASC cond/uncond skip, WA-RS residual sharing |
| `mlx_arsenal.moe` | `MoEGate`, `MoELayer` | Top-k mixture-of-experts dispatch |
| `mlx_arsenal.rasterize` | `rasterize_triangles`, `interpolate` | Tile-binned triangle rasterization with exact fixed-point coverage (Metal kernels) |
| `mlx_arsenal.tiling` | `tiled_process`, `temporal_slice_process` | Memory-efficient large tensor processing |
| `mlx_arsenal.streaming` | `BlockStreamer`, `BlockLoraSource`, `LoraFuser` | Low-RAM transformer block streaming from mmap'd safetensors |
| `mlx_arsenal.modulation` | `AdaLNModulation`, `ScaleShiftTable`, `modulate`, `gated_residual` | DiT AdaLN modulation primitives (1 / 2 / 6 / 9-param variants) |
| `mlx_arsenal.ffn` | `FeedForward`, `GatedFFN`, `GeGLU`, `SwiGLU` | Transformer FFN / MLP blocks (vanilla + gated variants) |
| `mlx_arsenal.loader` | `SDOps`, `SafetensorsStateDictLoader`, `StateDict`, `read_safetensors_metadata` | State-dict key remapping chain + safetensors loader |
| `mlx_arsenal.rope` | `rope_frequencies_1d`, `rope_frequencies_nd`, `apply_rotary_emb`, `rotate_half`, `meshgrid_nd` | Rotary Position Embeddings (N-D, interleaved + half-rotated variants) |

## Quick start

```python
from mlx_arsenal.spatial import interpolate_nearest, avg_pool1d, replicate_pad
from mlx_arsenal.layout import to_channels_last, convert_conv_weights
from mlx_arsenal.attention import causal_mask

# Resize a video tensor (B, D, H, W, C)
x_resized = interpolate_nearest(x, size=(8, 32, 32))

# Temporal pooling
pooled = avg_pool1d(temporal_features, kernel_size=2)

# Pad with edge replication (like F.pad mode="replicate")
padded = replicate_pad(x, [(0,0), (2,0), (1,1), (1,1), (0,0)])

# Convert PyTorch conv weights to MLX channels-last layout
mlx_weights = convert_conv_weights(pytorch_weights)

# Causal attention mask for autoregressive decoding
mask = causal_mask(seq_len=128, offset=kv_cache_len)
```

### Block streaming (low-RAM transformers)

Run a 20+ GB transformer on a Mac without holding every block resident
at once: keep one shared block module, and rebind its weights from
memory-mapped safetensors before each block's forward.

```python
from mlx_arsenal.streaming import BlockStreamer

# Build the model with ONE block in transformer_blocks (not num_layers).
model = build_my_transformer(num_layers=1)
load_non_block_weights(model, weights_path)

streamer = BlockStreamer(
    weights_path,
    block_prefix="transformer.transformer_blocks.",
)
assert streamer.block_count == num_layers  # discovered from safetensors

shared_block = model.transformer_blocks[0]
prev_idx = None
for i in range(streamer.block_count):
    streamer.bind(shared_block, idx=i, evict_previous=prev_idx)
    x = shared_block(x, ...)  # use the rebound block
    prev_idx = i
```

For LoRA: pass a `lora_fuser` callable to `BlockStreamer` and one or
more `BlockLoraSource` instances to `bind(..., lora_sources=...)`.
Quantization-aware fusion strategies stay in the caller — arsenal
only handles the discovery + indexing.

## Requirements

- Python >= 3.10
- MLX >= 0.32.1
- Apple Silicon Mac

## Development

```bash
pip install -e ".[dev]"
pytest tests/

# Optional: install the pre-commit hook so ruff runs on every `git commit`.
pip install pre-commit
pre-commit install
```

## License

Apache 2.0

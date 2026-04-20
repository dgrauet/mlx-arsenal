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
| `mlx_arsenal.attention` | `causal_mask`, `sliding_window_mask` | Attention mask creation |
| `mlx_arsenal.norm` | `PixelNorm`, `ScaleNorm` | Custom normalization layers |
| `mlx_arsenal.encoding` | `FourierEmbedder` | Sinusoidal positional encoding |
| `mlx_arsenal.diffusion` | `get_timestep_embedding`, `TimestepEmbedding`, `get_sampling_sigmas`, `dynamic_shift_schedule`, `FlowMatchEulerDiscreteScheduler`, `euler_step`, `classifier_free_guidance` | Flow-matching diffusion primitives |
| `mlx_arsenal.moe` | `MoEGate`, `MoELayer` | Top-k mixture-of-experts dispatch |
| `mlx_arsenal.rasterize` | `rasterize_triangles`, `interpolate` | Differentiable triangle rasterization with Metal z-buffer |
| `mlx_arsenal.tiling` | `tiled_process`, `temporal_slice_process` | Memory-efficient large tensor processing |

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

## Requirements

- Python >= 3.10
- MLX >= 0.27.0
- Apple Silicon Mac

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

Apache 2.0

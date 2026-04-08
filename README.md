# mlx-ops

Reusable mid-level building blocks for [MLX](https://github.com/ml-explore/mlx) — the missing layer between `mlx.nn` primitives and full model implementations.

## Why

Every MLX project (mlx-video, mflux, mlx-audio, mlx-vlm) re-implements the same building blocks: causal convolutions, multi-dimensional RoPE, adaptive layer norms, timestep embeddings, pixel shuffle, channel layout conversion, etc. `mlx-ops` extracts these into a single tested library.

## Install

```bash
pip install mlx-ops
```

## Modules

| Module | Components | Use case |
|--------|-----------|----------|
| `mlx_ops.conv` | `CausalConv1d/2d/3d`, `weight_norm` | Video/audio models |
| `mlx_ops.attention` | `sdpa`, `SDPAttention`, `RoPE2d/3d`, `causal_mask`, `sliding_window_mask` | Any transformer |
| `mlx_ops.norm` | `AdaLayerNormZero/Single/Continuous`, `PixelNorm`, `ScaleNorm` | Diffusion models |
| `mlx_ops.diffusion` | `TimestepEmbedding`, `ResnetBlock2d/3d`, `FlowMatchEulerScheduler` | Diffusion pipelines |
| `mlx_ops.spatial` | `patchify/unpatchify`, `PatchEmbed2d/3d`, `upsample_nearest/bilinear`, `pixel_shuffle/unshuffle` | Vision/video models |
| `mlx_ops.layout` | `to_channels_last/first`, `channels_last` ctx manager, `convert_conv_weights`, `load_safetensors` | PyTorch model porting |
| `mlx_ops.tiling` | `tiled_process`, `temporal_slice_process` | Memory-efficient inference |

## Quick start

```python
from mlx_ops.conv import CausalConv3d
from mlx_ops.attention import RoPE3d, sdpa
from mlx_ops.norm import AdaLayerNormZero

# Video model building blocks
conv = CausalConv3d(16, 32, kernel_size=3)
rope = RoPE3d(dim=64, max_t=32, max_h=16, max_w=16)
norm = AdaLayerNormZero(dim=64, cond_dim=128)
```

## Requirements

- Python >= 3.10
- MLX >= 0.27.0
- Apple Silicon Mac

## License

Apache 2.0

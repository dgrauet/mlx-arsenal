# mlx-ops

Low-level operations missing from [MLX](https://github.com/ml-explore/mlx) core — the CUDA ops you need when porting PyTorch models to Apple Silicon.

## Install

```bash
pip install mlx-ops
```

## Modules

| Module | Components | Replaces (PyTorch) |
|--------|-----------|-------------------|
| `mlx_ops.spatial` | `interpolate_nearest`, `interpolate_3d`, `avg_pool1d`, `replicate_pad`, `upsample_nearest/bilinear`, `pixel_shuffle/unshuffle`, `patchify/unpatchify`, `PatchEmbed2d/3d` | `F.interpolate`, `F.avg_pool1d`, `F.pad(mode="replicate")`, `F.pixel_shuffle` |
| `mlx_ops.layout` | `to_channels_last/first`, `channels_last` ctx manager, `convert_conv_weights`, `load_safetensors` | NCHW/NHWC conversion, weight transposition |
| `mlx_ops.conv` | `weight_norm` | `nn.utils.weight_norm` |
| `mlx_ops.attention` | `causal_mask`, `sliding_window_mask` | Attention mask creation |
| `mlx_ops.norm` | `PixelNorm`, `ScaleNorm` | Custom normalization layers |
| `mlx_ops.tiling` | `tiled_process`, `temporal_slice_process` | Memory-efficient large tensor processing |

## Quick start

```python
from mlx_ops.spatial import interpolate_nearest, avg_pool1d, replicate_pad
from mlx_ops.layout import to_channels_last, convert_conv_weights

# Resize a video tensor (B, D, H, W, C)
x_resized = interpolate_nearest(x, size=(8, 32, 32))

# Temporal pooling
pooled = avg_pool1d(temporal_features, kernel_size=2)

# Pad with edge replication (like F.pad mode="replicate")
padded = replicate_pad(x, [(0,0), (2,0), (1,1), (1,1), (0,0)])

# Convert PyTorch conv weights to MLX layout
mlx_weights = convert_conv_weights(pytorch_weights)
```

## Requirements

- Python >= 3.10
- MLX >= 0.27.0
- Apple Silicon Mac

## License

Apache 2.0

# mlx-arsenal

Mid-level building blocks for [Apple MLX](https://github.com/ml-explore/mlx) —
the missing layer between `mlx.nn` and full model implementations.

When porting PyTorch / diffusers / transformers models to MLX, you keep
hitting the same gaps: `F.interpolate`, weight-norm helpers, attention
masks, channels-last layout conversion, diffusion schedulers, MoE
routing, low-RAM block streaming. `mlx-arsenal` collects them in one
place so each port doesn't ship a sixth copy.

## Why

- **Channels-last by default** — MLX is NHWC / NDHWC; arsenal matches it.
- **MLX-native semantics** — no `torch.compile`-shaped abstractions, no
  shadow autograd. Lazy `mx.eval` boundaries are respected.
- **One missing-feature per module** — easy to compose, easy to ignore.
- **Strict `ty` type-checking** — full type coverage, no `Any` leaks.

## Modules

| Module | Components | Replaces (PyTorch) |
|--------|-----------|-------------------|
| [`spatial`](api/spatial.md) | `interpolate_nearest`, `interpolate_3d`, `avg_pool1d`, `replicate_pad`, `upsample_nearest/bilinear`, `pixel_shuffle/unshuffle`, `patchify/unpatchify`, `PatchEmbed2d/3d` | `F.interpolate`, `F.avg_pool1d`, `F.pad`, `F.pixel_shuffle` |
| [`layout`](api/layout.md) | `to_channels_last/first`, `channels_last` ctx, `convert_conv_weights`, `load_safetensors` | NCHW ↔ NHWC, weight transposition |
| [`conv`](api/conv.md) | `weight_norm`, `WeightNorm` | `nn.utils.weight_norm` |
| [`attention`](api/attention.md) | `causal_mask`, `sliding_window_mask` | Attention mask creation |
| [`norm`](api/norm.md) | `PixelNorm`, `ScaleNorm` | Custom normalization layers |
| [`encoding`](api/encoding.md) | `FourierEmbedder` | Sinusoidal positional encoding |
| [`diffusion`](api/diffusion.md) | `get_timestep_embedding`, `TimestepEmbedding`, `FlowMatchEulerDiscreteScheduler`, `DDIMScheduler`, `euler_step`, `classifier_free_guidance`, `TeaCacheController` | Flow-matching + DDIM diffusion primitives |
| [`moe`](api/moe.md) | `MoEGate`, `MoELayer` | Top-k mixture-of-experts dispatch |
| [`rasterize`](api/rasterize.md) | `rasterize_triangles`, `interpolate` | Differentiable triangle rasterization |
| [`tiling`](api/tiling.md) | `tiled_process`, `temporal_slice_process` | Memory-efficient large tensor processing |
| [`streaming`](api/streaming.md) | `BlockStreamer`, `BlockLoraSource`, `LoraFuser` | Low-RAM transformer block streaming |
| [`modulation`](api/modulation.md) | `AdaLNModulation`, `ScaleShiftTable`, `modulate`, `gated_residual` | DiT AdaLN modulation primitives |
| [`ffn`](api/ffn.md) | `FeedForward`, `GatedFFN`, `GeGLU`, `SwiGLU` | Transformer FFN / MLP blocks |
| [`loader`](api/loader.md) | `SDOps`, `SafetensorsStateDictLoader`, `StateDict`, `read_safetensors_metadata` | State-dict key remapping + safetensors loader |

## Quick start

```python
from mlx_arsenal.spatial import interpolate_nearest, replicate_pad
from mlx_arsenal.layout import to_channels_last, convert_conv_weights
from mlx_arsenal.attention import causal_mask

# Resize a video tensor (B, D, H, W, C)
x_resized = interpolate_nearest(x, size=(8, 32, 32))

# Pad with edge replication
padded = replicate_pad(x, [(0, 0), (2, 0), (1, 1), (1, 1), (0, 0)])

# Convert PyTorch conv weights to MLX channels-last
mlx_weights = convert_conv_weights(pytorch_weights)

# Causal attention mask for autoregressive decoding
mask = causal_mask(seq_len=128, offset=kv_cache_len)
```

See [Install](install.md) for setup and the per-module API pages for
full reference.

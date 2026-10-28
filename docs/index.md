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
- **Strict `ty` type-checking** — full type coverage (`Any` only at the
  MLX interop boundary in `_typing`).

## Modules

| Module | Components | Replaces (PyTorch) |
|--------|-----------|-------------------|
| [`spatial`](api/spatial.md) | `interpolate_nearest`, `interpolate_3d`, `avg_pool1d`, `replicate_pad`, `upsample_nearest/bilinear`, `pixel_shuffle/unshuffle`, `patchify/unpatchify`, `PatchEmbed2d/3d` | `F.interpolate`, `F.avg_pool1d`, `F.pad`, `F.pixel_shuffle` |
| [`layout`](api/layout.md) | `to_channels_last/first`, `channels_last` ctx, `convert_conv_weights`, `load_safetensors` | NCHW ↔ NHWC, weight transposition |
| [`conv`](api/conv.md) | `weight_norm`, `WeightNorm`, `CausalConv3d` | `nn.utils.weight_norm`, causal video-VAE `CausalConv3d` with chunk cache |
| [`attention`](api/attention.md) | `causal_mask`, `sliding_window_mask`, video-DiT masks (`spatial_only_mask`, `temporal_only_mask`, `sliding_tile_centered_mask`, `sliding_tile_block_mask`, `radial_box_mask`, `radial_gaussian_mask`, `frame_stride_diagonal_mask`, `vertical_stripe_mask`), head-pattern profiling (`Kind`, `classify`, `classify_heads_from_qk/probs`), token permutation (`block_contiguous_permutation`, `invert_permutation`), `block_causal_mask`, sparse-block compensation (`centroid_compensated_attention`, `probe_residual_correction`, `select_probe_rows`, `tile_labels`), dynamic block masks (`antidiagonal_block_scores`, `top_p_block_mask`, `block_self_similarity`, `select_tiling`, `minmax_block_scores`, `top_k_block_mask`, `radius_bounded_block_scores`, `relative_block_mask`), block-sparse attention (`block_sparse_attention`, `block_sparse_kernel_available`, `centroid_threshold_block_mask`), `RollingKVCache` | Attention masks (LLM + sparse video DiT), head archetype classification, SVG2-style permutation, block-diffusion LLM mask, sparse-block compensation (SVG-EAR, SparsePR), dynamic block masks (XAttention, SPADE, RBS, Sol-Attn), block-sparse attention call site (dense until a kernel passes ADR-0002's gates), sink + sliding-window KV cache for chunked causal generation |
| [`norm`](api/norm.md) | `PixelNorm`, `ScaleNorm` | Custom normalization layers |
| [`encoding`](api/encoding.md) | `FourierEmbedder` | Sinusoidal positional encoding |
| [`diffusion`](api/diffusion.md) | `get_timestep_embedding`, `TimestepEmbedding`, `FlowMatchEulerDiscreteScheduler`, `DDIMScheduler`, `euler_step`, `CurvatureAdaptiveStepper`, `classifier_free_guidance`, `posterior_mean_capped_guidance`, `normalized_attention_guidance`, step-aware caches (`TeaCacheController`, `sea_filter`, `SoftCacheController`, `regrid_tail`, `PerLayerAttentionCache`, `PerHeadAttentionCache`, `WindowResidualController`, `VerifiedFeatureCache`), CFG-skip (`CFGSkipController`, `CFGSimilarityProfiler`, `cfg_head_similarity`, `cfg_skip_mask`, `splice_heads`), sigma schedules (`get_sampling_sigmas`, `dynamic_shift_schedule`), `geometric_threshold`, dLLM block decoding (`token_stats`, `threshold_transfer`, `topk_transfer`, `factor_transfer`, `entropy_bound_transfer`, `edit_transfer`, `apply_block_edits`, `token_prob`, `drift_probe_masks`, `drift_remask`, `transfer_schedule`, `block_ranges`), uniform-noise dLLM decoding (`linear_temperature`, `uniform_canvas`, `renoise`, `StableConfidentStopping`), head-mask reuse (`pooled_qk`, `qk_drift`, `HeadMaskCache`) | Flow-matching + DDIM diffusion primitives, cache-then-reuse and forecast-then-verify controllers, CFG acceleration, masked diffusion LLM (dLLM) commit rules and CoDR drift remasking, uniform-noise dLLM (DiffusionGemma) decoding, head-wise sparse-mask reuse (HEART) |
| [`moe`](api/moe.md) | `MoEGate`, `MoELayer` | Top-k mixture-of-experts dispatch |
| [`rasterize`](api/rasterize.md) | `rasterize_triangles`, `interpolate` | Tile-binned triangle rasterization, exact fixed-point coverage |
| [`tiling`](api/tiling.md) | `tiled_process`, `temporal_slice_process` | Memory-efficient large tensor processing |
| [`streaming`](api/streaming.md) | `BlockStreamer`, `BlockLoraSource`, `LoraFuser` | Low-RAM transformer block streaming |
| [`modulation`](api/modulation.md) | `AdaLNModulation`, `ScaleShiftTable`, `modulate`, `gated_residual` | DiT AdaLN modulation primitives |
| [`ffn`](api/ffn.md) | `FeedForward`, `GatedFFN`, `GeGLU`, `SwiGLU` | Transformer FFN / MLP blocks |
| [`loader`](api/loader.md) | `SDOps`, `SafetensorsStateDictLoader`, `StateDict`, `read_safetensors_metadata` | State-dict key remapping + safetensors loader |
| [`rope`](api/rope.md) | `rope_frequencies_1d`, `rope_frequencies_nd`, `apply_rotary_emb`, `rotate_half`, `meshgrid_nd` | Rotary Position Embeddings (N-D, both pair conventions) |

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

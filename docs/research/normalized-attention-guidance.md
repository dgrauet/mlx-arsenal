# Normalized attention guidance (NAG)

Research notes on NAG and the function `mlx-arsenal` exposes. This page
documents a pattern; it does not open an ADR.

## The idea

*Normalized Attention Guidance: Universal Negative Guidance for Diffusion
Models* (Chen et al., arXiv 2505.21179; reference code under MIT).

Classifier-free guidance extrapolates model outputs, which breaks down on
few-step and distilled models: there is no unconditional branch worth
extrapolating from, and large scales blow up the sample. NAG guides inside
the cross-attention instead. The same queries attend to the positive and to
the negative context, the two attention outputs are extrapolated, and the
result is kept close to the positive output in two ways:

```text
z    = s·z⁺ − (s − 1)·z⁻
r    = ‖z‖₁ / ‖z⁺‖₁                     per token, over all heads
z    = z · min(r, τ) / r                  clip the feature-norm growth
out  = α·z + (1 − α)·z⁺                  blend back
```

The cost is one extra attention call per guided layer, not a second model
forward. The authors apply it to Flux, SD3.5, SDXL, Wan 2.1, Hunyuan Video
and LTX; ltx-2-mlx uses it on its distilled path.

## What `mlx-arsenal` ships

`mlx_arsenal.diffusion.normalized_attention_guidance(positive, negative, *,
scale, tau=2.5, alpha=0.5, axis=-1)`. It takes the two attention outputs
and returns the guided one; the caller runs both attention calls and owns
the hook point (after the attention, before the output projection).

- `axis` names the feature axes of the L1 norm: `-1` for a merged
  `(B, N, H·D)` output, `(1, 3)` for `(B, H, N, D)`. The reference
  normalizes the merged output, so the norm spans every head.
- Reference defaults: `tau = 2.5`, `alpha = 0.5`; `scale = 5` is typical
  for Flux. ltx-2-mlx's distilled path uses `scale = 11`, `alpha = 0.25`.

## Deviations from the reference

- **Float32.** The norms and the blend run in float32 and the result is cast
  back. Reducing in bf16 saturates on long rows (see the reductions
  convention in `CLAUDE.md`); a test covers a 2^20-element row.
- **No NaN on all-zero positives.** The reference divides by `‖z⁺‖₁`. Here
  a token where both norms are 0 keeps its (zero) guidance, and a token with
  a zero positive and a nonzero guidance is clipped to 0, the reference's
  limit.

## Validation

- Tests check the function against a transcription of the reference
  `NAGProcessor` (27 parameter combinations), the multi-axis layout, the
  identity at `scale = 1`, dtype handling and the long-row reduction.
- Outside the repo, it matches ltx-2-mlx's own NAG combine exactly (max abs
  difference 0.0) on float32 and bf16, for merged and per-head layouts.

## Out of scope

- The attention hook and the negative-context encoding: model plumbing.
- Per-layer or per-step NAG schedules: caller policy.

## References

- Paper: <https://arxiv.org/abs/2505.21179>
- Reference code (MIT): <https://github.com/ChenDarYen/Normalized-Attention-Guidance>
  (`nag/attention_nag.py`)

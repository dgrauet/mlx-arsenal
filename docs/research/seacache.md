# SeaCache spectral distance

Research notes on SeaCache and the part `mlx-arsenal` exposes. This page
documents a pattern; it does not open an ADR. It complements the
[verified feature caching](verified-feature-caching.md) note.

## The idea

*SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion
Models* (Chung et al., arXiv 2602.18993, CVPR 2026). Adopted in diffusers
([#14663](https://github.com/huggingface/diffusers/pull/14663), September
2026) for Cosmos 3 and Wan T2V.

TeaCache skips a transformer forward while the accumulated relative-L1
change of the first-block modulated input stays under a threshold, and it
maps raw distances through a polynomial calibrated per model. Early in
sampling that input is mostly noise, and its high frequencies inflate the
distance. SeaCache filters the input first with a Wiener-like gain derived
from the noise schedule:

```text
g_t(f) = a_t·S(f) / (a_t²·S(f) + b_t²),   S(f) = 1 / |f|^p
```

`S` is a power-law clean-signal spectrum (`p = 2` for images, `p = 3` for
video), `a_t` and `b_t` are the signal and noise coefficients of the
schedule. The gain is built per axis of the latent token grid, multiplied
across axes, and normalised to unit mean. Noisy steps are low-passed;
near-clean steps pass through almost unchanged. The distance, the
accumulation and the threshold are TeaCache's, without the polynomial.

## What `mlx-arsenal` ships

| API | Role |
|---|---|
| `sea_filter(x, signal_scale, noise_scale, *, axes, power_exp)` | the SEA filter on a channels-last grid |
| `TeaCacheController(..., coefficients=None)` | TeaCache rule on the raw distance |
| `TeaCacheController(..., max_consecutive_skips=2)` | the diffusers streak cap |

Schedule coefficients: flow matching `a = 1 − σ`, `b = σ`; VP / DDPM
`a = √ᾱ`, `b = √(1 − ᾱ)`.

## Recipe

The test suite executes this block and checks its decisions against a
transcription of the reference loop.

<!-- seacache-recipe -->
```python
import mlx.core as mx

from mlx_arsenal.diffusion import TeaCacheController, sea_filter


def seacache_should_compute(controller, step, modulated, grid, sigma, *, power_exp=2.0):
    """SeaCache gate for a flow-matching step: SEA-filter, then TeaCache's rule."""
    B, N, C = modulated.shape  # first-block modulated tokens
    x = modulated.reshape(B, *grid, C)  # back on the latent grid: (B, H, W, C) or (B, T, H, W, C)
    sigma = min(max(sigma, 1e-6), 1.0 - 1e-6)  # as both references
    filtered = sea_filter(x, 1.0 - sigma, sigma, power_exp=power_exp)
    return controller.should_compute(step, filtered)
```

In the denoising loop, with `controller = TeaCacheController(num_steps,
rel_l1_thresh=0.3, max_consecutive_skips=2)`:

```python
if seacache_should_compute(controller, i, modulated, grid, sigmas[i]):
    out = blocks(x)
    controller.cache_residual(out - x)
else:
    out = x + controller.previous_residual
```

Call the gate on every step, boundary steps included, so each distance
compares two filtered inputs. With classifier-free guidance run as two
passes, keep one controller per branch (the Wan reference does).

Thresholds from the paper: `0.3` (≈2×) and `0.6` (≈3×) on FLUX,
`0.12`–`0.35` on video. diffusers ships `0.25` with a cap of 2 consecutive
reuses.

## Deviations and notes

- **Boundary steps.** The authors' scripts store the *unfiltered* input
  at boundary steps, so their first interior distance compares a filtered
  tensor against a raw one. The recipe filters every step, as diffusers
  does.
- **Separable, not radial.** The paper describes a radial gain normalised
  over radial bins; both implementations build a separable product of
  per-axis gains normalised over the full grid. `sea_filter` follows the
  implementations (parity with both at float32 precision, 2.6e-6 relative).
- **Zero denominator.** `TeaCacheController` forces a compute when the
  previous input is all zeros; the references add `1e-16` instead.
- **Residual extrapolation.** diffusers extrapolates the cached residual
  linearly from the last two full computes. That is a caller choice: keep
  two residuals, or use `VerifiedFeatureCache` with `order=1`.
- **Precision.** diffusers keeps the sampling state in float32 while the
  transformer runs in model dtype, for stability under caching. The filter
  itself always computes in float32.

## Out of scope, and why

- **Where `σ` comes from and which tensor is the modulated input**:
  model- and scheduler-specific, the caller's port.
- **The hook / pipeline wiring**: an orchestrator (the `*Runner` pattern
  the doctrine excludes).

## References

- SeaCache: <https://arxiv.org/abs/2602.18993>, project page
  <https://jiwoogit.github.io/SeaCache/>
- diffusers SeaCache hook: <https://github.com/huggingface/diffusers/pull/14663>
- TeaCache: <https://github.com/ali-vilab/TeaCache>

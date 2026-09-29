# Verified feature caching

Research notes on verification-based caching techniques for iterative
models (image / video / audio diffusion, dLLMs). This page documents the
pattern and how it applies to `mlx-arsenal` — it does not commit to an
ADR or a roadmap.

## The pattern: forecast-then-verify

Two families coexist in the 2024-2025 literature:

### Cache-then-reuse (the `mlx-arsenal` status quo)

A step-to-step heuristic (typically relative-L1 on the inputs or hidden
states) decides whether the current step can reuse the previous step's
result — then applies the decision **without checking the actual
quality**.

Implementations in the library: `TeaCacheController`,
`PerLayerAttentionCache`, `PerHeadAttentionCache`,
`WindowResidualController.adaptive`. All of them use relative-L1.

Known limits: silent drift when the smoothness assumption breaks
(content transition, unusual prompt), and a speedup ceiling around 3-4×
before quality collapses.

### Forecast-then-verify

Three steps:

1. **Draft** — parameter-free extrapolation of the features (typically a
   Taylor expansion along the iteration axis, using finite differences
   of previous anchors). Near-zero cost: linear algebra on tensors
   already in memory.
2. **Verify** — a partial forward pass through **a single layer**
   (≈1.7-3.5% of the cost of a full forward) to measure the relative L2
   error between the drafted feature and the real one:
   ```
   e_k = ‖F_pred − F_real‖²₂ / (‖F_real‖²₂ + ε)
   ```
3. **Accept/reject** — an adaptive threshold along the iteration axis
   (`τ_t = τ₀ · β^((T-t)/T)` in SpeCa).

Reference papers:

- **SpeCa** — *Accelerating Diffusion Transformers with Speculative
  Feature Caching* (Zou et al., Sept. 2025). Empirically chosen target
  layer (the 27th on an image DiT), 6-7× speedup with stable FID.
- **TaylorSeer** — *Forecasting Features rather than Caching Them*
  (March 2025). A variant without verification; it collapses to 17.5%
  degradation at the speedup where SpeCa holds.
- **Spiffy / SSD** (Sept-Oct. 2025) — speculative decoding for diffusion
  LLMs (token masking). Same pattern, different axis.

## Caveat: bounded lossy, not lossless

Unlike LLM speculative decoding (lossless by construction through
rejection sampling), SpeCa is **lossy but bounded**: convergence in total
variation, provided the threshold schedule is chosen correctly (see SpeCa
Appendix G).

Do not use it where bit-exact reproducibility is required. For creative
generation, the trade-off is documented and acceptable.

Keep the sampling state (latent, scheduler arithmetic) in float32 when
caching, even if the transformer runs in bf16: reused or extrapolated
features accumulate rounding across skipped steps. diffusers made float32
sampling state its default alongside SeaCache
([diffusers #14663](https://github.com/huggingface/diffusers/pull/14663),
September 2026).

## Applicability to `mlx-arsenal`

### What is a *primitive* (extractable into the library)

- Taylor extrapolation via finite differences (but it is ~20 lines of
  algebra — too thin to deserve its own module).
- The draft → verify → accept/reject orchestration along an abstract
  iteration axis.
- The geometric threshold schedule.

### What stays *caller-side* (per model)

- **Choosing the verification target layer**: pure empirical ablation.
  SpeCa identifies layer 27 on their image DiT; every port has to redo
  the ablation against its own quality metric (FID, ImageReward, VBench,
  perplexity…).
- **Calibrating τ₀ and β**: depends on the model, the sampling schedule,
  and the domain.
- **Choosing the Taylor order**: SpeCa typically uses 1-3 depending on
  the regime.
- **Defining the tracked "feature"**: architecture-dependent.

### Current decision

A single opinionated controller with SpeCa defaults, exposed in
`mlx_arsenal.diffusion.verified_cache`. No batch of extracted primitives
and no ADR — the added value over `TeaCache` is too thin to justify a
multi-PR roadmap, and the real hard parts (target-layer ablation,
threshold calibration) are per-model anyway.

Revisit if 2+ ports adopt the pattern and surface reusable hard parts.

## Adjacent directions noted but not covered here

- **Parallel sampling via Picard-Lindelöf iteration** (Shih et al. 2023,
  *Accelerating Parallel Sampling* 2024) — solves the diffusion ODE in
  parallel across several timesteps. Significant memory cost;
  interesting on a Mac Studio M3 Ultra (generous unified memory),
  prohibitive on modest M-series machines.
- **Accelerated Diffusion via Speculative Sampling** (Jan-July 2025) —
  exploits the link between speculative sampling and reflection maximal
  coupling for stochastic samplers. *Lossless* in the strict sense,
  relevant for evaluation / reproducibility.
- **Parallel Sampling via Autospeculation** (Nov. 2025) — theoretical
  result: O(n) → O(√n) speedup at high precision.

These directions have no MLX implementation (yet) and their mid-level
ROI is not obvious — to be explored separately.

## References

- SpeCa: <https://arxiv.org/abs/2509.11628>
- TaylorSeer: <https://arxiv.org/abs/2503.06923>
- DiTFastAttn: <https://arxiv.org/abs/2406.08552>
- Reference repo for SpeCa: <https://github.com/Shenyi-Z/Cache4Diffusion>
- TeaCache: <https://github.com/ali-vilab/TeaCache>

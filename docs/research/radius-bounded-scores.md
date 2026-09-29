# Radius-bounded block scores (RBS)

Research notes on RBS-Attention's block estimator and the parts
`mlx-arsenal` exposes. This page documents a pattern; it does not open an
ADR. It complements the [dynamic block masks](head-mask-reuse.md) and
[adaptive tiling](adaptive-tiling.md) notes.

## The idea

*RBS-Attention: Radius-Bounded Sparse Prefill for Long-Context LLMs*
(Song, Wei, Peng, arXiv 2609.20971, September 2026, no public code).

Block-sparse attention scores each (query block, key block) pair from a
summary of the key block, usually its centroid `c_b`. A centroid can hide one
strongly matching key among keys that cancel it on average ("mean
dilution"). RBS adds the block's radius `r_b = max ‖k − c_b‖₂`; by
Cauchy-Schwarz every key of the block satisfies `qᵀk ≤ qᵀc_b + ‖q‖·r_b`.
Per query token:

```text
ℓ_base   = qᵀc_b / √D
ℓ_rescue = (qᵀc_b + ‖q‖·r_b·β_b) / √D,   β_b = clip((r_b − r_low)/(r_high − r_low), 0, 1)
```

`r_low` / `r_high` are the median and 90th percentile of the head's block
radii, computed at runtime: only unusually spread blocks get the bound. Each
branch keeps the key blocks whose summed `exp(ℓ)` over the query block is at
least `α` times the row maximum (`α = 0.22` base, `0.18` rescue, tuned
offline on LLM prompts), and the final mask is the union of both branches
and the forced blocks (sinks, local window). The ablation reports the union
beating centroid-only, a full (unattenuated) radius bound and a Quest-style
box bound at matched density on RULER-128K.

It is the L2-ball counterpart of SPADE's element-wise min/max box bound
(`minmax_block_scores`).

## What `mlx-arsenal` ships

| Function | Role |
|---|---|
| `radius_bounded_block_scores(q, k, *, block_size, low, high)` | `(base, rescue)` log block scores |
| `relative_block_mask(log_scores, alpha)` | keep blocks within `α` of the row maximum |

The scores are logsumexps over each query block's tokens — the log of the
paper's `Σ exp(ℓ)`. They work with `relative_block_mask` and
`top_k_block_mask`; soft-max each row before `top_p_block_mask`.

## Recipe

The test suite executes this block.

<!-- rbs-recipe -->
```python
import mlx.core as mx

from mlx_arsenal.attention import radius_bounded_block_scores, relative_block_mask


def rbs_block_mask(q, k, *, block_size=64, alpha_base=0.22, alpha_rescue=0.18, diagonal=1):
    """RBS block mask: union of the base and rescue branches plus a forced diagonal band."""
    base, rescue = radius_bounded_block_scores(q, k, block_size=block_size)
    mask = mx.maximum(relative_block_mask(base, alpha_base), relative_block_mask(rescue, alpha_rescue))
    Cq, Ck = mask.shape[-2:]
    rows, cols = mx.arange(Cq)[:, None], mx.arange(Ck)[None, :]
    band = mx.where(mx.abs(rows - cols) < diagonal, 0.0, float("-inf"))
    return mx.maximum(mask, band)  # additive (B, H, Cq, Ck)
```

Expand it to tokens with `mx.repeat` on both block axes, or pass it as
`block_mask` to `centroid_compensated_attention` with
`labels = mx.arange(N) // block_size`.

## Measured on ERNIE-Image (MLX)

Image-image attention of ERNIE-Image SFT (MLX port, int8) at 1024² (a 64 × 64
token grid), step 10 of 28 (`σ ≈ 0.84`), five layers × 32 heads,
`block_size = 64`, tokens in raster order or regrouped into 8 × 8 tiles
(`tile_labels`). Recall is the true attention mass (dense softmax) captured
by the kept blocks, averaged over query blocks.

Estimators ranked by recall with `top_k_block_mask`, raster / tiled order:

| Kept blocks | oracle | antidiagonal (XAttention) | centroid (RBS base) | min/max (SPADE) | RBS rescue alone |
|---|---|---|---|---|---|
| 5 % | .475 / .586 | .460 / .545 | .444 / .533 | .424 / .498 | .103 / .107 |
| 10 % | .606 / .705 | .591 / .671 | .568 / .660 | .548 / .636 | .144 / .150 |
| 20 % | .744 / .817 | .728 / .791 | .703 / .781 | .685 / .767 | .237 / .248 |

The rescue score is not a ranking on its own: it only makes sense in the
union. RBS's union at the paper's thresholds (`0.22` / `0.18`) against the
base branch alone with its threshold lowered to the same density:

| Order | union density | union recall | base-only recall, same density |
|---|---|---|---|
| raster | .266 | .772 | .799 (union −2.7 points) |
| 8 × 8 tiles | .165 | .699 | .728 (union −2.9 points) |

The union loses on every one of the five layers. On this DiT, mean dilution
does not show up: the rescue branch spends the budget on spread-out key
blocks that carry little mass. Keep the base branch and tune its threshold,
or use XAttention's estimate, unless a model shows the dilution the paper
measured on long-context LLMs.

## Deviations and notes

- **Memory.** The estimator scores every query token: about three
  `(B, H, Nq, Nk / block_size)` float32 tensors at once, ≈1.1 GB per head for
  a Wan 720p layer. Call it per head at video scale.
- **Degenerate quantiles.** When the two radius quantiles coincide, `β_b` is
  1 for blocks strictly wider than `r_low` and 0 otherwise.
- **Thresholds.** `α` values come from LLM prefill; they are hyperparameters
  to re-tune for DiT attention, not calibration tooling this library ships.

## Out of scope, and why

- **Calibrating `α`**, **sink / window widths**, **causal candidate sets**
  (compose with `block_causal_mask`): caller choices.
- **GQA grouping** and the **block-sparse kernel**: the estimators here are
  dense MLX math (ADR-0001).

## References

- RBS-Attention: <https://arxiv.org/abs/2609.20971>
- SPADE (min/max box bound): <https://arxiv.org/abs/2608.03335>

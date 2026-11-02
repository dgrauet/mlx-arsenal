# Skip-error block scores (Triage)

Research notes on Triage's router score and the part `mlx-arsenal` exposes.
This page documents a pattern; it does not open an ADR. It complements the
[sparse-block compensation](sparse-block-compensation.md) and
[RBS](radius-bounded-scores.md) notes.

## The idea

*Triage* (arXiv 2610.35706, October 2026, no public code) routes each
128 × 128 attention tile to one of three tiers: skip, NVFP4 ("skim"), or
NVFP4 with a residual pass ("refine"). A skipped tile is represented by the
mean key (with a `log n` logit bias) and mean value of each of its two
64-key clusters, which is the SVG-EAR / Sol-Attn compensation that
`block_sparse_attention(compensate=True)` already applies.

The router does not rank tiles by attention mass. It predicts the error of
skipping each tile from block moments. With `s` the logit scale, `‖q‖²` the
mean squared query norm of the query block, and `Σ_K`, `Σ_KV`, `v̄` the key
covariance, key-value covariance and mean value of a key cluster:

```text
J = 1 − exp(−s²·‖q‖²·tr Σ_K / 2D)       softmax weight the mean key loses (Jensen)
R = s·‖q‖·‖Σ_KV‖_F / (√D·‖v̄‖)           shift of the mean value from the attended value
E = m̃·(J + R)·√w_h
```

`m̃` averages two mass estimates: the block's mean query against every key,
with a query-variance correction, and every query against the
`log n`-biased centroids, divided by `1 − J`. `w_h` is the head's output
energy at the previous call relative to the layer mean, so one threshold
compares errors across heads. A tile is kept when `N_t·E` clears a single
price `π` (`N_t` is the number of tiles in the row), which keeps `π`
comparable across lengths. The derivation is first order: under a small
change of the softmax normalizer, the row error is bounded by the sum of
single-tile errors, so tiles can be chosen independently.

In the paper's ablation (Wan 2.2 and HunyuanVideo, at equal kept tiles),
ranking by mass alone raises the error by 18 %, dropping `√w_h` by 7 %, and
dropping the two mass corrections by 3 %.

## What `mlx-arsenal` ships

`mlx_arsenal.attention.skip_error_block_scores(q, k, v, *, block_size=64,
head_energy=None, scale=None)` returns `E` per (query block, key block), as
float32 of shape `(B, H, Cq, Ck)`. Select with `top_k_block_mask` for a
fixed budget, or with the price rule
`mx.where(Ck * err >= price, 0.0, -mx.inf)`, and pass the mask to
`block_sparse_attention(..., compensate=True)`:

```python
energy = mx.mean(mx.sum(prev_out * prev_out, axis=-1), axis=-1)  # (B, H), last call
err = skip_error_block_scores(q, k, v, block_size=64, head_energy=energy)
mask = mx.where(err.shape[-1] * err >= price, 0.0, -mx.inf)
out = block_sparse_attention(q, k, v, block_mask=mask, block_size=64)
```

Carrying `head_energy` from one call to the next is the caller's job, as
with every other cache in the library. Without it, the function uses the
paper's first-call value, `‖mean(v)‖²` per head.

## Measured on ERNIE-Image (MLX)

Image-image attention of ERNIE-Image SFT (MLX port, int8) at 1024² (4096
tokens, 64 × 64 grid), one prompt, seed 0, 50 steps with shift 4. Q, K and V
are post-RoPE, from the conditional branch, captured at steps 5, 15 and 40
on layers 2, 9, 18, 27 and 34 (15 calls × 32 heads). `head_energy` comes
from the dense output of the previous step. Tokens are in raster order or
regrouped into 8 × 8 tiles (`tile_labels`). Blocks are 64 tokens. The
metric is the relative error of `block_sparse_attention(compensate=True)`
against dense attention over the whole layer,
`sqrt(Σ_h ‖ΔO_h‖² / Σ_h ‖O_h‖²)`, as a geometric mean over the 15 calls.

Every ranking keeps the same number of key blocks in each row
(`top_k_block_mask`). The two oracles use the dense attention: exact block
mass, and the exact error of skipping each block alone.

| Order | kept | Triage `E` | `m̃` alone | XAttention (stride 8) | centroid | oracle mass | oracle skip error |
|---|---|---|---|---|---|---|---|
| raster | 10 % | .218 | .242 | .221 | .303 | .203 | .199 |
| raster | 20 % | .142 | .156 | .147 | .204 | .130 | .124 |
| raster | 30 % | .108 | .117 | .115 | .147 | .097 | .091 |
| 8 × 8 tiles | 10 % | .178 | .185 | .183 | .226 | .135 | .135 |
| 8 × 8 tiles | 20 % | .097 | .103 | .119 | .139 | .081 | .078 |
| 8 × 8 tiles | 30 % | .069 | .075 | .092 | .107 | .058 | .055 |

The `J + R` factor lowers the error in every one of the 90 (call,
order, budget) cases; the geometric mean drops by 4–11 %, the same
direction as the paper's 18 %.
Triage beats the centroid score everywhere and XAttention in 53–87 % of
calls in raster order and 73–100 % in tiled order. The gain holds at early,
middle and late steps. Exact mass still does better than any estimate, by
7–11 % in raster order and 19–32 % in tiled order.

The price rule, with one threshold over all heads of the layer and the
previous step's head energy, lowers the error further at the same overall
density: .218/.132/.090 in raster order and .147/.085/.056 in tiled order
at 10/20/30 %. It does so by giving more blocks to the heads with large
outputs. At 30 % it matches or beats the exact-mass oracle restricted to a
uniform per-row budget.

Cost: 17 ms for the 32 heads of one layer at 4096 tokens, against 7 ms for
XAttention at stride 8 and 79 ms for dense `mx.fast.scaled_dot_product_attention`
(fp16 inputs, Apple M2 Pro). The score computes two products of size
`N × N / block_size`, so its share of the dense cost falls as `N` grows,
but at image lengths it is not free. Without a block-sparse kernel
(ADR-0002), `block_sparse_attention` runs dense, so these numbers measure
routing quality, not speed.

## Deviations and notes

- **One cluster per block.** The paper pairs two 64-key clusters into a
  128-key tile and averages their moments. Here a block is one cluster, so
  the weights `ω_β` disappear and `J`, `R` are the cluster's own values.
- **No clustering tree.** Triage groups tokens with a PDDP tree (principal
  direction splits, rebuilt every five steps). Its ablation shows the tree
  matters (error floor 9.6 % → 12.9 % on Wan 2.2 without it). The caller
  can permute tokens first, as for any block mask; the 8 × 8 tile order
  above is the cheap version.
- **Two tiers.** With only "exact" and "skip", Triage's `α = 0.22` and
  `c_rf = 3` drop out and the rule is `N_t·E ≥ π`. Its prices
  (`π = 0.15`, `0.3`) are for the three-tier NVFP4 kernel and do not
  transfer.
- **`‖v̄‖` floor.** A block with zero mean value has an undefined relative
  shift; its norm is floored at `1e-6`, which keeps such blocks.
- **`1/(1 − J)` in the log domain**, so the correction cannot overflow when
  a block is very spread.
- **Clean-room.** No code is public; the function follows the paper's
  appendix, and the tests check it against a float64 NumPy transcription.

## Out of scope, and why

- **The NVFP4 skim and refine tiers**: Metal has no FP4 tensor cores, and
  low-bit attention is a kernel, not a mid-level primitive.
- **The PDDP clustering tree**: a separate permutation primitive, worth its
  own evaluation if a port needs it.
- **Choosing `π` per model**: a hyperparameter, like `τ` in Sol-Attn.

## References

- Triage: <https://arxiv.org/abs/2610.35706>
- OpenAttention ErrorGate (same idea for LLMs): <https://arxiv.org/abs/2610.30951>
- SVG-EAR (centroid compensation): <https://arxiv.org/abs/2603.08982>

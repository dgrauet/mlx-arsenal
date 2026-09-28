# Adaptive tiling (SPADE)

Research notes on SPADE's input-adaptive block-sparse attention for video
DiTs and the parts `mlx-arsenal` exposes. This page documents a pattern; it
does not open an ADR. It complements the
[dynamic block masks](head-mask-reuse.md) and
[sparse-block compensation](sparse-block-compensation.md) notes.

## The idea

*SPADE: An Input-Adaptive Sparse Attention Engine for Fast Video Diffusion
Models Inference* (Liu et al., arXiv 2608.03335, DAC '26; code
[6somehow/DAC-SPADE](https://github.com/6somehow/DAC-SPADE), BSD-3-Clause).

Block-sparse attention estimates each (query block, key block) score from
per-block summaries. A summary represents its block well only when the
block's queries are alike, and which 3D tile shape groups similar queries
differs per head: some heads are spatial, others temporal. SPADE therefore:

1. scores each candidate tiling per head by **query cohesion** — the mean
   pairwise cosine similarity of the queries inside each tile (its SICS) —
   and picks the most cohesive tiling;
2. reorders Q/K/V so each tile is contiguous;
3. estimates block scores from element-wise **max/min** summaries,
   `max((q_max + q_min)·k_maxᵀ, (q_max + q_min)·k_minᵀ)`;
4. keeps a fixed budget of key blocks per query block (top-k: 17 % on
   Wan 2.1, 10 % on HunyuanVideo), plus static sink and diagonal blocks.

Nothing is trained or calibrated offline; the budgets, warm-up steps and
candidate tilings are per-model constants. An August 2026 sweep described
SPADE as "token criticality" — it has no per-token criticality score.

## What `mlx-arsenal` ships

`mlx_arsenal.attention`:

| Function | Role |
|---|---|
| `block_self_similarity(x, *, block_size)` | per-block mean pairwise cosine (SICS), `O(block·D)` |
| `select_tiling(q, grid, tiles)` | per-head index of the most cohesive candidate tiling |
| `minmax_block_scores(q, k, *, block_size)` | SPADE's min/max block estimate |
| `top_k_block_mask(scores, k)` | fixed per-row block budget, additive mask |

They compose with `tile_labels` (tile order), `top_p_block_mask`, and
`centroid_compensated_attention`.

## Recipe

Batch 1; heads that picked the same tiling share one attention call. The
test suite executes this exact block.

<!-- spade-recipe -->
```python
import mlx.core as mx

from mlx_arsenal.attention import (
    centroid_compensated_attention,
    minmax_block_scores,
    select_tiling,
    tile_labels,
    top_k_block_mask,
)


def spade_attention(q, k, v, grid, tiles, *, budget=0.17):
    """Per-head adaptive tiling, min/max block estimate, top-k budget, compensated attention."""
    if q.shape[0] != 1:
        raise ValueError("this recipe is written for batch 1 (tilings are chosen per batch row)")
    T, H, W = grid
    choice = select_tiling(q, grid, tiles)[0].tolist()  # one tiling index per head
    out = mx.zeros((*q.shape[:3], v.shape[-1]), dtype=q.dtype)
    for c, tile in enumerate(tiles):
        heads = [h for h, pick in enumerate(choice) if pick == c]
        if not heads:
            continue
        idx = mx.array(heads)
        order = mx.argsort(tile_labels(T, H, W, tile=tile))  # tiles become contiguous
        qg, kg, vg = (mx.take(mx.take(x, idx, axis=1), order, axis=2) for x in (q, k, v))
        block = tile[0] * tile[1] * tile[2]
        scores = minmax_block_scores(qg, kg, block_size=block)
        mask = top_k_block_mask(scores, max(1, int(budget * scores.shape[-1])))  # floor, as SPADE
        labels = mx.arange(qg.shape[2]) // block
        og = centroid_compensated_attention(
            qg, kg, vg, q_labels=labels, k_labels=labels, block_mask=mask
        )
        out[:, idx] = mx.take(og, mx.argsort(order), axis=2)  # back to T-major order
    return out
```

To drop skipped blocks instead of compensating them, expand the block mask
to tokens (`mx.repeat` on both axes by `block`) and call
`mx.fast.scaled_dot_product_attention`. SPADE also always keeps a few sink
key blocks and a diagonal band (its Wan configs: `fixed_sink_width=5`,
`fixed_diag_width=40`): OR them into the block mask — top-k alone at 17 % is
not the same mask as SPADE's.

## Dense by design

SPADE's speedups (2.3–3.4× attention on H800) come from its fused CUDA
estimator and Hopper block-sparse kernel. On MLX the mask still runs through
dense attention, so these functions are a quality tool and a reference for a
future kernel — they let a port measure what adaptive tiling costs or saves
in quality against a fixed tile shape.

## Deviations and notes

- **Selection statistic.** SPADE's engine picks tilings with its CUDA
  `cossim` kernel, which pools pairwise similarities over the whole tiling
  (Σ pair similarities / Σ pairs) and counts every token, including
  zero-norm ones. `select_tiling` averages per-tile mean cosines
  (`block_self_similarity`, like SPADE's torch `_block_summarize`) and
  excludes zero-norm tokens. With tiles of equal volume across candidates
  (every SPADE candidate is 64 tokens) and no padding, both pick the same
  tiling; the paper's "sum of SICS" is then also equivalent.
- **Divisible grids only.** Every candidate tile must divide the latent
  grid. SPADE's kernel allows partial edge tiles, and real grids often do
  not divide evenly — Wan's `16×45×80` rejects its own `(1, 4, 16)` and
  `(1, 8, 8)` candidates. Pad the grid to a multiple of the tiles with
  zero tokens (excluded from the cohesion) and mask the padded keys.
- **Batch**: `select_tiling` decides per (batch, head); the recipe is
  written for batch 1 and rejects larger batches.

## Out of scope, and why

- **Candidate tilings, budgets, warm-up schedule, start layer**: per-model
  constants, the caller's choice.
- **Static sink / diagonal blocks**: a caller-side OR (widths are
  per-model config values, see the recipe note).
- **Intra-block filter**: disabled in the released configurations.
- **The policy function / engine**: an orchestrator (the `*Runner` pattern
  the doctrine excludes).
- **Kernels**: out of scope for a mid-level library (ADR-0001).

## References

- SPADE: <https://arxiv.org/abs/2608.03335>, code
  <https://github.com/6somehow/DAC-SPADE>
- SpargeAttn (self-similarity check, mean summaries):
  <https://arxiv.org/abs/2502.18137>

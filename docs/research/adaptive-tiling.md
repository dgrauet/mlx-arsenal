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
        mask = top_k_block_mask(scores, max(1, round(budget * scores.shape[-1])))
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
key blocks and a diagonal band: OR them into the block mask.

## Dense by design

SPADE's speedups (2.3–3.4× attention on H800) come from its fused CUDA
estimator and Hopper block-sparse kernel. On MLX the mask still runs through
dense attention, so these functions are a quality tool and a reference for a
future kernel — they let a port measure what adaptive tiling costs or saves
in quality against a fixed tile shape.

## Deviations and notes

- **Mean, not sum.** The paper aggregates per-block SICS by sum; the released
  torch reference averages cosines (`_block_summarize`), which
  `block_self_similarity` follows. With equal tile sizes the argmax in
  `select_tiling` is the same either way.
- **Zero-norm tokens** (padding) are excluded from the cohesion, as in the
  reference.
- **Batch**: `select_tiling` decides per (batch, head); the recipe is
  written for batch 1.

## Out of scope, and why

- **Candidate tilings, budgets, warm-up schedule, start layer**: per-model
  constants, the caller's choice.
- **Static sink / diagonal blocks**: trivial to OR in; their exact semantics
  in the reference are lightly specified.
- **Intra-block filter**: disabled in the released configurations.
- **The policy function / engine**: an orchestrator (the `*Runner` pattern
  the doctrine excludes).
- **Kernels**: out of scope for a mid-level library (ADR-0001).

## References

- SPADE: <https://arxiv.org/abs/2608.03335>, code
  <https://github.com/6somehow/DAC-SPADE>
- SpargeAttn (self-similarity check, mean summaries):
  <https://arxiv.org/abs/2502.18137>

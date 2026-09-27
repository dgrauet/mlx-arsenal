# Sparse-block compensation

Research notes on training-free methods that recover the signal lost by
block-sparse attention in video DiTs, and how `mlx-arsenal` exposes them.
This page documents a pattern and its fit with the library; it does not
open an ADR. It extends the sparse-attention work of
[ADR-0001](../adr/0001-sparse-attention-roadmap.md).

## The problem

Block-sparse attention (STA, SVG2, Radial…) groups queries and keys into
clusters and computes only some (query cluster, key cluster) blocks. The
skipped blocks are usually **hard-dropped**: the softmax is renormalized
over the kept keys, and the probability mass of the dropped keys is lost.
Two 2026 methods recover part of it without training.

### SVG-EAR — centroid compensation

*SVG-EAR: Parameter-Free Linear Compensation for Sparse Video Generation
via Error-aware Routing* (arXiv 2603.08982, ECCV 2026; code Apache-2.0 in
`svg-project/Sparse-VideoGen`).

Every skipped key `k_j` is replaced by its cluster centroid `k̄_c`, and its
value by `v̄_c`, inside one softmax. Keys of a cluster then share one
logit, so the skipped part collapses to one term per cluster with a
log-count prior:

```text
s_ic = q_i · k̄_c · scale + log n_c        (skipped clusters only)
```

The output stays an exact softmax average over a modified key set. The
paper merges the sparse kernel's log-sum-exp with the centroid terms in a
custom Triton kernel, and adds an error-aware router choosing which blocks
to compute exactly.

### SparsePR — probe-fitted residual repair

*Partition the Support, Reconstruct the Residual: Training-Free Sparse
Attention for Video Generation and World Models* (arXiv 2608.18484; code
Apache-2.0 at `PardisTaghavi/SparsePR`).

The sparse output is hard-dropped. A few **probe rows** (64 by default)
get exact dense attention; the residual `R = O_dense − O_sparse` on those
rows fits a weighted ridge regression from the standardized sparse output,
projected on the top-16 principal residual directions, which then predicts
the residual on every other row. The fit happens on every attention call,
per (batch, head); nothing is calibrated offline. The paper's ablation
attributes most of its quality gain to this repair step.

## What `mlx-arsenal` ships

`mlx_arsenal.attention`:

| Function | Role |
|---|---|
| `centroid_compensated_attention` | SVG-EAR compensation for caller-given clusters and a cluster-level `0 / -inf` block mask |
| `probe_residual_correction` | SparsePR repair of any approximate attention output |
| `select_probe_rows` | Head-agnostic probe choice, round-robin over query clusters, with SparsePR's `\|G_a\| / m_a` weights |
| `tile_labels` | Cluster labels matching Sliding Tile Attention tiles |

### No LSE needed

`mx.fast.scaled_dot_product_attention` does not return the log-sum-exp,
which the paper's merge relies on. The identity above removes the need:
the compensated output equals one SDPA call over the extended keys
`[K; K̄]`, values `[V; V̄]`, with an additive `(Sq, Sk + Ck)` mask that is
`0 / -inf` on token columns (kept / skipped block) and `log n_c / -inf` on
centroid columns (skipped / kept block).

### Dense by design

MLX has no block-sparse attention kernel, and ADR-0001 keeps writing one
out of scope. Everything here runs dense — the compensated call costs
slightly **more** than dense attention. These functions are quality tools:
measure what a sparse pattern costs in a port, with and without
compensation, and serve as the numerical reference for a future kernel.

### Recipe: STA with both repairs

```python
import mlx.core as mx
from mlx_arsenal.attention import (
    centroid_compensated_attention, probe_residual_correction,
    select_probe_rows, sliding_tile_block_mask, tile_labels,
)

tile = (tt, th, tw)
labels = tile_labels(T, H, W, tile=tile)
block_mask = sliding_tile_block_mask(T // tt, H // th, W // tw, tile=(1, 1, 1), window=window)

out = centroid_compensated_attention(q, k, v, q_labels=labels, k_labels=labels, block_mask=block_mask)

probe_idx, weights = select_probe_rows(labels, 64)
o_probe = mx.fast.scaled_dot_product_attention(q[:, :, probe_idx], k, v, scale=q.shape[-1] ** -0.5)
out = probe_residual_correction(out, o_probe, probe_idx, weights=weights)
```

Evaluating `sliding_tile_block_mask` at tile resolution gives exactly the
cluster-level mask: expanding it through `tile_labels` reproduces the
token-level STA mask (pinned by a test).

## Deviations from the papers

- **Clusters are inputs.** Both papers cluster Q and K with k-means (SVG2
  semantic permutation, or SparsePR's response-coupled embedding) and warm
  start it across denoising steps. That state belongs to the caller's loop
  (see the doctrine: the caller orchestrates); pass any integer labels.
- **Probe selection** takes the middle-out member of each query cluster
  instead of the row nearest to the cluster centroid, so one `probe_idx`
  serves all heads without needing `q`.
- **Probe weights are normalized** to sum to 1, so `ridge` does not scale
  with the probe count. The reference code's blend factor and norm cap
  (disabled by default there) are not exposed.
- **SparsePR on top of centroid compensation** is allowed; the paper
  applies its repair to a hard-dropped output.

## Out of scope, and why

- **Error-aware routing** (SVG-EAR's block selector): a policy on top of
  the primitive, cheap for a caller to add once a port needs it.
- **k-means / cluster permutation**: stateful across steps, see above.
- **GQA**: per-head labels are ambiguous with shared KV heads.
- **A Metal block-sparse kernel**: a project of its own (ADR-0001).

## Reported results (from the papers)

| Method | Model | Density | Speedup | PSNR / LPIPS |
|---|---|---|---|---|
| SVG-EAR | HunyuanVideo-13B | 22.2% | 1.93× | 31.04 / 0.092 |
| SVG-EAR | Wan2.2-14B T2V 720p | 26.0% | 1.59× | 25.00 / 0.153 |
| SparsePR | HunyuanVideo | 21.9% | 2.61× | 31.84 / 0.087 |
| SparsePR | Wan2.2-I2V-A14B | 22.0% | 1.80× | 30.66 / 0.044 |

Speedups are CUDA figures with custom kernels; they do not carry over to
this dense implementation.

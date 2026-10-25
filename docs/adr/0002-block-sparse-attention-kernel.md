# ADR-0002: Block-sparse attention contract and gated Metal kernel

- **Status**: accepted
- **Date**: 2026-10-23
- **Stacks affected**: `python` / `mlx_arsenal.attention`
- **Amends**: [ADR-0001](0001-sparse-attention-roadmap.md) (kernel layer
  no longer out of scope, for this one kernel)

## Context

ADR-0001 shipped the sparse-attention primitives (static video masks,
head profiling, attention caches, CFG skip, window residual, permutation)
and kept "the kernel layer (Metal block-sparse)" out of scope. Since then
`mlx_arsenal.attention` has also gained block estimators
(`antidiagonal_block_scores`, `minmax_block_scores`,
`radius_bounded_block_scores`), block selectors (`top_p_block_mask`,
`top_k_block_mask`, `relative_block_mask`) and a dense reference for
centroid compensation (`centroid_compensated_attention`). MLX core still
has no block-sparse attention kernel and exposes no log-sum-exp from
`mx.fast.scaled_dot_product_attention` (MLX 0.32.4). Every mask in this
library therefore runs dense: it buys accuracy at a given density, never
wall-clock time.

ADR-0001's exit clause anticipated this: "if a Metal block-sparse kernel
becomes available … step 6 (permutation) must be validated against that
kernel's effective block granularity". Two such kernels now exist:

- **mlx-mfa** (MIT, PyPI) ships a block-sparse forward, about 3.85× over
  masked SDPA at 30% density. It has a single maintainer and was pinned
  to `mlx <= 0.32.2` in September 2026. `docs/research/head-mask-reuse.md`
  already documents it as a third-party option.
- **ltx-2-mlx** (MIT, our port) ships Sol-Attn routing in
  `ltx_core_mlx/model/transformer/sparse_attention.py`. It runs MLX's own
  steel flash-attention loop (MIT, Apple) over the routed 64-token key
  blocks only, then adds one centroid tile per skipped block. On an
  M1 Max with the LTX-2.5 q8 pack in float16, stage-2 steps are 21%
  faster at 17,856 tokens and 36% faster at 32,640, at about 30–32 dB
  PSNR against the dense render. Routing follows NVIDIA's Sol-Attn
  (arXiv 2607.24027; reference in NVlabs/Sana, Apache-2.0 at repo level).

The ltx-2-mlx kernel has known fragilities:

- It reads **private steel headers** from the installed `mlx` wheel and
  compiles them through `mx.fast.metal_kernel`. Any MLX release may break
  it. It probes itself against dense SDPA once per dtype and falls back to
  the dense kernel on failure.
- It is measured on **one chip (M1 Max), one head dim (128), one model**.
  Its tile shape (BQ=32, BK=16, 4 simdgroups) is the one MLX picks for
  head dim 128 on GPUs without neural accelerators. On chips where MLX's
  dense SDPA uses the neural accelerators, the sparse kernel may lose to
  dense attention.
- `mx.fast.metal_kernel` output is not guaranteed identical across chips
  and compilers (mlx#4603, macOS 27).

The math the kernel computes is already in this library.
`centroid_compensated_attention`, with contiguous token blocks as
clusters, is exactly Sol's length-weighted correction: a skipped key
block enters the softmax once as its key centroid with a `log L` bias and
its value mean (`exp(s + ln L) · mean(V) = exp(s) · sum(V)`).

## Decision

Option **B, gated**: define a block-sparse attention contract in
`mlx_arsenal.attention` now, with the dense path as reference and
fallback. Move a Metal kernel behind that contract only after four gates
pass. The contract is the durable part; the kernel is replaceable.

### Phase 1 — contract and dense backend (no kernel)

```python
def block_sparse_attention(
    q: mx.array,              # (B, H, Sq, D)
    k: mx.array,              # (B, H, Sk, D)
    v: mx.array,              # (B, H, Sk, Dv)
    *,
    block_mask: mx.array,     # (B|1, H|1, ceil(Sq/b), ceil(Sk/b))
    block_size: int = 64,
    compensate: bool = True,
    scale: float | None = None,
) -> mx.array:                # (B, H, Sq, Dv)
```

- **Route format.** `block_mask` is the additive block mask that
  `top_p_block_mask`, `top_k_block_mask` and `relative_block_mask` already
  return: `0` keeps a block, anything else skips it. No new route type.
  Static video masks reach the same format at tile resolution, as the
  `tile_labels` recipe already shows.
- **Blocks.** Tokens form contiguous blocks of `block_size` along each
  sequence, in the order the caller passes them; the last block may be
  partial. Semantic clustering is the caller's job, through
  `block_contiguous_permutation` before the call and its inverse after.
- **Skipped blocks.** With `compensate=True`, each skipped key block
  contributes its key centroid with a `log L` bias and its value mean
  (SVG-EAR / Sol). With `compensate=False` it is dropped. A query row
  with every block skipped then has an empty softmax; this is the
  caller's error, documented but not checked.
- **No host synchronization.** Only shapes are validated. Mask values are
  not read on the host, so the call stays lazy and can sit inside a
  compiled forward. This differs on purpose from
  `centroid_compensated_attention`, which validates its mask values.
- **Dense backend.** `compensate=True` computes what
  `centroid_compensated_attention` computes with
  `q_labels = arange(Sq) // b` and `k_labels = arange(Sk) // b`, through
  an internal core without its host-side label and mask checks (labels
  built from `arange` are valid by construction). The public function
  keeps its checks and becomes a thin wrapper over that core.
  `compensate=False` calls `mx.fast.scaled_dot_product_attention` with
  the mask expanded to token resolution. This path is the numerical
  reference for any kernel.
- **Gate function.** `block_sparse_kernel_available(dtype, head_dim) ->
  bool` is public and returns `False` in phase 1. Ports can write their
  dispatch logic once.
- **Router.** Sol's "diag" estimator joins `dynamic_masks` as one more
  selector: a key block is kept when its centroid score exceeds
  `mean + τ · std` of the query block's centroid scores, plus a
  `|i - j| <= band` neighbour band. It returns the same additive mask.
  Model-specific rules (LTX's same-position blocks one latent frame away,
  dense block 0, per-step `τ` tables, the minimum token count) stay in the
  port; the caller unions extra masks with `mx.maximum`.

### Phase 2 — Metal backend, after four gates

When `block_sparse_kernel_available(q.dtype, D)` is true and the call
fits the kernel (`Sq == Sk`, `block_size == 64`), `block_sparse_attention`
runs the Metal kernel; otherwise it runs the dense backend. The kernel
source is adapted from ltx-2-mlx with the steel copyright and the Sol
attribution kept. It returns a transposed view, so callers get
`(B, H, Sq, Dv)` with no copy. Inputs need a contiguous head dim only.

The kernel moves only when all four gates pass:

1. **Hardware and shapes.** It has been parity-probed and benchmarked on
   at least two chip generations, including one where MLX's dense SDPA
   uses the neural accelerators. It has been tested at head dim 64 and
   128, in float16 and bfloat16. The minimum token count above which it
   beats dense SDPA is documented per chip; below it, the dense path runs.
2. **Second consumer.** A port other than ltx-2-mlx drives it through an
   arsenal router and reports wall-clock time and quality against dense
   attention. Candidates: Matrix-Game 3's Wan2.2-5B self-attention, or
   ERNIE-Image at a resolution where the sequence is long enough.
3. **MLX coupling.** A CI test builds the kernel against the installed
   `mlx` and asserts that the probe passes. CI installs the newest `mlx`
   allowed by `pyproject.toml`, so a header change fails the next run
   (including the Dependabot bump) instead of silently degrading users to
   the dense path. At runtime a failed probe still means dense fallback
   with a one-time warning, never a wrong output.
4. **Licence.** The `sol-engine` files of NVlabs/Sana that the routing
   follows are checked file by file, and the Apache-2.0 attribution is
   recorded in the module docstring.

If the gates are not met, the kernel stays in ltx-2-mlx (option A). The
port then consumes the arsenal contract and routers and keeps its kernel
as a local backend. No public API is lost.

### Scope rule for other kernels

This ADR does not open a general custom-kernel programme. A Metal kernel
enters `mlx-arsenal` only when:

- MLX core cannot express the operation without materializing a large
  intermediate or issuing many launches (sparse attention, irregular
  scatter such as the existing `rasterize` binning and tile kernels);
- profiling of a real port shows that the operation dominates the step;
- a dense reference stays in the tree as fallback and test oracle.

Hand-written replacements for operations MLX already ships tuned per
chip (GEMM, dense SDPA, normalization, RoPE, quantized matmul) are out of
scope.

## Consequences

- Every shipped estimator and static mask becomes a wall-clock lever on
  Apple Silicon once phase 2 lands. Until then, phase 1 still gives ports
  one call site and a reference to test their own kernels against.
- ADR-0001's exit clause is discharged: the permutation and the masks
  are validated against a 64-token block granularity by the phase-1
  integration tests.
- Phase 2 adds a maintenance cost on every MLX release, and CI only
  covers the macos-14 runner's chip. The second-chip check stays manual
  and is recorded in the benchmark notes.
- Once phase 2 lands, ltx-2-mlx deletes its copy and imports the arsenal
  kernel.

## Alternatives considered

- **A. Keep the kernel in the port.** Amend ADR-0001 only to record the
  integration test. Rejected as the target because every other port would
  re-copy the kernel, but kept as the fallback if the gates fail.
- **C. Separate package, duck-typed** like mlx-mfa. Rejected: it moves the
  MLX-coupling problem into a second release train without removing it,
  and the dense reference and routers would still live here.
- **Depend on mlx-mfa.** Rejected: single maintainer and an upper bound
  on `mlx`. It stays documented as a third-party option.
- **Contribute block-sparse SDPA to MLX core.** This is the right end
  state, but not under our control. A sliding-window SDPA is proposed
  upstream (mlx#4734); nothing block-sparse is. If MLX core ships it,
  see the exit clause.
- **Ship the kernel now, without gates.** Rejected: one chip, one head
  dim and one model are not enough evidence for a library default, and
  private-header coupling needs a CI tripwire before release.

## Exit / revision

- If MLX core ships a block-sparse SDPA, or exposes log-sum-exp so that
  block attention can be composed from public kernels, the Metal backend
  switches to it and the private-header kernel is deleted. The contract
  does not change.
- If the kernel probe fails on two consecutive MLX releases, the Metal
  backend is removed and the contract runs dense until the kernel is
  repaired.
- If phase 2's gates are still open at the 2027-04 watch-list review,
  this ADR is revised to option A.

## References

- [Sol-Attn (arXiv 2607.24027)](https://arxiv.org/abs/2607.24027) and
  [NVlabs/Sana](https://github.com/NVlabs/Sana) (`sol-engine` branch)
- [SVG-EAR (arXiv 2603.08982)](https://arxiv.org/abs/2603.08982) — the
  centroid compensation behind `centroid_compensated_attention`
- [mlx-mfa](https://pypi.org/project/mlx-mfa/)
- MLX steel attention: `mlx/backend/metal/kernels/steel/attn/` (MIT)
- [ADR-0001](0001-sparse-attention-roadmap.md)
